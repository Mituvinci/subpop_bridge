"""
s20_subpop_bridge_merge_vit_erm.py - BRIDGE merge with ERM-as-third-specialist.

Same sub-population shift setup as s20_subpop_bridge_merge_vit.py, but the
merge includes the ERM model as an extra specialist (alongside the per-
group BRIDGE specialists). The ERM visual encoder provides a third task
vector; its head is still the reference head.

Motivation (2026-04-22):
    The sibling s20 run on CelebA showed a level-down artifact: BRIDGE
    lost to GroupDRO on BOTH Avg AND worst-group accuracy (54 vs 66 WG,
    74 vs 95 Avg). Diagnosis: the two per-group specialists individually
    overfit their subgroup and the merge cannot recover the joint
    accuracy without pulling some mass from a globally-trained model.
    The ERM model (trained on the full mixture) provides that pull.

Merge (per block):
    visual_merged[k] = visual_zero[k]
                     + sum_g lambda_{g, block(k)} * (visual_spec_g[k] - visual_zero[k])
                     + lambda_{erm, block(k)} * (visual_erm[k] - visual_zero[k])
    head_merged      = head_erm                                   (fixed)

Lambdas are per-(specialist, block), with specialist index in
{0, ..., N_groups-1, ERM}. For derm (2 groups) with ~13 blocks that is
(2+1) * 13 = 39 lambdas; for CelebA/Waterbirds (4 groups) it is
(4+1) * 13 = 65 lambdas. Optuna TPE searches the [0, 1] box.

All four objective modes from the sibling (uniform, dro, combined, direct)
are available. Default is combined.

Inputs: identical to s20_subpop_bridge_merge_vit.py. The ERM checkpoint
is the same one whose head is reused; we just additionally read its
visual state dict as an extra task vector.

Output:
    results/s20_subpop_bridge_erm_{dataset}/{objective}/{timestamp}/
        trials.jsonl, summary.json, merged_model.pt, eval_test.json

Usage:
    python experiments/s20_subpop_bridge_merge_vit_erm.py --dataset celeba \\
        --n-trials 150
    python experiments/s20_subpop_bridge_merge_vit_erm.py --dataset derm \\
        --n-trials 3 --smoke

Author: BRIDGE / NeurIPS 2026 submission.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.subpop_common import (
    SubpopImageDataset,
    SubpopTextDataset,
    SubpopManifest,
    collect_predictions,
    collect_preds_and_losses,
    dro_weighted_loss,
    is_text_dataset,
    load_manifest,
    load_text_tokens,
    per_group_loss,
    summarize_split,
)
from lib.optuna_resume import create_or_resume_study, remaining_trials
from lib.backbones import build_backbone, block_regex as backbone_block_regex
from train_erm import CLIPVisualWithHead

import optuna


DEFAULT_S19_ROOT = PROJECT_ROOT / "results" / "s19_subpop_vit"

# Reserved specialist index for the ERM task vector. Chosen above all
# valid group indices so it never collides with a real group.
ERM_IDX = -1


# -------------------------------------------------------------------------
# Checkpoint discovery
# -------------------------------------------------------------------------

def _latest_timestamp_dir(parent: Path) -> Path:
    candidates = [p for p in parent.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No timestamped runs under {parent}")
    return max(candidates, key=lambda p: p.name)


def find_erm_checkpoint(dataset: str, s19_root: Path) -> Path:
    base = s19_root / dataset / "erm"
    if not base.exists():
        raise FileNotFoundError(
            f"No ERM checkpoint for {dataset} under {base}. "
            f"Run: python experiments/s19_subpop_train_vit.py "
            f"--dataset {dataset} --method erm"
        )
    return _latest_timestamp_dir(base) / "model.pt"


def find_specialist_checkpoints(dataset: str, num_groups: int,
                                 s19_root: Path) -> Dict[int, Path]:
    base = s19_root / dataset / "bridge_specialist"
    out: Dict[int, Path] = {}
    for g in range(num_groups):
        group_dir = base / f"group_{g}"
        if not group_dir.exists():
            raise FileNotFoundError(
                f"Missing BRIDGE specialist for {dataset} group {g} "
                f"under {group_dir}."
            )
        out[g] = _latest_timestamp_dir(group_dir) / "model.pt"
    return out


# -------------------------------------------------------------------------
# Block-grouping of CLIP visual keys
# -------------------------------------------------------------------------

_BLOCK_RE = re.compile(r"(transformer\.resblocks\.\d+)")


def _set_block_regex(pattern: str) -> None:
    """Override the default ViT block regex (used when --backbone=resnet50)."""
    global _BLOCK_RE
    _BLOCK_RE = re.compile(f"({pattern})")


def block_of(key: str) -> str:
    m = _BLOCK_RE.search(key)
    return m.group(1) if m else "other"


def block_buckets(sample_state: Dict[str, torch.Tensor]) -> List[str]:
    seen: List[str] = []
    for k in sample_state.keys():
        b = block_of(k)
        if b not in seen:
            seen.append(b)
    return seen


# -------------------------------------------------------------------------
# Merge (per-specialist, per-block lambdas, including ERM as specialist)
# -------------------------------------------------------------------------

def _is_bn_running_stat(key: str) -> bool:
    """BatchNorm running statistics that must NOT be task-vector merged.

    Linear task-vector arithmetic on `running_var` can produce negative
    values (since lambdas can sum >1 across specialists), and BatchNorm
    then computes sqrt(var+eps) -> NaN. `num_batches_tracked` is an
    integer counter that breaks similarly. ViT-CLIP has no BN so this
    is a no-op there; only RN50 hits this path.
    """
    return (key.endswith("running_mean")
            or key.endswith("running_var")
            or key.endswith("num_batches_tracked"))


def merged_visual_state(zero_state: Dict[str, torch.Tensor],
                         spec_states: Dict[int, Dict[str, torch.Tensor]],
                         lambdas: Dict[Tuple[int, str], float]) -> Dict[str, torch.Tensor]:
    """Return merged visual state dict.

    spec_states maps specialist_idx -> visual_state_dict, where
    specialist_idx is in {0, ..., num_groups - 1, ERM_IDX}.
    lambdas[(spec_idx, block_id)] is that specialist's weight at that
    block, applied to the task-vector (specialist - zero_shot).
    """
    merged: Dict[str, torch.Tensor] = {}
    for k, v0 in zero_state.items():
        if _is_bn_running_stat(k):
            # Keep zero_shot's running stats verbatim. See _is_bn_running_stat.
            merged[k] = v0.clone()
            continue
        b = block_of(k)
        acc = v0.clone()
        for s_idx, spec_state in spec_states.items():
            lam = lambdas.get((s_idx, b), 0.0)
            if lam == 0.0:
                continue
            if k not in spec_state:
                continue
            acc = acc + lam * (spec_state[k] - v0)
        merged[k] = acc
    return merged


# -------------------------------------------------------------------------
# Optuna objective
# -------------------------------------------------------------------------

def _spec_key(s_idx: int, b: str) -> str:
    """Flat parameter name for Optuna. ERM uses 'erm__...' to avoid
    collision with valid integer group indices."""
    prefix = "erm" if s_idx == ERM_IDX else f"g{s_idx}"
    return f"{prefix}__{b}"


def build_objective(model_shell: CLIPVisualWithHead,
                     zero_state: Dict[str, torch.Tensor],
                     spec_states: Dict[int, Dict[str, torch.Tensor]],
                     block_ids: List[str],
                     specialist_ids: List[int],
                     val_loader, device: str,
                     manifest: SubpopManifest,
                     penalty: float, t_mean: float, t_worst: float,
                     log_jsonl: Path,
                     objective_mode: str = "combined",
                     dro_step_size: float = 0.01,
                     delta: float = 1.0):
    """Same four modes as s20_subpop_bridge_merge_vit. Lambdas are
    searched across all specialist_ids (includes ERM_IDX)."""
    def _objective(trial):
        trial_t0 = time.time()
        lambdas: Dict[Tuple[int, str], float] = {}
        for s_idx in specialist_ids:
            for b in block_ids:
                lambdas[(s_idx, b)] = trial.suggest_float(
                    _spec_key(s_idx, b), 0.0, 1.0)

        merged = merged_visual_state(zero_state, spec_states, lambdas)
        model_shell.visual.load_state_dict(merged, strict=False)

        if objective_mode == "combined":
            preds, labels, groups, losses = collect_preds_and_losses(
                model_shell, val_loader, device)
            summary = summarize_split(preds, labels, groups,
                                       manifest.num_groups, manifest.num_labels)
            pg_loss = per_group_loss(losses, groups, manifest.num_groups)
            dro_loss, q = dro_weighted_loss(pg_loss, step_size=dro_step_size)
            gap = summary["group_gap_acc"]
            overall = summary["overall_acc"]
            worst = summary["worst_group_acc"]["value"]
            mmf = max(0.0, t_mean - overall)
            wmf = max(0.0, t_worst - worst)
            obj = gap + dro_loss + penalty * mmf + 0.5 * penalty * wmf
            components = {
                "mg_gap": gap,
                "dro_loss": dro_loss,
                "mmf": mmf,
                "wmf": wmf,
                "per_group_ce": {str(k): v for k, v in pg_loss.items()},
                "q": {str(k): v for k, v in q.items()},
            }
        elif objective_mode == "direct":
            preds, labels, groups = collect_predictions(model_shell,
                                                          val_loader, device)
            summary = summarize_split(preds, labels, groups,
                                       manifest.num_groups, manifest.num_labels)
            gap = summary["group_gap_acc"]
            overall = summary["overall_acc"]
            obj = gap - delta * overall
            components = {
                "mg_gap": gap,
                "overall_acc": overall,
                "delta": delta,
            }
        elif objective_mode == "dro":
            preds, labels, groups, losses = collect_preds_and_losses(
                model_shell, val_loader, device)
            summary = summarize_split(preds, labels, groups,
                                       manifest.num_groups, manifest.num_labels)
            pg_loss = per_group_loss(losses, groups, manifest.num_groups)
            dro_loss, q = dro_weighted_loss(pg_loss, step_size=dro_step_size)
            obj = dro_loss
            components = {
                "per_group_ce": {str(k): v for k, v in pg_loss.items()},
                "q": {str(k): v for k, v in q.items()},
                "dro_loss": dro_loss,
            }
        else:  # uniform
            preds, labels, groups = collect_predictions(model_shell,
                                                          val_loader, device)
            summary = summarize_split(preds, labels, groups,
                                       manifest.num_groups, manifest.num_labels)
            gap = summary["group_gap_acc"]
            overall = summary["overall_acc"]
            worst = summary["worst_group_acc"]["value"]
            mmf = max(0.0, t_mean - overall)
            wmf = max(0.0, t_worst - worst)
            obj = gap + penalty * mmf + 0.5 * penalty * wmf
            components = {"mg_gap": gap, "mmf": mmf, "wmf": wmf}

        trial_elapsed = time.time() - trial_t0
        record = {
            "trial": trial.number,
            "objective_mode": objective_mode,
            "include_erm": True,
            "lambdas": {_spec_key(s, b): lambdas[(s, b)]
                         for s in specialist_ids for b in block_ids},
            "val_summary": summary,
            "objective": obj,
            "components": components,
            "trial_elapsed_seconds": round(trial_elapsed, 2),
        }
        with open(log_jsonl, "a") as f:
            f.write(json.dumps(record) + "\n")
        return obj

    return _objective


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True,
                    choices=["derm", "celeba", "waterbirds",
                             "metashift", "imagenetbg", "living17",
                             "chexpert", "civilcomments", "multinli"])
    ap.add_argument("--backbone", default="clip_vit_b32",
                    choices=["clip_vit_b32", "resnet50", "bert-base-uncased"],
                    help="Must match the backbone used in the s19 "
                         "specialists being merged.")
    ap.add_argument("--n-trials", type=int, default=150)
    ap.add_argument("--objective", choices=["uniform", "dro", "combined", "direct"],
                    default="combined",
                    help="Same modes as sibling s20_subpop_bridge_merge_vit. "
                         "Default 'combined' (gap+DRO+MMF+WMF).")
    ap.add_argument("--delta", type=float, default=1.0,
                    help="Weight on overall_acc for --objective=direct.")
    ap.add_argument("--dro-step-size", type=float, default=0.01)
    ap.add_argument("--penalty", type=float, default=10.0)
    ap.add_argument("--t-mean", type=float, default=0.60)
    ap.add_argument("--t-worst", type=float, default=0.40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--s19-root", type=Path, default=DEFAULT_S19_ROOT)
    ap.add_argument("--erm-path", type=Path, default=None,
                    help="Override auto-discovered ERM model.pt")
    ap.add_argument("--out-dir", type=Path,
                    default=PROJECT_ROOT / "results")
    ap.add_argument("--manifest-dir", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="Use a tiny val subset for the objective.")
    ap.add_argument("--resume-dir", type=Path, default=None)
    ap.add_argument("--resume-from", type=Path, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Backbone-specific block regex (ViT resblocks vs ResNet bottlenecks).
    _set_block_regex(backbone_block_regex(args.backbone))

    # --- manifest + checkpoints ---------------------------------------
    manifest: SubpopManifest = load_manifest(args.dataset, args.manifest_dir)

    erm_path = args.erm_path or find_erm_checkpoint(args.dataset, args.s19_root)
    spec_paths = find_specialist_checkpoints(args.dataset, manifest.num_groups,
                                               args.s19_root)
    print(f"[s20-erm] dataset={args.dataset}")
    print(f"[s20-erm] erm checkpoint (HEAD + visual task-vector): {erm_path}")
    for g, p in spec_paths.items():
        print(f"[s20-erm] specialist g={g}: {p}")

    shell = build_backbone(args.backbone,
                            num_labels=manifest.num_labels).to(device)
    erm_state = torch.load(erm_path, map_location=device)
    shell.head.load_state_dict(erm_state["head"])
    if "visual" not in erm_state:
        raise KeyError(
            f"ERM checkpoint {erm_path} has no 'visual' key. "
            f"Cannot use as third specialist.")
    zero_state = {k: v.detach().clone()
                   for k, v in shell.visual.state_dict().items()}

    spec_states: Dict[int, Dict[str, torch.Tensor]] = {}
    for g, p in spec_paths.items():
        spec_states[g] = torch.load(p, map_location=device)["visual"]
    spec_states[ERM_IDX] = {k: v.detach().clone()
                              for k, v in erm_state["visual"].items()}

    specialist_ids: List[int] = list(range(manifest.num_groups)) + [ERM_IDX]
    block_ids = block_buckets(zero_state)
    total_lambdas = len(block_ids) * len(specialist_ids)
    print(f"[s20-erm] {len(block_ids)} block buckets, "
          f"{len(specialist_ids)} specialists ({manifest.num_groups} group + 1 ERM) "
          f"-> {total_lambdas} lambdas total")

    # --- val + test loaders -------------------------------------------
    val_df = manifest.split("val")
    test_df = manifest.split("test")
    if args.smoke:
        val_df = val_df.head(min(len(val_df), 256)).reset_index(drop=True)

    if is_text_dataset(args.dataset):
        all_tokens = load_text_tokens(args.dataset)
        full_df = manifest.df.copy()
        val_mask = full_df["split"] == "val"
        test_mask = full_df["split"] == "test"
        val_ds = SubpopTextDataset(full_df[val_mask].reset_index(drop=True),
                                   all_tokens[val_mask.values])
        test_ds = SubpopTextDataset(full_df[test_mask].reset_index(drop=True),
                                    all_tokens[test_mask.values])
        if args.smoke:
            val_ds = SubpopTextDataset(
                val_ds.df.head(min(len(val_ds), 256)).reset_index(drop=True),
                val_ds.token_data[:min(len(val_ds), 256)])
    else:
        val_ds = SubpopImageDataset(val_df, shell.val_preprocess)
        test_ds = SubpopImageDataset(test_df, shell.val_preprocess)

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    # --- output dir ---------------------------------------------------
    if args.resume_dir is not None:
        out_dir = Path(args.resume_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = out_dir.name
        print(f"[s20-erm] --resume-dir set: reusing {out_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = (Path(args.out_dir)
                   / f"s20_subpop_bridge_erm_{args.dataset}"
                   / args.objective / f"{timestamp}_s{args.seed}")
        out_dir.mkdir(parents=True, exist_ok=True)
    log_jsonl = out_dir / "trials.jsonl"

    # --- Optuna -------------------------------------------------------
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = create_or_resume_study(
        db_path=out_dir / "study.db",
        study_name=f"s20_bridge_erm_{args.dataset}_{args.objective}",
        direction="minimize",
        sampler=sampler,
        resume_from_jsonl=args.resume_from,
    )
    to_run = remaining_trials(study, args.n_trials)
    print(f"[s20-erm] target total trials: {args.n_trials} | "
          f"already in study: {len(study.get_trials(deepcopy=False))} | "
          f"new trials to run: {to_run}")
    objective = build_objective(
        shell, zero_state, spec_states, block_ids, specialist_ids,
        val_loader, device, manifest,
        args.penalty, args.t_mean, args.t_worst, log_jsonl,
        objective_mode=args.objective,
        dro_step_size=args.dro_step_size,
        delta=args.delta,
    )
    t0 = time.time()
    if to_run > 0:
        study.optimize(objective, n_trials=to_run, show_progress_bar=False)
    elapsed = time.time() - t0

    # --- lock in best and evaluate on test ----------------------------
    best = study.best_trial
    best_lambdas_flat = best.params
    best_lambdas: Dict[Tuple[int, str], float] = {}
    for key, val in best_lambdas_flat.items():
        prefix, b = key.split("__", 1)
        if prefix == "erm":
            s_idx = ERM_IDX
        else:
            s_idx = int(prefix[1:])
        best_lambdas[(s_idx, b)] = val

    merged = merged_visual_state(zero_state, spec_states, best_lambdas)
    shell.visual.load_state_dict(merged, strict=False)
    preds, labels, groups = collect_predictions(shell, test_loader, device)
    test_summary = summarize_split(preds, labels, groups,
                                    manifest.num_groups, manifest.num_labels)
    test_summary["num_examples"] = int(len(preds))

    # --- save ---------------------------------------------------------
    summary = {
        "dataset": args.dataset,
        "objective_mode": args.objective,
        "include_erm": True,
        "best_trial": best.number,
        "best_objective": best.value,
        "best_lambdas": best_lambdas_flat,
        "n_trials": args.n_trials,
        "total_lambdas": total_lambdas,
        "elapsed_seconds": elapsed,
        "avg_trial_seconds": round(elapsed / max(to_run, 1), 2),
        "hyperparams": {
            "penalty": args.penalty,
            "t_mean": args.t_mean,
            "t_worst": args.t_worst,
            "dro_step_size": args.dro_step_size,
            "delta": args.delta,
            "seed": args.seed,
        },
        "timestamp": timestamp,
        "num_groups": manifest.num_groups,
        "num_labels": manifest.num_labels,
        "n_blocks": len(block_ids),
        "block_ids": block_ids,
        "specialist_ids": [("erm" if s == ERM_IDX else s) for s in specialist_ids],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "eval_test.json", "w") as f:
        json.dump(test_summary, f, indent=2)
    shell.save(out_dir / "merged_model.pt")

    print(f"[s20-erm] done. best_obj={best.value:.4f} in {elapsed:.0f}s")
    print(f"[s20-erm] test: overall_acc={test_summary['overall_acc']:.4f} "
          f"worst_group_acc={test_summary['worst_group_acc']['value']:.4f} "
          f"gap={test_summary['group_gap_acc']:.4f}")
    print(f"[s20-erm] out_dir: {out_dir}")


if __name__ == "__main__":
    main()
