"""s26g -- AG-BRIDGE: Covariance-regularized 3-head BRIDGE for AG datasets.

For Attribute Generalization datasets (ImageNetBG, Living17) where group
annotations are unavailable, BRIDGE replaces role-based head specialization
(rho) with prototype diversity enforcement via covariance regularization.

Three heads are trained SEQUENTIALLY on the class-balanced val set:
  - Head 0: IsoMax+ loss only (no covariance penalty)
  - Head 1: IsoMax+ loss + covariance penalty against frozen head 0
  - Head 2: IsoMax+ loss + covariance penalty against frozen heads 0,1

This matches DPE's sequential stage training. Each head decorrelates from
frozen previous heads rather than a moving target.

Model selection: last epoch (no val-based early stopping), matching DPE's
-ec last setting for AG datasets.

Usage:
    python experiments/s26g_ag_bridge_cov.py \
        --dataset living17 \
        --backbone resnet50 \
        --merged-ckpt results/.../model.pt \
        --cov-reg 1e5 --lr 5e-5 --epochs 30

    python experiments/s26g_ag_bridge_cov.py \
        --dataset imagenetbg \
        --backbone resnet50 \
        --merged-ckpt results/.../model_best_wga.pt \
        --cov-reg 1e5 --lr 1e-3 --epochs 20
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR
sys.path.insert(0, str(THIS_DIR))

from lib.group_prototype_head import GroupPrototypeHead
from lib.isomax_head import select_balanced_indices
from lib.backbones import build_backbone, embed_dim as backbone_embed_dim
from lib.subpop_common import (
    SubpopImageDataset,
    load_manifest,
    summarize_split,
)


def instance_normalize(feats: torch.Tensor) -> torch.Tensor:
    mean = feats.mean(dim=-1, keepdim=True)
    std = feats.std(dim=-1, keepdim=True) + 1e-6
    return (feats - mean) / std


def dpe_cov_loss(current_protos: torch.Tensor,
                 frozen_protos_list: list[torch.Tensor]) -> torch.Tensor:
    """Covariance loss matching DPE exactly.

    current_protos: (C, D) -- current head's raw prototypes
    frozen_protos_list: list of (C, D) tensors from frozen previous heads
    Returns scalar penalty.
    """
    # Stack: (C, 1 + num_prev, D)
    all_p = torch.cat(
        [current_protos[:, None]] + [fp[:, None] for fp in frozen_protos_list],
        dim=1,
    )
    n_pro, n_dim = all_p.shape[1:]
    # Raw covariance matrix per class: (C, n_pro, n_pro)
    cov = torch.einsum("ijk,ilk->ijl", all_p, all_p) / (n_dim - 1)
    # Penalize covariance between current (index 0) and all previous (1:)
    return torch.abs(cov[:, 0, 1:].sum(1).div(n_pro).mean())


class AGBridgeEnsemble(nn.Module):
    """Three covariance-diversified heads with confidence routing."""

    def __init__(self, num_features: int, num_labels: int,
                 temperature: float = 1.0):
        super().__init__()
        self.num_heads = 3
        self.num_labels = num_labels
        self.heads = nn.ModuleList([
            GroupPrototypeHead(num_features, num_labels, num_groups=1,
                               inference="max", temperature=temperature)
            for _ in range(3)
        ])

    def forward(self, features: torch.Tensor,
                mode: str = "confidence") -> torch.Tensor:
        per_head = torch.stack([h(features) for h in self.heads], dim=0)
        probs = F.softmax(per_head, dim=-1)  # (3, B, C)

        if mode == "mean":
            combined = probs.mean(dim=0)
        else:
            conf = probs.max(dim=-1).values  # (3, B)
            gate = F.softmax(conf, dim=0)    # (3, B)
            combined = (probs * gate.unsqueeze(-1)).sum(dim=0)

        return torch.log(combined.clamp_min(1e-12))


def train_sequential_heads(
    ensemble: AGBridgeEnsemble,
    backbone_fn,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    entropic_scale: float,
    wd_weight: float,
    cov_reg: float,
    device: str,
    log_jsonl: Path,
):
    """Train 3 heads sequentially. Each head decorrelates from frozen previous."""
    heads = list(ensemble.heads)
    frozen_protos = []

    for h_idx, head in enumerate(heads):
        print(f"\n--- Head {h_idx}/{len(heads)-1} "
              f"(cov against {len(frozen_protos)} frozen) ---")

        for p in head.parameters():
            p.requires_grad = True
        optimizer = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9)

        for epoch in range(epochs):
            head.train()
            epoch_clf = 0.0
            epoch_cov = 0.0
            n_batches = 0

            for x, y, g in train_loader:
                x, y = x.to(device), y.to(device)
                feats = backbone_fn(x)
                feats = instance_normalize(feats)

                d = head._all_distances(feats)
                flat_idx = y  # num_groups=1, so flat_idx = y
                probs = F.softmax(-entropic_scale * d, dim=1)
                prob_targets = probs[torch.arange(d.size(0), device=device), flat_idx]
                clf_loss = -torch.log(prob_targets.clamp_min(1e-12)).mean()

                P = head.prototypes.unsqueeze(1)
                wd = torch.einsum("ijk,ilk->ijl", P, P).squeeze().mean() * wd_weight

                loss = clf_loss + wd

                if len(frozen_protos) > 0:
                    cov_l = dpe_cov_loss(head.prototypes, frozen_protos)
                    loss = loss + cov_reg * cov_l
                    epoch_cov += cov_l.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_clf += clf_loss.item()
                n_batches += 1

            log_entry = {
                "head": h_idx, "epoch": epoch,
                "clf_loss": epoch_clf / max(n_batches, 1),
                "cov_loss": epoch_cov / max(n_batches, 1),
            }
            with open(log_jsonl, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                print(f"  [head {h_idx}] epoch {epoch}: "
                      f"clf={epoch_clf/max(n_batches,1):.4f} "
                      f"cov={epoch_cov/max(n_batches,1):.6f}")

        # Freeze this head and save its prototypes for next head
        frozen_protos.append(head.prototypes.detach().clone().to(device))
        for p in head.parameters():
            p.requires_grad = False
        print(f"  Head {h_idx} frozen.")

    # Return last-epoch state (no val-based selection, matching DPE -ec last)
    return {k: v.clone().cpu() for k, v in ensemble.state_dict().items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True,
                    choices=["imagenetbg", "living17"])
    ap.add_argument("--backbone", default="resnet50",
                    choices=["clip_vit_b32", "resnet50"])
    ap.add_argument("--merged-ckpt", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--entropic-scale", type=float, default=10.0)
    ap.add_argument("--wd-weight", type=float, default=10.0)
    ap.add_argument("--cov-reg", type=float, default=1e5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load manifest ---
    mf = load_manifest(args.dataset)
    manifest = mf.df
    num_labels = mf.num_labels
    num_groups = mf.num_groups
    print(f"[ag-bridge-cov] {args.dataset}: {num_labels} labels, "
          f"{num_groups} groups, {len(manifest)} rows")
    print(f"[ag-bridge-cov] cov_reg={args.cov_reg}, lr={args.lr}, "
          f"epochs={args.epochs}, es={args.entropic_scale}")
    print(f"[ag-bridge-cov] SEQUENTIAL training, last-epoch selection")

    # --- Build backbone (frozen feature extractor) ---
    full_model = build_backbone(
        args.backbone, num_labels,
        pretrained_path=str(args.merged_ckpt))
    full_model.to(device).eval()
    for p in full_model.parameters():
        p.requires_grad = False
    feat_dim = backbone_embed_dim(args.backbone)

    def backbone_fn(x):
        with torch.no_grad():
            return full_model.visual(x).float()

    # --- Class-balanced val subset for training (same as DPE) ---
    val_mask = manifest["split"] == "val"
    val_df = manifest[val_mask].reset_index(drop=True)
    bal_idx = select_balanced_indices(val_df, "class")
    bal_df = val_df.iloc[bal_idx].reset_index(drop=True)
    print(f"[ag-bridge-cov] class-balanced val subset: {len(bal_df)} rows")

    # --- Datasets ---
    preprocess = full_model.val_preprocess
    train_ds = SubpopImageDataset(bal_df, preprocess=preprocess)

    # Test set
    if args.dataset == "living17":
        test_split = "zs"
    elif args.dataset == "imagenetbg":
        test_split = "mixed_rand"
    else:
        test_split = "test"
    test_mask = manifest["split"] == test_split
    test_df = manifest[test_mask].reset_index(drop=True)
    test_ds = SubpopImageDataset(test_df, preprocess=preprocess)

    if args.smoke:
        train_ds = Subset(train_ds, range(min(200, len(train_ds))))
        test_ds = Subset(test_ds, range(min(200, len(test_ds))))
        args.epochs = 2

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

    # --- Build ensemble ---
    ensemble = AGBridgeEnsemble(feat_dim, num_labels).to(device)
    print(f"[ag-bridge-cov] 3 heads, feat_dim={feat_dim}, "
          f"num_labels={num_labels}")

    # --- Output dir ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (PROJECT_ROOT / "results" / "s26g_ag_bridge_cov" /
               args.dataset /
               f"{ts}_s{args.seed}_cov{args.cov_reg:.0e}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_jsonl = out_dir / "train_log.jsonl"

    # --- Train sequentially ---
    print("\n" + "=" * 60)
    print(f"Training AG-BRIDGE SEQUENTIAL (3 heads, cov_reg={args.cov_reg})")
    print("=" * 60)

    final_state = train_sequential_heads(
        ensemble, backbone_fn, train_loader,
        epochs=args.epochs, lr=args.lr,
        entropic_scale=args.entropic_scale,
        wd_weight=args.wd_weight, cov_reg=args.cov_reg,
        device=device, log_jsonl=log_jsonl,
    )

    # --- Load final state and evaluate on test ---
    ensemble.load_state_dict(final_state)
    ensemble.to(device).eval()

    print("\n" + "=" * 60)
    print("Test Evaluation")
    print("=" * 60)

    # Collect labels/groups once
    labels_all, groups_all = [], []
    for mode in ["confidence", "mean"]:
        preds = []
        labels_all_m, groups_all_m = [], []
        with torch.no_grad():
            for x, y, g in test_loader:
                x = x.to(device)
                feats = backbone_fn(x)
                feats = instance_normalize(feats)
                logits = ensemble(feats, mode=mode)
                preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
                labels_all_m.extend(y.numpy().tolist())
                groups_all_m.extend(g.numpy().tolist())

        summary = summarize_split(
            np.asarray(preds), np.asarray(labels_all_m),
            np.asarray(groups_all_m), num_groups, num_labels)
        wga = summary["worst_group_acc"]["value"]
        gap = summary["group_gap_acc"]
        avg = summary["overall_acc"]
        print(f"  [{mode}] test WGA={wga:.4f}, gap={gap:.4f}, avg={avg:.4f}")

        if mode == "confidence":
            labels_all = labels_all_m
            groups_all = groups_all_m
            with open(out_dir / "eval_test.json", "w") as f:
                json.dump(summary, f, indent=2)

    # Per-head test results
    print("\n  Per-head test results:")
    for i, head in enumerate(ensemble.heads):
        preds_h = []
        with torch.no_grad():
            for x, y, g in test_loader:
                x = x.to(device)
                feats = backbone_fn(x)
                feats = instance_normalize(feats)
                logits = head(feats)
                preds_h.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        summary_h = summarize_split(
            np.asarray(preds_h), np.asarray(labels_all),
            np.asarray(groups_all), num_groups, num_labels)
        wga_h = summary_h["worst_group_acc"]["value"]
        print(f"    head {i}: test WGA={wga_h:.4f}")

    # --- Save config ---
    config = {
        "dataset": args.dataset,
        "backbone": args.backbone,
        "merged_ckpt": str(args.merged_ckpt),
        "epochs": args.epochs,
        "lr": args.lr,
        "entropic_scale": args.entropic_scale,
        "wd_weight": args.wd_weight,
        "cov_reg": args.cov_reg,
        "seed": args.seed,
        "num_heads": 3,
        "ag_mode": True,
        "feature_norm": "instance",
        "diversity": "sequential_covariance",
        "model_selection": "last_epoch",
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    torch.save(final_state, out_dir / "ensemble.pt")
    print(f"\nResults saved to: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
