"""IsoMax+ prototype head on a frozen BRIDGE backbone.

Loads a ViT-CLIP checkpoint (visual trunk + linear head),
freezes the visual trunk, replaces the linear head with a single
IsoMax+ prototype classifier (Macedo 2021), and fine-tunes only the
head on a balanced subset of the validation split. Reports test-set
worst-group accuracy / per-group accuracy / group gap.

Usage (CelebA, mode A = val class-balanced, ~ DPE supervision "val"):
    python lib/isomax_head.py \
        --dataset celeba \
        --merged-ckpt results/s19_subpop_vit/celeba/erm/<ts>/model_best_wga.pt \
        --val-balance class \
        --epochs 20 --batch-size 256 --lr 5e-4 \
        --entropic-scale 30 --wd-weight 10

Mode B (val attribute-balanced, ~ DPE supervision "val+train"):
    --val-balance attribute

The IsoMax+ head and its loss are imported verbatim (with attribution)
from the entropic OOD detection repo:
  https://github.com/dlmacedo/entropic-out-of-distribution-detection
DPE acknowledges the same source; we do too. The single-head case (no
ensemble, no IPS / cov_reg) is *not* DPE -- it's the Macedo head
sitting on top of our frozen backbone.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from train_erm import CLIPVisualWithHead, EMBED_DIM
from lib.backbones import build_backbone, embed_dim as backbone_embed_dim
from lib.subpop_common import (
    SubpopImageDataset,
    load_manifest,
    summarize_split,
    collect_predictions,
)


# ---------------------------------------------------------------------
# IsoMax+ head and loss -- Macedo (2021), entropic OOD detection
#   https://github.com/dlmacedo/entropic-out-of-distribution-detection
# Two classes copied verbatim. DPE imports the same code; we vendor it
# here so s26 has no path-dependency on the DPE local clone.
# ---------------------------------------------------------------------

class IsoMaxPlusHead(nn.Module):
    """Distance-based prototype head replacing nn.Linear.

    For input feature f, returns logits = -|distance_scale| *
    L2(normalize(f), normalize(prototype_k)) / temperature for each
    class k. Closest prototype = highest logit.
    """

    def __init__(self, num_features: int, num_classes: int,
                 temperature: float = 1.0):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.temperature = temperature
        self.prototypes = nn.Parameter(torch.empty(num_classes, num_features))
        self.distance_scale = nn.Parameter(torch.empty(1))
        nn.init.normal_(self.prototypes, mean=0.0, std=1e-2)
        nn.init.constant_(self.distance_scale, 1.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        distances = torch.abs(self.distance_scale) * torch.cdist(
            F.normalize(features), F.normalize(self.prototypes), p=2.0,
            compute_mode="donot_use_mm_for_euclid_dist",
        )
        return -distances / self.temperature


class IsoMaxPlusLoss(nn.Module):
    """Replaces nn.CrossEntropyLoss for an IsoMax+ head.

    Note: works with -distances (i.e. raw IsoMaxPlusHead output). The
    softmax is taken over -entropic_scale * distances, which sharpens
    the target probability (Macedo 2021 Theorem 1).
    """

    def __init__(self, entropic_scale: float = 30.0,
                 reduction: str = "mean"):
        super().__init__()
        self.entropic_scale = entropic_scale
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        distances = -logits
        probs = F.softmax(-self.entropic_scale * distances, dim=1)
        prob_targets = probs[torch.arange(distances.size(0)), targets]
        loss = -torch.log(prob_targets.clamp_min(1e-12))
        if self.reduction == "none":
            return loss
        return loss.mean()


# ---------------------------------------------------------------------
# Backbone wrapper that exposes the frozen feature extractor + IsoMax+
# head as one module (so collect_predictions / summarize_split work).
# ---------------------------------------------------------------------

class FrozenBackboneIsoMax(nn.Module):
    def __init__(self, backbone: CLIPVisualWithHead, isomax_head: IsoMaxPlusHead):
        super().__init__()
        self.visual = backbone.visual
        self.head = isomax_head
        for p in self.visual.parameters():
            p.requires_grad = False

    def features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.visual(x)
        return feats.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        return self.head(feats)


# ---------------------------------------------------------------------
# Balanced val-subset construction
# ---------------------------------------------------------------------

def select_balanced_indices(
    df, by: str, seed: int = 0,
) -> np.ndarray:
    """Pick a balanced subset of df row indices.

    by="class": equal count per label_idx (mode A; âœ“ supervision).
    by="attribute": equal count per group_idx (mode B; âœ“âœ“ supervision).
    """
    rng = np.random.default_rng(seed)
    if by == "class":
        key = "label_idx"
    elif by == "attribute":
        key = "group_idx"
    else:
        raise ValueError(f"Unknown balance mode: {by}")
    groups = sorted(df[key].unique())
    counts = [int((df[key] == g).sum()) for g in groups]
    take = min(counts)
    out = []
    for g in groups:
        pool = np.where(df[key].to_numpy() == g)[0]
        pick = rng.choice(pool, size=take, replace=False)
        out.extend(pick.tolist())
    rng.shuffle(out)
    return np.asarray(out, dtype=np.int64)


# ---------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------

def train_head(model: FrozenBackboneIsoMax,
               train_loader, val_loader,
               epochs: int, lr: float, momentum: float,
               entropic_scale: float, wd_weight: float,
               num_groups: int, num_labels: int,
               device: str, log_jsonl: Path) -> None:
    head_params = list(model.head.parameters())
    optimizer = torch.optim.SGD(head_params, lr=lr, momentum=momentum)
    criterion = IsoMaxPlusLoss(entropic_scale=entropic_scale, reduction="mean")

    model.visual.eval()
    for epoch in range(epochs):
        model.head.train()
        for x, y, _g in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            clf_loss = criterion(logits, y)
            # Prototype regularization: ||p_i^T p_j||^2 averaged.
            # Same form as DPE main.py:241-246 with cov_reg=0 (no
            # ensemble) -- collapses to per-prototype L2 norm.
            P = model.head.prototypes.unsqueeze(1)  # (C, 1, D)
            wd = torch.einsum("ijk,ilk->ijl", P, P).squeeze().mean() * wd_weight
            loss = clf_loss + wd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with open(log_jsonl, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch, "loss": float(loss),
                    "clf_loss": float(clf_loss), "wd": float(wd),
                }) + "\n")

        model.head.eval()
        preds, labels, groups = collect_predictions(model, val_loader, device)
        summary = summarize_split(preds, labels, groups, num_groups, num_labels)
        with open(log_jsonl, "a") as f:
            f.write(json.dumps({
                "epoch": epoch, "eval_tag": "val", **summary,
            }) + "\n")


def eval_split(model: FrozenBackboneIsoMax, loader, device: str,
               num_groups: int, num_labels: int) -> dict:
    model.eval()
    preds, labels, groups = collect_predictions(model, loader, device)
    return summarize_split(preds, labels, groups, num_groups, num_labels)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True,
                    choices=["celeba", "waterbirds", "derm"])
    ap.add_argument("--backbone", default="clip_vit_b32",
                    choices=["clip_vit_b32", "resnet50"],
                    help="Must match the backbone used in the checkpoint.")
    ap.add_argument("--merged-ckpt", type=Path, required=True,
                    help="Path to the backbone checkpoint "
                         "(a torch.save of the "
                         "CLIPVisualWithHead state_dict).")
    ap.add_argument("--val-balance", choices=["class", "attribute"],
                    default="class",
                    help="class = mode A (DPE âœ“), "
                         "attribute = mode B (DPE âœ“âœ“).")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4,
                    help="DPE CelebA stage-1+ default.")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--entropic-scale", type=float, default=30.0)
    ap.add_argument("--wd-weight", type=float, default=10.0,
                    help="Prototype L2 regularization coefficient "
                         "(DPE main.py wd_weight).")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path,
                    default=PROJECT_ROOT / "results" / "s26_bridge_isomax")
    ap.add_argument("--manifest-dir", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="2 epochs, capped train batches, capped eval.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- manifest + splits ------------------------------------------
    manifest = load_manifest(args.dataset, args.manifest_dir)
    print(f"[s26] {args.dataset}: {manifest.num_labels} labels, "
          f"{manifest.num_groups} groups, {len(manifest.df)} rows.")

    val_df = manifest.split("val")
    test_df = manifest.split("test")

    # --- backbone ---------------------------------------------------
    backbone = build_backbone(args.backbone, num_labels=manifest.num_labels)
    state = torch.load(args.merged_ckpt, map_location="cpu")
    if "visual" not in state:
        raise ValueError(
            f"{args.merged_ckpt} has keys {list(state.keys())[:5]}; "
            "expected 'visual' (and optionally 'head'). Is this an "
            "a valid backbone checkpoint?"
        )
    backbone.visual.load_state_dict(state["visual"])
    if "head" in state:
        backbone.head.load_state_dict(state["head"])
    backbone.to(device)

    # --- IsoMax+ head + frozen wrapper ------------------------------
    embed_dim = backbone_embed_dim(args.backbone)
    isomax = IsoMaxPlusHead(
        num_features=embed_dim, num_classes=manifest.num_labels,
        temperature=args.temperature,
    ).to(device)
    model = FrozenBackboneIsoMax(backbone, isomax).to(device)

    # --- balanced val-subset -> train_loader ------------------------
    val_idx = select_balanced_indices(val_df, by=args.val_balance,
                                      seed=args.seed)
    train_subset_df = val_df.iloc[val_idx].reset_index(drop=True)
    print(f"[s26] balanced ({args.val_balance}) val subset: "
          f"{len(train_subset_df)} rows "
          f"({len(train_subset_df) // manifest.num_groups} per group "
          f"if attribute-balanced).")

    train_ds = SubpopImageDataset(train_subset_df, backbone.train_preprocess)
    val_ds = SubpopImageDataset(val_df, backbone.val_preprocess)
    test_ds = SubpopImageDataset(test_df, backbone.val_preprocess)

    if args.smoke:
        # Take ~2 batches worth of training data for smoke.
        n = min(len(train_ds), 2 * args.batch_size)
        train_ds = SubpopImageDataset(
            train_subset_df.iloc[:n].reset_index(drop=True),
            backbone.train_preprocess,
        )
        val_ds = SubpopImageDataset(
            val_df.iloc[:args.batch_size].reset_index(drop=True),
            backbone.val_preprocess,
        )
        test_ds = SubpopImageDataset(
            test_df.iloc[:args.batch_size].reset_index(drop=True),
            backbone.val_preprocess,
        )
        args.epochs = 2

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=False,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # --- output dir -------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir / args.dataset / args.val_balance / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    log_jsonl = out_dir / "train_log.jsonl"

    config = {k: (str(v) if isinstance(v, Path) else v)
              for k, v in vars(args).items()}
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # --- train head -------------------------------------------------
    train_head(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, momentum=args.momentum,
        entropic_scale=args.entropic_scale, wd_weight=args.wd_weight,
        num_groups=manifest.num_groups, num_labels=manifest.num_labels,
        device=device, log_jsonl=log_jsonl,
    )

    # --- final eval -------------------------------------------------
    val_summary = eval_split(model, val_loader, device,
                             manifest.num_groups, manifest.num_labels)
    test_summary = eval_split(model, test_loader, device,
                              manifest.num_groups, manifest.num_labels)

    with open(out_dir / "eval_val.json", "w") as f:
        json.dump(val_summary, f, indent=2)
    with open(out_dir / "eval_test.json", "w") as f:
        json.dump(test_summary, f, indent=2)

    summary = {
        "config": config,
        "val": val_summary,
        "test": test_summary,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- head checkpoint --------------------------------------------
    torch.save({
        "prototypes": model.head.prototypes.detach().cpu(),
        "distance_scale": model.head.distance_scale.detach().cpu(),
        "config": config,
    }, out_dir / "isomax_head.pt")

    print(f"[s26] done. test worst-group acc = "
          f"{test_summary['worst_group_acc']['value']:.4f}, "
          f"gap = {test_summary['group_gap_acc']:.4f}")
    print(f"[s26] outputs at {out_dir}")


if __name__ == "__main__":
    main()
