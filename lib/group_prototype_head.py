"""s26-D -- Group-prototype IsoMax+ head on the BRIDGE-merged backbone.

Same setup as s26 (frozen merged ViT, train head only on a balanced
subset of val), but the IsoMax+ head has G * C prototypes instead of C.
Each (label_idx l, group_idx g) cell gets its own prototype p_{l,g}.
At training time the loss matches the prototype indexed by the sample's
true (l, g). At test time we have no access to g so we either:

    --inference max-over-group : pred_l = max_g  -dist(f, p_{l, g})
                                 (closest matching subgroup prototype
                                  per class wins)
    --inference mean-over-group: pred_l = mean_g -dist(f, p_{l, g})

This is intentionally *not* DPE (no ensembling, no diversity loss). It
is a single-head extension of s26 that uses our group labels to give
the head per-subgroup centers, which is the cheapest way to ask "does
the head benefit from group structure at all, before we go ensemble?"

Usage (CelebA, val attribute-balanced):
    python experiments/s26d_bridge_group_proto.py --dataset celeba \\
        --merged-ckpt results/s20_subpop_bridge_erm_celeba/combined/20260422_164100/merged_model.pt \\
        --val-balance attribute --inference max --epochs 20
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
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from train_erm import CLIPVisualWithHead, EMBED_DIM
from lib.isomax_head import select_balanced_indices
from lib.backbones import build_backbone, embed_dim as backbone_embed_dim
from lib.subpop_common import (
    SubpopImageDataset,
    load_manifest,
    summarize_split,
)


# ---------------------------------------------------------------------
# Group-prototype head
# ---------------------------------------------------------------------

class GroupPrototypeHead(nn.Module):
    """IsoMax+ head with one prototype per (label, group).

    Internally we store a flat (G * C, D) prototype matrix and a single
    distance scale (Macedo's |s|). Forward returns (B, C) logits where
    each column is the negative distance to the *best-matching subgroup
    prototype* for that class (max-over-group), or the mean (mean-over-
    group). The aggregation is set at construction time.
    """

    def __init__(self, num_features: int, num_labels: int, num_groups: int,
                 inference: str = "max", temperature: float = 1.0):
        super().__init__()
        if inference not in ("max", "mean"):
            raise ValueError(f"inference must be max|mean, got {inference}")
        self.num_features = num_features
        self.num_labels = num_labels
        self.num_groups = num_groups
        self.temperature = temperature
        self.inference = inference
        # Flat row index = label * num_groups + group_within_label.
        # We do NOT assume groups are nested in labels; we just store one
        # prototype per *training* (label, group) pair we observe. Cells
        # never seen at training time will end up at their init values
        # and do not matter at test (they won't dominate the max).
        self.prototypes = nn.Parameter(
            torch.empty(num_labels * num_groups, num_features))
        self.distance_scale = nn.Parameter(torch.empty(1))
        nn.init.normal_(self.prototypes, mean=0.0, std=1e-2)
        nn.init.constant_(self.distance_scale, 1.0)

    def _all_distances(self, features: torch.Tensor) -> torch.Tensor:
        """Return (B, num_labels * num_groups) distances."""
        return torch.abs(self.distance_scale) * torch.cdist(
            F.normalize(features),
            F.normalize(self.prototypes),
            p=2.0,
            compute_mode="donot_use_mm_for_euclid_dist",
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # (B, L*G) -> (B, L, G)
        d = self._all_distances(features).view(
            features.shape[0], self.num_labels, self.num_groups)
        if self.inference == "max":
            # closest matching subgroup prototype wins per class
            d_per_label, _ = d.min(dim=2)
        else:
            d_per_label = d.mean(dim=2)
        return -d_per_label / self.temperature

    def supervised_loss(self, features: torch.Tensor,
                        labels: torch.Tensor, groups: torch.Tensor,
                        entropic_scale: float) -> torch.Tensor:
        """IsoMax+-style loss against the prototype indexed by (l, g)."""
        d = self._all_distances(features)  # (B, L*G)
        flat_idx = labels * self.num_groups + groups  # (B,)
        # Softmax is taken over ALL prototypes (Macedo's entropic trick).
        # Target is the (l, g) prototype matching the sample.
        probs = F.softmax(-entropic_scale * d, dim=1)
        prob_targets = probs[torch.arange(d.size(0)), flat_idx]
        return -torch.log(prob_targets.clamp_min(1e-12)).mean()


# ---------------------------------------------------------------------
# Frozen wrapper
# ---------------------------------------------------------------------

class FrozenBackboneGroupProto(nn.Module):
    def __init__(self, backbone: CLIPVisualWithHead,
                 head: GroupPrototypeHead):
        super().__init__()
        self.visual = backbone.visual
        self.head = head
        for p in self.visual.parameters():
            p.requires_grad = False

    def features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            f = self.visual(x)
        return f.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


# ---------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------

def train_head(model: FrozenBackboneGroupProto, train_loader, val_loader,
               epochs: int, lr: float, momentum: float,
               entropic_scale: float, wd_weight: float,
               num_groups: int, num_labels: int,
               device: str, log_jsonl: Path) -> None:
    optimizer = torch.optim.SGD(model.head.parameters(),
                                lr=lr, momentum=momentum)
    model.visual.eval()
    for epoch in range(epochs):
        model.head.train()
        for x, y, g in train_loader:
            x, y, g = x.to(device), y.to(device), g.to(device)
            feats = model.features(x)
            clf_loss = model.head.supervised_loss(
                feats, y, g, entropic_scale=entropic_scale)
            P = model.head.prototypes.unsqueeze(1)
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
        # epoch eval
        model.head.eval()
        preds, labels, groups = [], [], []
        with torch.no_grad():
            for x, y, g in val_loader:
                x = x.to(device)
                logits = model(x)
                preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
                labels.extend(y.numpy().tolist())
                groups.extend(g.numpy().tolist())
        summary = summarize_split(
            np.asarray(preds), np.asarray(labels), np.asarray(groups),
            num_groups, num_labels)
        with open(log_jsonl, "a") as f:
            f.write(json.dumps({
                "epoch": epoch, "eval_tag": "val", **summary,
            }) + "\n")


@torch.no_grad()
def eval_split(model: FrozenBackboneGroupProto, loader, device: str,
               num_groups: int, num_labels: int) -> dict:
    model.eval()
    preds, labels, groups = [], [], []
    for x, y, g in loader:
        x = x.to(device)
        logits = model(x)
        preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        labels.extend(y.numpy().tolist())
        groups.extend(g.numpy().tolist())
    return summarize_split(
        np.asarray(preds), np.asarray(labels), np.asarray(groups),
        num_groups, num_labels)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True,
                    choices=["celeba", "waterbirds", "derm"])
    ap.add_argument("--backbone", default="clip_vit_b32",
                    choices=["clip_vit_b32", "resnet50"])
    ap.add_argument("--merged-ckpt", type=Path, required=True)
    ap.add_argument("--val-balance", choices=["class", "attribute"],
                    default="attribute",
                    help="attribute-balanced is the natural choice for "
                         "a per-(label,group) prototype head.")
    ap.add_argument("--inference", choices=["max", "mean"], default="max",
                    help="How to aggregate per-group prototypes back to "
                         "a per-class score at inference (no group label "
                         "available).")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--entropic-scale", type=float, default=30.0)
    ap.add_argument("--wd-weight", type=float, default=10.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path,
                    default=PROJECT_ROOT / "results" / "s26d_bridge_group_proto")
    ap.add_argument("--manifest-dir", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    manifest = load_manifest(args.dataset, args.manifest_dir)
    print(f"[s26d] {args.dataset}: {manifest.num_labels} labels, "
          f"{manifest.num_groups} groups, {len(manifest.df)} rows.")
    val_df = manifest.split("val")
    test_df = manifest.split("test")

    # backbone (frozen)
    backbone = build_backbone(args.backbone, num_labels=manifest.num_labels)
    state = torch.load(args.merged_ckpt, map_location="cpu")
    backbone.visual.load_state_dict(state["visual"])
    if "head" in state:
        backbone.head.load_state_dict(state["head"])
    backbone.to(device)

    # group-prototype head
    embed_dim = backbone_embed_dim(args.backbone)
    head = GroupPrototypeHead(
        num_features=embed_dim,
        num_labels=manifest.num_labels,
        num_groups=manifest.num_groups,
        inference=args.inference,
        temperature=args.temperature,
    ).to(device)
    model = FrozenBackboneGroupProto(backbone, head).to(device)

    val_idx = select_balanced_indices(val_df, by=args.val_balance,
                                       seed=args.seed)
    train_subset_df = val_df.iloc[val_idx].reset_index(drop=True)
    print(f"[s26d] balanced ({args.val_balance}) val subset: "
          f"{len(train_subset_df)} rows.")

    train_ds = SubpopImageDataset(train_subset_df, backbone.train_preprocess)
    val_ds = SubpopImageDataset(val_df, backbone.val_preprocess)
    test_ds = SubpopImageDataset(test_df, backbone.val_preprocess)

    if args.smoke:
        n = min(len(train_ds), 2 * args.batch_size)
        train_ds = SubpopImageDataset(
            train_subset_df.iloc[:n].reset_index(drop=True),
            backbone.train_preprocess)
        val_ds = SubpopImageDataset(
            val_df.iloc[:args.batch_size].reset_index(drop=True),
            backbone.val_preprocess)
        test_ds = SubpopImageDataset(
            test_df.iloc[:args.batch_size].reset_index(drop=True),
            backbone.val_preprocess)
        args.epochs = 2

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=False,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir / args.dataset / args.val_balance / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    log_jsonl = out_dir / "train_log.jsonl"

    config = {k: (str(v) if isinstance(v, Path) else v)
              for k, v in vars(args).items()}
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    train_head(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, momentum=args.momentum,
        entropic_scale=args.entropic_scale, wd_weight=args.wd_weight,
        num_groups=manifest.num_groups, num_labels=manifest.num_labels,
        device=device, log_jsonl=log_jsonl)

    val_summary = eval_split(model, val_loader, device,
                             manifest.num_groups, manifest.num_labels)
    test_summary = eval_split(model, test_loader, device,
                              manifest.num_groups, manifest.num_labels)

    with open(out_dir / "eval_val.json", "w") as f:
        json.dump(val_summary, f, indent=2)
    with open(out_dir / "eval_test.json", "w") as f:
        json.dump(test_summary, f, indent=2)
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"config": config, "val": val_summary, "test": test_summary},
                  f, indent=2)
    torch.save({
        "prototypes": head.prototypes.detach().cpu(),
        "distance_scale": head.distance_scale.detach().cpu(),
        "config": config,
    }, out_dir / "group_proto_head.pt")
    print(f"[s26d] done. test worst={test_summary['worst_group_acc']['value']:.4f} "
          f"gap={test_summary['group_gap_acc']:.4f} "
          f"avg={test_summary['overall_acc']:.4f}")
    print(f"[s26d] outputs at {out_dir}")


if __name__ == "__main__":
    main()
