"""s26-F -- Structured-specialization heads on the BRIDGE-merged backbone.

Multiple group-prototype heads (each one is the s26-D head) trained
jointly on the same batches but with different per-group sample
weights. Each head has an explicit role:

    head 0 -- minority-weighted  (peaked on the smallest train group)
    head 1 -- majority-weighted  (peaked on the largest train group)
    head 2 -- balanced           (uniform across groups)

At inference, head outputs are aggregated by confidence-weighted
softmax (each head votes proportionally to its own max-prob on that
input). The combination is therefore data-dependent: minority-leaning
samples lean on head 0, majority-leaning samples lean on head 1.

This is *not* DPE. DPE heads are interchangeable random-bootstrap
members, with diversity emerging from a covariance penalty (IPS).
s26-F heads have explicit roles assigned at training time via per-
group sample weights, with diversity guaranteed by design (heads
cannot collapse because their objectives differ). No IPS, no
bootstrap, no symmetric ensemble. Closest precedent is mixture-of-
experts, not DPE.

Usage:
    python experiments/s26f_bridge_structured_heads.py \\
        --dataset celeba \\
        --backbone clip_vit_b32 \\
        --merged-ckpt results/s20_subpop_bridge_erm_celeba/combined/<ts>/merged_model.pt \\
        --role-strength 0.7 --epochs 20 --val-balance attribute
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR
sys.path.insert(0, str(THIS_DIR))

from train_erm import CLIPVisualWithHead, EMBED_DIM
from lib.isomax_head import select_balanced_indices
from lib.group_prototype_head import GroupPrototypeHead
from lib.backbones import build_backbone, embed_dim as backbone_embed_dim
from lib.subpop_common import (
    SubpopImageDataset,
    SubpopTextDataset,
    is_text_dataset,
    load_manifest,
    load_text_tokens,
    summarize_split,
)


# ---------------------------------------------------------------------
# Role-weight construction
# ---------------------------------------------------------------------

def build_role_weights(
    train_df: pd.DataFrame, num_groups: int, role_strength: float,
) -> torch.Tensor:
    """Return (num_heads, num_groups) sample weights.

    Three roles:
      head 0 (minority): role_strength on argmin-frequency group, (1-rs)/(G-1) elsewhere
      head 1 (majority): role_strength on argmax-frequency group, (1-rs)/(G-1) elsewhere
      head 2 (balanced): uniform 1/G everywhere

    Train counts come from the *underlying* train split (not the
    balanced-val training subset), so role assignments reflect
    the real population skew.
    """
    counts = np.array([
        int((train_df["group_idx"] == g).sum()) for g in range(num_groups)
    ], dtype=np.float64)
    minority = int(np.argmin(counts))
    majority = int(np.argmax(counts))
    rs = role_strength
    other = (1.0 - rs) / max(1, num_groups - 1)

    minority_w = np.full(num_groups, other)
    minority_w[minority] = rs
    majority_w = np.full(num_groups, other)
    majority_w[majority] = rs
    balanced_w = np.full(num_groups, 1.0 / num_groups)

    weights = np.stack([minority_w, majority_w, balanced_w], axis=0)
    # Re-normalise so each row sums to 1 (no-op for these constructions
    # but it's a guardrail).
    weights = weights / weights.sum(axis=1, keepdims=True)
    return torch.tensor(weights, dtype=torch.float32), minority, majority


def build_class_role_weights(
    train_df: pd.DataFrame, num_groups: int, num_labels: int,
    role_strength: float,
) -> torch.Tensor:
    """Class-based role weights for the no-group-annotation setting.

    Instead of specializing by individual group, specializes by class:
      head 0 (minority class): high weight on all groups of the minority class
      head 1 (majority class): high weight on all groups of the majority class
      head 2 (balanced): uniform

    This gives BRIDGE specialization pressure without group annotations,
    analogous to DPE's class-balanced subsampling in the no-attr setting.
    """
    class_counts = np.array([
        int((train_df["label_idx"] == c).sum()) for c in range(num_labels)
    ], dtype=np.float64)
    minority_class = int(np.argmin(class_counts))
    majority_class = int(np.argmax(class_counts))

    # Map each group to its class
    group_to_class = {}
    for g in range(num_groups):
        rows = train_df[train_df["group_idx"] == g]
        if len(rows) > 0:
            group_to_class[g] = int(rows["label_idx"].mode().iloc[0])
        else:
            group_to_class[g] = -1

    rs = role_strength
    minority_w = np.zeros(num_groups)
    majority_w = np.zeros(num_groups)

    minority_groups = [g for g in range(num_groups) if group_to_class[g] == minority_class]
    majority_groups = [g for g in range(num_groups) if group_to_class[g] == majority_class]
    other_groups = [g for g in range(num_groups) if g not in minority_groups and g not in majority_groups]

    # Minority head: spread role_strength across minority-class groups
    if minority_groups:
        for g in minority_groups:
            minority_w[g] = rs / len(minority_groups)
    remaining = 1.0 - rs
    non_minority = [g for g in range(num_groups) if g not in minority_groups]
    if non_minority:
        for g in non_minority:
            minority_w[g] = remaining / len(non_minority)

    # Majority head: spread role_strength across majority-class groups
    if majority_groups:
        for g in majority_groups:
            majority_w[g] = rs / len(majority_groups)
    remaining = 1.0 - rs
    non_majority = [g for g in range(num_groups) if g not in majority_groups]
    if non_majority:
        for g in non_majority:
            majority_w[g] = remaining / len(non_majority)

    balanced_w = np.full(num_groups, 1.0 / num_groups)

    weights = np.stack([minority_w, majority_w, balanced_w], axis=0)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return torch.tensor(weights, dtype=torch.float32), minority_class, majority_class


def build_label_role_weights(
    train_df: pd.DataFrame, num_labels: int, role_strength: float,
) -> tuple[torch.Tensor, int, int]:
    """Per-label role weights for AG datasets (shape: 3 x num_labels).

    Used when groups==labels and we collapse num_groups_head=1.
    The weight is indexed by LABEL instead of group, so each head
    can specialize even though there is only 1 collapsed group.
    """
    class_counts = np.array([
        int((train_df["label_idx"] == c).sum()) for c in range(num_labels)
    ], dtype=np.float64)
    minority_class = int(np.argmin(class_counts))
    majority_class = int(np.argmax(class_counts))

    rs = role_strength
    other = (1.0 - rs) / max(1, num_labels - 1)

    minority_w = np.full(num_labels, other)
    minority_w[minority_class] = rs

    majority_w = np.full(num_labels, other)
    majority_w[majority_class] = rs

    balanced_w = np.full(num_labels, 1.0 / num_labels)

    weights = np.stack([minority_w, majority_w, balanced_w], axis=0)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return torch.tensor(weights, dtype=torch.float32), minority_class, majority_class


# ---------------------------------------------------------------------
# Structured ensemble of group-prototype heads
# ---------------------------------------------------------------------

class StructuredEnsembleHead(nn.Module):
    """N group-prototype heads with role-based supervision and
    confidence-weighted inference."""

    def __init__(self, num_features: int, num_labels: int, num_groups: int,
                 role_weights: torch.Tensor, temperature: float = 1.0,
                 inference: str = "max",
                 weight_by_label: bool = False):
        super().__init__()
        self.num_heads = int(role_weights.shape[0])
        self.num_labels = num_labels
        self.num_groups = num_groups
        self._weight_by_label = weight_by_label
        self.heads = nn.ModuleList([
            GroupPrototypeHead(num_features, num_labels, num_groups,
                               inference=inference, temperature=temperature)
            for _ in range(self.num_heads)
        ])
        self.register_buffer("role_weights", role_weights)

    @torch.no_grad()
    def per_head_logits(self, features: torch.Tensor) -> torch.Tensor:
        """(num_heads, B, num_labels). Used for diagnostics."""
        return torch.stack([h(features) for h in self.heads], dim=0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Confidence-weighted ensemble. Returns (B, num_labels)
        log-probabilities so .argmax(dim=1) gives the prediction
        and a downstream NLL would still be sensible."""
        per_head = torch.stack([h(features) for h in self.heads], dim=0)
        # per_head: (num_heads, B, num_labels) -- these are -dist scores
        probs = F.softmax(per_head, dim=-1)               # (H, B, L)
        conf = probs.max(dim=-1).values                   # (H, B)
        gate = F.softmax(conf, dim=0)                     # (H, B), sums to 1 over heads
        weighted = (probs * gate.unsqueeze(-1)).sum(dim=0)  # (B, L)
        return torch.log(weighted.clamp_min(1e-12))

    def supervised_loss(self, features: torch.Tensor, labels: torch.Tensor,
                        groups: torch.Tensor, entropic_scale: float
                        ) -> tuple[torch.Tensor, list[float]]:
        """Per-head IsoMax+ NLL with per-sample weights from
        ``self.role_weights[head_idx, groups]``. Returns total loss
        (sum across heads) and a list of per-head losses for logging."""
        total = features.new_zeros(())
        per_head = []
        for h_idx, head in enumerate(self.heads):
            d = head._all_distances(features)             # (B, L*G)
            flat_idx = labels * head.num_groups + groups  # (B,)
            probs = F.softmax(-entropic_scale * d, dim=1)
            prob_targets = probs[torch.arange(d.size(0)), flat_idx]
            nll = -torch.log(prob_targets.clamp_min(1e-12))   # (B,)
            if self._weight_by_label:
                w = self.role_weights[h_idx, labels]             # (B,)
            else:
                w = self.role_weights[h_idx, groups]             # (B,)
            # Normalise by the batch-mean weight so role-strength does
            # not change the effective LR per head.
            w_eff = w / w.mean().clamp_min(1e-8)
            loss_h = (w_eff * nll).mean()
            total = total + loss_h
            per_head.append(float(loss_h.detach()))
        return total, per_head


# ---------------------------------------------------------------------
# Frozen-backbone wrapper
# ---------------------------------------------------------------------

class FrozenBackboneStructured(nn.Module):
    def __init__(self, backbone, head: StructuredEnsembleHead,
                 text_mode: bool = False):
        super().__init__()
        self.visual = backbone.visual
        self.head = head
        self._text_mode = text_mode
        if text_mode:
            self._dropout = getattr(backbone, "dropout", nn.Identity())
        for p in self.visual.parameters():
            p.requires_grad = False

    def features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self._text_mode:
                kwargs = {"input_ids": x[:, :, 0],
                          "attention_mask": x[:, :, 1]}
                if x.shape[-1] == 3:
                    kwargs["token_type_ids"] = x[:, :, 2]
                out = self.visual(**kwargs)
                if hasattr(out, "pooler_output"):
                    f = self._dropout(out.pooler_output)
                else:
                    f = self._dropout(out.last_hidden_state[:, 0, :])
            else:
                f = self.visual(x)
        return f.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


# ---------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------

def train_head(model: FrozenBackboneStructured, train_loader, val_loader,
               epochs: int, lr: float, momentum: float,
               entropic_scale: float, wd_weight: float,
               num_groups: int, num_labels: int,
               device: str, log_jsonl: Path,
               out_dir: Optional[Path] = None,
               ag_mode: bool = False) -> None:
    optimizer = torch.optim.SGD(model.head.parameters(),
                                lr=lr, momentum=momentum)
    model.visual.eval()
    best_wga = -1.0
    best_head_state = None
    for epoch in range(epochs):
        model.head.train()
        for x, y, g in train_loader:
            x, y, g = x.to(device), y.to(device), g.to(device)
            g_loss = torch.zeros_like(g) if ag_mode else g
            feats = model.features(x)
            clf_loss, per_head = model.head.supervised_loss(
                feats, y, g_loss, entropic_scale=entropic_scale)
            wd = feats.new_zeros(())
            for h in model.head.heads:
                P = h.prototypes.unsqueeze(1)
                wd = wd + torch.einsum("ijk,ilk->ijl", P, P).squeeze().mean()
            wd = wd * wd_weight / model.head.num_heads
            loss = clf_loss + wd
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with open(log_jsonl, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch, "loss": float(loss),
                    "clf_loss": float(clf_loss), "wd": float(wd),
                    "per_head": per_head,
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
        val_wga = summary["worst_group_acc"]["value"]
        if val_wga > best_wga:
            best_wga = val_wga
            best_head_state = {
                k: v.clone().cpu() for k, v in model.head.state_dict().items()
            }
            print(f"[s26f] epoch {epoch}: new best val WGA={val_wga:.4f}")
    if best_head_state is not None:
        model.head.load_state_dict(best_head_state)
        print(f"[s26f] restored best val WGA={best_wga:.4f} for final eval")


@torch.no_grad()
def eval_split(model: FrozenBackboneStructured, loader, device: str,
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
                    choices=["celeba", "waterbirds", "derm",
                             "metashift", "imagenetbg", "living17",
                             "chexpert", "civilcomments", "multinli",
                             "ham10000", "fitzpatrick"])
    ap.add_argument("--backbone", default="clip_vit_b32",
                    choices=["clip_vit_b32", "resnet50", "bert-base-uncased"])
    ap.add_argument("--merged-ckpt", type=Path, required=True)
    ap.add_argument("--val-balance", choices=["class", "attribute"],
                    default="attribute")
    ap.add_argument("--no-group-roles", action="store_true",
                    help="Table 1 mode: all heads use uniform weights "
                         "(no minority/majority specialization). "
                         "Implies --val-balance class.")
    ap.add_argument("--inference", choices=["max", "mean"], default="max")
    ap.add_argument("--role-strength", type=float, default=0.7,
                    help="Weight on each head's role group at training. "
                         "0.7 = head spends 70%% of its supervision on "
                         "its assigned group, 30%% spread over the rest.")
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
                    default=PROJECT_ROOT / "results" / "s26f_bridge_structured")
    ap.add_argument("--manifest-dir", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _DPE_TEST_SPLIT = {"living17": "zs", "imagenetbg": "mixed_rand"}

    manifest = load_manifest(args.dataset, args.manifest_dir)
    train_df = manifest.split("train")
    val_df = manifest.split("val")
    eval_split_name = _DPE_TEST_SPLIT.get(args.dataset, "test")
    test_df = manifest.split(eval_split_name)
    print(f"[s26f] {args.dataset}: {manifest.num_labels} labels, "
          f"{manifest.num_groups} groups, {len(manifest.df)} rows.")

    # AG datasets (ImageNetBG, Living17): groups == labels.  The group-
    # prototype head would create L*L prototypes of which only L diagonal
    # cells are trained.  Collapse to num_groups_head=1 so each class gets
    # exactly 1 prototype per head (standard IsoMax).
    ag_mode = (manifest.num_groups == manifest.num_labels and
               (train_df["group_idx"] == train_df["label_idx"]).all())
    num_groups_head = 1 if ag_mode else manifest.num_groups
    if ag_mode:
        print(f"[s26f] AG mode: groups==labels, collapsing head to "
              f"num_groups=1 (standard IsoMax per head)")

    weight_by_label = False
    if args.no_group_roles:
        args.val_balance = "class"
        if ag_mode:
            role_weights, minority_c, majority_c = build_label_role_weights(
                train_df, manifest.num_labels, args.role_strength)
            weight_by_label = True
            minority_g, majority_g = -1, -1
            print(f"[s26f] --no-group-roles + AG mode: label-based specialization "
                  f"(minority class={minority_c}, majority class={majority_c})")
        else:
            role_weights, minority_c, majority_c = build_class_role_weights(
                train_df, manifest.num_groups, manifest.num_labels,
                args.role_strength)
            minority_g, majority_g = -1, -1
            print(f"[s26f] --no-group-roles: class-based specialization "
                  f"(minority class={minority_c}, majority class={majority_c})")
    else:
        if ag_mode:
            role_weights, minority_c, majority_c = build_label_role_weights(
                train_df, manifest.num_labels, args.role_strength)
            weight_by_label = True
            minority_g, majority_g = -1, -1
            print(f"[s26f] AG mode: label-based specialization "
                  f"(minority class={minority_c}, majority class={majority_c})")
        else:
            role_weights, minority_g, majority_g = build_role_weights(
                train_df, manifest.num_groups, args.role_strength)
            print(f"[s26f] role assignment: minority head -> group {minority_g}, "
                  f"majority head -> group {majority_g}, balanced head -> uniform")
    print(f"[s26f] role_weights (by={'label' if weight_by_label else 'group'}):\n"
          f"{role_weights.numpy()}")

    # Backbone (frozen).
    backbone = build_backbone(args.backbone, num_labels=manifest.num_labels)
    state = torch.load(args.merged_ckpt, map_location="cpu")
    backbone.visual.load_state_dict(state["visual"])
    if "head" in state:
        backbone.head.load_state_dict(state["head"])
    if args.dataset == "celeba":
        from torchvision.transforms import v2
        import torch as _torch
        _norm = dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        backbone.val_preprocess = v2.Compose([
            v2.CenterCrop(178),
            v2.Resize((224, 224)),
            v2.ToImage(), v2.ToDtype(_torch.float32, scale=True),
            v2.Normalize(**_norm),
        ])
        backbone.train_preprocess = backbone.val_preprocess
        print(f"[s26f] CelebA: CenterCrop(178)->Resize(224) (DPE-matched)")
    backbone.to(device)

    # Structured head.
    embed_dim = backbone_embed_dim(args.backbone)
    head = StructuredEnsembleHead(
        num_features=embed_dim,
        num_labels=manifest.num_labels,
        num_groups=num_groups_head,
        role_weights=role_weights,
        temperature=args.temperature,
        inference=args.inference,
        weight_by_label=weight_by_label,
    ).to(device)
    model = FrozenBackboneStructured(
        backbone, head, text_mode=is_text_dataset(args.dataset)).to(device)

    val_idx = select_balanced_indices(val_df, by=args.val_balance,
                                       seed=args.seed)
    train_subset_df = val_df.iloc[val_idx].reset_index(drop=True)
    print(f"[s26f] balanced ({args.val_balance}) val subset: "
          f"{len(train_subset_df)} rows.")

    if is_text_dataset(args.dataset):
        all_tokens = load_text_tokens(args.dataset)
        full_df = manifest.df.copy()
        train_mask = full_df["split"] == "train"
        val_mask = full_df["split"] == "val"
        test_mask = full_df["split"] == "test"
        all_val_tokens = all_tokens[val_mask.values]
        val_idx_tokens = all_val_tokens[val_idx]
        train_ds = SubpopTextDataset(train_subset_df, val_idx_tokens)
        val_ds = SubpopTextDataset(
            full_df[val_mask].reset_index(drop=True), all_val_tokens)
        test_ds = SubpopTextDataset(
            full_df[test_mask].reset_index(drop=True),
            all_tokens[test_mask.values])
    else:
        train_ds = SubpopImageDataset(train_subset_df, backbone.train_preprocess)
        val_ds = SubpopImageDataset(val_df, backbone.val_preprocess)
        test_ds = SubpopImageDataset(test_df, backbone.val_preprocess)

    if args.smoke:
        n = min(len(train_ds), 2 * args.batch_size)
        if is_text_dataset(args.dataset):
            train_ds = SubpopTextDataset(
                train_subset_df.iloc[:n].reset_index(drop=True),
                val_idx_tokens[:n])
            val_ds = SubpopTextDataset(
                val_df.iloc[:args.batch_size].reset_index(drop=True),
                all_val_tokens[:args.batch_size])
            test_ds = SubpopTextDataset(
                test_df.iloc[:args.batch_size].reset_index(drop=True),
                all_tokens[test_mask.values][:args.batch_size])
        else:
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
    config["minority_group"] = minority_g
    config["majority_group"] = majority_g
    config["role_weights"] = role_weights.numpy().tolist()
    config["eval_split"] = eval_split_name
    config["ag_mode"] = bool(ag_mode)
    config["num_groups_head"] = int(num_groups_head)
    config["weight_by_label"] = weight_by_label
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    train_head(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, momentum=args.momentum,
        entropic_scale=args.entropic_scale, wd_weight=args.wd_weight,
        num_groups=manifest.num_groups, num_labels=manifest.num_labels,
        device=device, log_jsonl=log_jsonl, out_dir=out_dir,
        ag_mode=ag_mode)

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
        "heads": [{"prototypes": h.prototypes.detach().cpu(),
                   "distance_scale": h.distance_scale.detach().cpu()}
                  for h in head.heads],
        "role_weights": role_weights,
        "config": config,
    }, out_dir / "structured_heads.pt")
    print(f"[s26f] done. test worst={test_summary['worst_group_acc']['value']:.4f} "
          f"gap={test_summary['group_gap_acc']:.4f} "
          f"avg={test_summary['overall_acc']:.4f}")
    print(f"[s26f] outputs at {out_dir}")


if __name__ == "__main__":
    main()
