"""
train_erm.py - Train CLIP ViT-B/32 specialists for Table B
(sub-population shift: derm / CelebA / Waterbirds).

Three training methods exposed through `--method`:

    erm               : standard cross-entropy on all train images.
                        Vanilla baseline.
    groupdro          : Sagawa et al. 2020, online exponentiated
                        gradient on per-group weights. Baseline.
    bridge_specialist : ERM-style fine-tune on a single group's images.
                        Produces one checkpoint per group (ablation only).

Backbone: open_clip ViT-B/32 visual encoder + a linear classification
head sized to the dataset. The CLIP text tower is not used (we are not
doing zero-shot here; we are fine-tuning for a specific supervised
task). The visual encoder weights and the head are saved separately.

Shared utilities (manifest loader, group sampler, GroupDRO loss,
per-group metrics) live in lib/subpop_common.py.

Output layout under `--out-dir`:
    {dataset}/{method}[/group_{g}]/{timestamp}/
        model.pt                 encoder + head state dicts
        config.json              run config
        train_log.jsonl          per-step losses
        eval_val.json            per-group metrics on val split
        eval_test.json           per-group metrics on test split

Usage:
    # ERM on derm (all groups):
    python train_erm.py --dataset derm --method erm

    # GroupDRO on CelebA:
    python train_erm.py --dataset celeba --method groupdro

    # BRIDGE specialist on Waterbirds group 2 (waterbird_on_land):
    python train_erm.py --dataset waterbirds \\
        --method bridge_specialist --group-idx 2

Smoke (1 epoch, tiny batch, cpu ok):
    python train_erm.py --dataset derm --method erm \\
        --epochs 1 --batch-size 8 --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.subpop_common import (
    GroupBalancedSampler,
    GroupDROLoss,
    SubpopImageDataset,
    SubpopTextDataset,
    SubpopManifest,
    collect_predictions,
    is_text_dataset,
    load_manifest,
    load_text_tokens,
    summarize_split,
)
from lib.backbones import build_backbone, embed_dim as backbone_embed_dim


# -------------------------------------------------------------------------
# Strong augmentation (DPE-style ERM*)
# -------------------------------------------------------------------------

_NORM_CLIP = dict(mean=(0.48145466, 0.4578275, 0.40821073),
                  std=(0.26862954, 0.26130258, 0.27577711))
_NORM_IMAGENET = dict(mean=(0.485, 0.456, 0.406),
                      std=(0.229, 0.224, 0.225))


def build_strong_augmentation(backbone_name: str, dataset: str = ""):
    """DPE-style ERM* training augmentation, matched per-dataset.

    CelebA uses HFlip + Rotation + Resize + CenterCrop (DPE's CelebA class).
    All other image datasets use RandomResizedCrop + PhotometricDistort + HFlip
    (DPE's generic augmentation).
    """
    from torchvision.transforms import v2
    import torch as _torch
    norm = _NORM_CLIP if "clip" in backbone_name else _NORM_IMAGENET
    if dataset == "celeba":
        return v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation((-15, 15)),
            v2.Resize((256, 256)),
            v2.CenterCrop((224, 224)),
            v2.ToImage(), v2.ToDtype(_torch.float32, scale=True),
            v2.Normalize(**norm),
        ])
    return v2.Compose([
        v2.RandomResizedCrop(size=(224, 224)),
        v2.RandomPhotometricDistort(p=0.5),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToImage(), v2.ToDtype(_torch.float32, scale=True),
        v2.Normalize(**norm),
    ])


CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
EMBED_DIM = 512

# open_clip caches the OpenAI ViT-B/32 weights here; override with the
# OPENCLIP_CACHE_DIR env var. On an offline machine, pre-download
# 'ViT-B-32.pt' into this dir (filename must match open_clip's URL basename);
# otherwise open_clip downloads it on first use.
CLIP_CACHE_DIR = os.environ.get(
    "OPENCLIP_CACHE_DIR", os.path.expanduser("~/.cache/openclip")
)


# -------------------------------------------------------------------------
# Backbone + head
# -------------------------------------------------------------------------

class CLIPVisualWithHead(nn.Module):
    """open_clip ViT-B/32 visual tower + linear classification head.

    The visual tower returns a 512-dim embedding. The head is a single
    Linear(512, num_labels). The visual encoder and head are saved
    separately so the encoder can be reused (frozen) by the BRIDGE
    head-training stage.
    """

    def __init__(self, num_labels: int, pretrained_path: Optional[str] = None):
        super().__init__()
        import open_clip
        model, _, val_preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, cache_dir=CLIP_CACHE_DIR
        )
        # open_clip exposes the visual tower at model.visual
        self.visual = model.visual
        self.val_preprocess = val_preprocess
        self.train_preprocess = val_preprocess  # CLIP uses the same transform
        self.head = nn.Linear(EMBED_DIM, num_labels)
        if pretrained_path is not None:
            state = torch.load(pretrained_path, map_location="cpu")
            if "visual" in state:
                self.visual.load_state_dict(state["visual"])
            if "head" in state:
                self.head.load_state_dict(state["head"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.visual(x)
        if feats.dtype != self.head.weight.dtype:
            feats = feats.to(self.head.weight.dtype)
        return self.head(feats)

    def save(self, path: Path):
        torch.save(
            {"visual": self.visual.state_dict(),
             "head": self.head.state_dict()},
            path,
        )


# -------------------------------------------------------------------------
# Training loops
# -------------------------------------------------------------------------

def _build_optimizer(model: nn.Module, optimizer: str, lr: float,
                     momentum: float, weight_decay: float) -> torch.optim.Optimizer:
    """Build either SGD (Sagawa 2020 recipe) or AdamW. SGD+momentum+wd=1e-4
    matches the group-dro paper's published CelebA/Waterbirds hyperparams,
    which is why it is the default here."""
    if optimizer == "bert_adamw":
        no_decay = ["bias", "LayerNorm.weight"]
        decay_params = []
        nodecay_params = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(nd in n for nd in no_decay):
                nodecay_params.append(p)
            else:
                decay_params.append(p)
        return torch.optim.AdamW([
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ], lr=lr, eps=1e-8)
    params = filter(lambda p: p.requires_grad, model.parameters())
    if optimizer == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum,
                               weight_decay=weight_decay)
    if optimizer == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {optimizer}")


def train_erm(model: CLIPVisualWithHead, train_loader, val_loader,
              epochs: int, lr: float, device: str, log_jsonl: Path,
              num_groups: int, num_labels: int,
              optimizer: str = "sgd", momentum: float = 0.9,
              weight_decay: float = 1e-4,
              specialist_group: Optional[int] = None,
              group_weight: float = 1.0,
              out_dir: Optional[Path] = None) -> None:
    """Plain ERM with best-WGA checkpoint saving (DPE-style)."""
    opt = _build_optimizer(model, optimizer, lr, momentum, weight_decay)
    model.train()
    step = 0
    best_wga = -1.0
    best_acc = -1.0
    weighted = specialist_group is not None and group_weight != 1.0
    for epoch in range(epochs):
        for x, y, g in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if weighted:
                g = g.to(device)
                per_sample = F.cross_entropy(logits, y, reduction="none")
                w = torch.where(
                    g == specialist_group,
                    torch.tensor(float(group_weight), device=device),
                    torch.tensor(1.0, device=device),
                )
                w = w / w.mean().clamp(min=1e-6)
                loss = (per_sample * w).mean()
            else:
                loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with open(log_jsonl, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch, "step": step, "loss": float(loss),
                }) + "\n")
            step += 1
        val_summary = _log_epoch_eval(model, val_loader, device, num_groups,
                                       num_labels, epoch, log_jsonl, tag="val")
        if out_dir is not None:
            val_wga = val_summary["worst_group_acc"]["value"]
            val_acc = val_summary["overall_acc"]
            if val_wga > best_wga:
                best_wga = val_wga
                model.save(out_dir / "model_best_wga.pt")
                print(f"[s19] epoch {epoch}: new best val WGA={val_wga:.4f}, saved model_best_wga.pt")
            if val_acc > best_acc:
                best_acc = val_acc
                model.save(out_dir / "model_best_acc.pt")
    if out_dir is not None:
        print(f"[s19] best val WGA={best_wga:.4f}, best val acc={best_acc:.4f}")


def train_groupdro(model: CLIPVisualWithHead, train_loader, val_loader,
                   epochs: int, lr: float, step_size: float, device: str,
                   num_groups: int, num_labels: int, log_jsonl: Path,
                   optimizer: str = "sgd", momentum: float = 0.9,
                   weight_decay: float = 1e-4,
                   out_dir: Optional[Path] = None) -> None:
    opt = _build_optimizer(model, optimizer, lr, momentum, weight_decay)
    dro = GroupDROLoss(num_groups=num_groups, step_size=step_size, device=device)
    model.train()
    step = 0
    best_wga = -1.0
    for epoch in range(epochs):
        for x, y, g in train_loader:
            x, y, g = x.to(device), y.to(device), g.to(device)
            logits = model(x)
            per_sample = F.cross_entropy(logits, y, reduction="none")
            loss = dro(per_sample, g)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with open(log_jsonl, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch, "step": step, "loss": float(loss),
                    "q": dro.q.detach().cpu().tolist(),
                }) + "\n")
            step += 1
        val_summary = _log_epoch_eval(model, val_loader, device, num_groups,
                                       num_labels, epoch, log_jsonl, tag="val")
        if out_dir is not None:
            val_wga = val_summary["worst_group_acc"]["value"]
            if val_wga > best_wga:
                best_wga = val_wga
                model.save(out_dir / "model_best_wga.pt")


def _log_epoch_eval(model, loader, device, num_groups, num_labels, epoch,
                    log_jsonl: Path, tag: str = "val") -> dict:
    model.eval()
    with torch.no_grad():
        preds, labels, groups = collect_predictions(model, loader, device)
    summary = summarize_split(preds, labels, groups, num_groups, num_labels)
    with open(log_jsonl, "a") as f:
        f.write(json.dumps({
            "epoch": epoch, "eval_tag": tag, **summary,
        }) + "\n")
    model.train()
    return summary


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def _slug(method: str, group_idx: Optional[int]) -> str:
    return method if group_idx is None else f"{method}/group_{group_idx}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True,
                    choices=["derm", "celeba", "waterbirds",
                             "metashift", "imagenetbg", "living17",
                             "chexpert", "civilcomments", "multinli",
                             "ham10000", "fitzpatrick"])
    ap.add_argument("--method", required=True,
                    choices=["erm", "groupdro", "bridge_specialist"])
    ap.add_argument("--backbone", default="clip_vit_b32",
                    choices=["clip_vit_b32", "resnet50", "bert-base-uncased"],
                    help="Encoder. Default clip_vit_b32 matches all "
                         "existing Table 2 / Table 3 numbers; resnet50 is "
                         "for the DPE apples-to-apples sweep.")
    ap.add_argument("--group-idx", type=int, default=None,
                    help="Required for bridge_specialist; selects which "
                         "group's subset to fine-tune on.")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--optimizer", choices=["sgd", "adamw", "bert_adamw"],
                    default="sgd",
                    help="Default sgd matches Sagawa 2020 recipe "
                         "(SGD + momentum=0.9 + weight_decay=1e-4). "
                         "bert_adamw matches DPE's BERT optimizer.")
    ap.add_argument("--momentum", type=float, default=0.9,
                    help="Only used when --optimizer=sgd.")
    ap.add_argument("--weight-decay", type=float, default=1e-4,
                    help="L2 regularization. Sagawa 2020 uses 1e-4 on "
                         "Waterbirds and CelebA.")
    ap.add_argument("--dro-step-size", type=float, default=0.01,
                    help="Sagawa 2020 eta. Only used for --method groupdro.")
    ap.add_argument("--group-weight", type=float, default=1.0,
                    help="BRIDGE specialist Option A: when > 1.0 with "
                         "--method bridge_specialist, train on the FULL "
                         "train set and upweight the specialist group's "
                         "per-sample cross-entropy by this factor. The "
                         "default (1.0) keeps the legacy 'own-group only' "
                         "behavior. Recommended: 10.0 for Waterbirds.")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path,
                    default=PROJECT_ROOT / "results" / "s19_subpop_vit")
    ap.add_argument("--manifest-dir", type=Path, default=None,
                    help="Override default data/manifests/")
    ap.add_argument("--strong-aug", action="store_true",
                    help="DPE-style ERM* augmentation (RandomResizedCrop + "
                         "PhotometricDistort + HFlip). Use for ERM* backbone "
                         "training; produces stronger features at the cost "
                         "of more epochs.")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="Dropout on BERT pooler output. SubpopBench uses "
                         "0.5 for text datasets; DPE code uses 0.1 (BERT "
                         "default). Only affects bert-base-uncased backbone.")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="Train ResNet-50 from random init (Kaiming/He) "
                         "instead of ImageNet pretrained weights. DPE uses "
                         "this for CheXpert.")
    ap.add_argument("--smoke", action="store_true",
                    help="1 epoch, 20 train steps per epoch, 1 eval batch.")
    args = ap.parse_args()

    if args.method == "bridge_specialist" and args.group_idx is None:
        ap.error("--group-idx is required for --method bridge_specialist")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- manifest + dataset splits ------------------------------------
    manifest: SubpopManifest = load_manifest(args.dataset, args.manifest_dir)
    print(f"[s19] {args.dataset}: "
          f"{manifest.num_labels} labels, {manifest.num_groups} groups, "
          f"{len(manifest.df)} rows total.")

    # Build model first (gives us the preprocess transform).
    model = build_backbone(args.backbone,
                            num_labels=manifest.num_labels,
                            use_pretrained=not args.no_pretrained,
                            dropout=args.dropout).to(device)

    if args.strong_aug:
        model.train_preprocess = build_strong_augmentation(args.backbone, args.dataset)
        print(f"[s19] ERM* mode: strong augmentation enabled for {args.backbone}/{args.dataset}")

    if args.dataset == "celeba":
        from torchvision.transforms import v2
        import torch as _torch
        norm = _NORM_CLIP if "clip" in args.backbone else _NORM_IMAGENET
        model.val_preprocess = v2.Compose([
            v2.CenterCrop(178),
            v2.Resize((224, 224)),
            v2.ToImage(), v2.ToDtype(_torch.float32, scale=True),
            v2.Normalize(**norm),
        ])
        print(f"[s19] CelebA eval: CenterCrop(178)->Resize(224) (DPE-matched)")

    # DPE evaluates Living17 on the zero-shot (zs) sub-classes and
    # ImageNetBG on mixed_rand backgrounds.  All other datasets use the
    # standard test split.
    _DPE_TEST_SPLIT = {"living17": "zs", "imagenetbg": "mixed_rand"}
    test_split_name = _DPE_TEST_SPLIT.get(args.dataset, "test")

    train_df = manifest.split("train")
    val_df = manifest.split("val")
    test_df = manifest.split(test_split_name)

    # BRIDGE specialist has two training regimes:
    #   - legacy (group_weight == 1.0): restrict TRAIN to one group.
    #     This collapses to constant-label prediction on datasets where
    #     each group contains only one (label, attribute) combo.
    #   - Option A (group_weight > 1.0): keep the FULL train set and
    #     upweight the specialist group in the loss, so the specialist
    #     has full waterbird-vs-landbird signal while still tilting
    #     toward its group.
    use_full_set = (args.method == "bridge_specialist"
                    and args.group_weight > 1.0)
    group_filter = (args.group_idx
                    if (args.method == "bridge_specialist" and not use_full_set)
                    else None)

    if is_text_dataset(args.dataset):
        print(f"[s19] Loading text tokens for {args.dataset}...")
        all_tokens = load_text_tokens(args.dataset)
        split_map = {"train": 0, "val": 1, "test": 2}
        full_df = manifest.df.copy()
        train_mask = full_df["split"] == "train"
        val_mask = full_df["split"] == "val"
        test_mask = full_df["split"] == "test"
        train_ds = SubpopTextDataset(
            full_df[train_mask].reset_index(drop=True),
            all_tokens[train_mask.values],
            group_filter=group_filter)
        val_ds = SubpopTextDataset(
            full_df[val_mask].reset_index(drop=True),
            all_tokens[val_mask.values])
        test_ds = SubpopTextDataset(
            full_df[test_mask].reset_index(drop=True),
            all_tokens[test_mask.values])
    else:
        train_ds = SubpopImageDataset(train_df, model.train_preprocess,
                                       group_filter=group_filter)
        val_ds = SubpopImageDataset(val_df, model.val_preprocess)
        test_ds = SubpopImageDataset(test_df, model.val_preprocess)

    if len(train_ds) == 0:
        sys.exit(f"Empty training set for {args.dataset}/{args.method}"
                 + (f"/group_{args.group_idx}" if group_filter is not None else ""))

    # --- sampler ------------------------------------------------------
    if args.method == "groupdro":
        sampler = GroupBalancedSampler(
            group_idx=train_ds.df["group_idx"].to_numpy(),
            batch_size=args.batch_size,
            num_batches=max(1, len(train_ds) // args.batch_size),
            seed=args.seed,
        )
        train_loader = DataLoader(train_ds, batch_sampler=sampler,
                                   num_workers=args.num_workers)
    else:
        # DPE uses class-balanced WeightedRandomSampler for all ERM.
        from torch.utils.data import WeightedRandomSampler
        labels_arr = train_ds.df["label_idx"].to_numpy()
        class_counts = np.bincount(labels_arr, minlength=manifest.num_labels).astype(float)
        class_counts[class_counts == 0] = 1.0
        sample_weights = 1.0 / class_counts[labels_arr]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                   sampler=sampler, drop_last=True,
                                   num_workers=args.num_workers)
        print(f"[s19] class-balanced sampling: class_counts={class_counts.astype(int).tolist()}")

    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    # --- output dir ---------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"{timestamp}_s{args.seed}"
    out_dir = Path(args.out_dir) / args.dataset / _slug(args.method, args.group_idx) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log_jsonl = out_dir / "train_log.jsonl"

    config = {
        "dataset": args.dataset,
        "method": args.method,
        "group_idx": args.group_idx,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "dro_step_size": args.dro_step_size,
        "group_weight": args.group_weight,
        "use_full_set": use_full_set,
        "seed": args.seed,
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "num_test": len(test_ds),
        "num_labels": manifest.num_labels,
        "num_groups": manifest.num_groups,
        "backbone": (f"{CLIP_MODEL_NAME}/{CLIP_PRETRAINED}"
                       if args.backbone == "clip_vit_b32"
                       else "resnet50/imagenet1k_v2"),
        "backbone_key": args.backbone,
        "embed_dim": backbone_embed_dim(args.backbone),
        "timestamp": timestamp,
        "smoke": args.smoke,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"[s19] out_dir: {out_dir}")
    print(f"[s19] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    # --- smoke mode: truncate loaders ---------------------------------
    if args.smoke:
        args.epochs = 1
        class _Trunc:
            def __init__(self, loader, n): self.loader = loader; self.n = n
            def __iter__(self):
                for i, b in enumerate(self.loader):
                    if i >= self.n: break
                    yield b
            def __len__(self): return self.n
        train_loader = _Trunc(train_loader, 20)
        val_loader = _Trunc(val_loader, 2)
        test_loader = _Trunc(test_loader, 2)

    # --- train --------------------------------------------------------
    t0 = time.time()
    if args.method in ("erm", "bridge_specialist"):
        specialist_group = (args.group_idx if use_full_set else None)
        group_weight = (args.group_weight if use_full_set else 1.0)
        train_erm(model, train_loader, val_loader, args.epochs, args.lr,
                  device, log_jsonl, manifest.num_groups, manifest.num_labels,
                  optimizer=args.optimizer, momentum=args.momentum,
                  weight_decay=args.weight_decay,
                  specialist_group=specialist_group,
                  group_weight=group_weight,
                  out_dir=out_dir)
    elif args.method == "groupdro":
        train_groupdro(model, train_loader, val_loader, args.epochs, args.lr,
                       args.dro_step_size, device,
                       manifest.num_groups, manifest.num_labels, log_jsonl,
                       optimizer=args.optimizer, momentum=args.momentum,
                       weight_decay=args.weight_decay,
                       out_dir=out_dir)
    elapsed = time.time() - t0

    # --- final eval + save --------------------------------------------
    # Reload the best-WGA checkpoint for final eval (BERT overfits
    # minority groups in later epochs, so last != best).
    best_wga_path = out_dir / "model_best_wga.pt"
    if best_wga_path.exists():
        state = torch.load(best_wga_path, map_location=device)
        model.visual.load_state_dict(state["visual"])
        model.head.load_state_dict(state["head"])
        print(f"[s19] loaded model_best_wga.pt for final eval")
    model.eval()

    for split_name, loader in (("val", val_loader), ("test", test_loader)):
        result = collect_predictions(model, loader, device, return_logits=True)
        preds, labels, groups, logits = result
        summary = summarize_split(preds, labels, groups,
                                   manifest.num_groups, manifest.num_labels,
                                   logits=logits)
        summary["num_examples"] = int(len(preds))
        if split_name == "test":
            summary["eval_split"] = test_split_name
        with open(out_dir / f"eval_{split_name}.json", "w") as f:
            json.dump(summary, f, indent=2)
        actual = test_split_name if split_name == "test" else split_name
        print(f"[s19] {actual}: overall_acc={summary['overall_acc']:.4f} "
              f"balanced_acc={summary['balanced_acc']:.4f} "
              f"worst_group_acc(g={summary['worst_group_acc']['group_idx']})="
              f"{summary['worst_group_acc']['value']:.4f} "
              f"gap={summary['group_gap_acc']:.4f}"
              + (f" auroc_ovr={summary['auroc_ovr']:.4f}" if 'auroc_ovr' in summary else ""))

    model.save(out_dir / "model.pt")

    # Resource usage
    resource_info = {"elapsed_seconds": elapsed}
    if torch.cuda.is_available():
        resource_info["gpu_name"] = torch.cuda.get_device_name(0)
        resource_info["gpu_memory_allocated_mb"] = round(
            torch.cuda.max_memory_allocated(0) / 1024**2, 1)
        resource_info["gpu_memory_reserved_mb"] = round(
            torch.cuda.max_memory_reserved(0) / 1024**2, 1)
    import platform
    resource_info["cpu"] = platform.processor() or platform.machine()
    resource_info["hostname"] = platform.node()

    with open(out_dir / "config.json", "w") as f:
        json.dump({**config, **resource_info}, f, indent=2)
    print(f"[s19] done in {elapsed:.1f}s. model: {out_dir / 'model.pt'}")
    if "gpu_name" in resource_info:
        print(f"[s19] GPU: {resource_info['gpu_name']}, "
              f"peak mem: {resource_info['gpu_memory_allocated_mb']}MB")


if __name__ == "__main__":
    main()
