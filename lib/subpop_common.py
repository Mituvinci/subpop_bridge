"""
subpop_common.py - Shared utilities for the sub-population shift pipeline
(paper Table B: derm / CelebA / Waterbirds).

Consumers:
    s19_subpop_train_vit.py     (ERM / GroupDRO / BRIDGE-specialist trainer)
    s20_subpop_bridge_merge_vit.py (Optuna merge over BRIDGE specialists)
    s21_aggregate_subpop.py     (Table B aggregator)

What lives here (backbone-agnostic):
    - Manifest loader that reads the s18-emitted CSV + label/group JSONs.
    - Torch Dataset wrapper that returns (image_tensor, label_idx, group_idx).
    - Group-aware sampler (balanced across groups, not across classes).
    - GroupDRO loss (Sagawa et al. 2020).
    - Per-group metrics: macro accuracy, per-group accuracy, worst-group acc,
      group-gap, per-group F1.

What does NOT live here:
    - The CLIP ViT-B/32 backbone (s19_vit imports from open_clip directly).
    - The LLaMA LoRA backbone (future s19_llama).
    - Per-block lambda key generation (backbone-specific).

Design note:
    This module intentionally does not import torchvision, open_clip, or
    any backbone. That way s19_vit and a future s19_llama share the same
    data / loss / metric code without pulling each other's heavyweight
    image-model dependencies.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"


# -------------------------------------------------------------------------
# Manifest triple loader
# -------------------------------------------------------------------------

@dataclass
class SubpopManifest:
    """One dataset's manifest + label/group index maps."""
    name: str                   # "derm" | "celeba" | "waterbirds"
    df: pd.DataFrame            # columns: dataset, image_path, label_idx, group_idx, split, source
    labels: Dict[int, str]      # label_idx -> label_name
    groups: Dict[int, str]      # group_idx -> group_name

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    @property
    def num_groups(self) -> int:
        return len(self.groups)

    def split(self, name: str) -> pd.DataFrame:
        return self.df[self.df["split"] == name].reset_index(drop=True)


def load_manifest(name: str, manifest_dir: Optional[Path] = None) -> SubpopManifest:
    """Load the s18-emitted triple for `name` in {derm, celeba, waterbirds}."""
    root = Path(manifest_dir) if manifest_dir else MANIFEST_DIR
    csv_path = root / f"{name}_manifest.csv"
    labels_path = root / f"{name}_labels.json"
    groups_path = root / f"{name}_groups.json"
    for p in (csv_path, labels_path, groups_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Run `python experiments/s18_subpop_data_prep.py`."
            )
    df = pd.read_csv(csv_path)
    with open(labels_path) as f:
        labels = {int(k): v for k, v in json.load(f).items()}
    with open(groups_path) as f:
        groups = {int(k): v for k, v in json.load(f).items()}
    return SubpopManifest(name=name, df=df, labels=labels, groups=groups)


# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------

class SubpopImageDataset(Dataset):
    """Returns (image_tensor, label_idx, group_idx) triples.

    The caller supplies `preprocess` — typically CLIP's val_preprocess for
    ViT-B/32 or LLaVA's image processor for LLaMA. This keeps the dataset
    backbone-agnostic.
    """

    def __init__(self, df: pd.DataFrame, preprocess: Callable,
                 group_filter: Optional[int] = None):
        self.preprocess = preprocess
        if group_filter is not None:
            df = df[df["group_idx"] == group_filter].reset_index(drop=True)
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        tensor = self.preprocess(img)
        return tensor, int(row["label_idx"]), int(row["group_idx"])


class SubpopTextDataset(Dataset):
    """Returns (token_tensor, label_idx, group_idx) triples for text data.

    token_tensor shape: (seq_len, 2 or 3) where channels are
    [input_ids, attention_mask, token_type_ids(optional)].
    """

    def __init__(self, df: pd.DataFrame, token_data: torch.Tensor,
                 group_filter: Optional[int] = None):
        if group_filter is not None:
            mask = df["group_idx"] == group_filter
            df = df[mask].reset_index(drop=True)
            token_data = token_data[mask.values]
        self.df = df
        self.token_data = token_data

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        row = self.df.iloc[idx]
        return self.token_data[idx], int(row["label_idx"]), int(row["group_idx"])


def is_text_dataset(name: str) -> bool:
    return name in ("civilcomments", "multinli")


def load_text_tokens(name: str) -> torch.Tensor:
    """Load or build the full token tensor for a text dataset.

    Returns a (N, seq_len, 2 or 3) long tensor aligned with the manifest
    row order (train rows first, then val, then test -- matching s18 output).
    """
    data_root = PROJECT_ROOT / "data"

    if name == "multinli":
        # The cached BERT features were pickled with utils_glue.InputFeatures
        # from the DPE/SubpopBench codebase.  Make that module importable.
        import sys as _sys
        _dpe_dir = str(PROJECT_ROOT / "Other_methods" / "dpe4subpop")
        if _dpe_dir not in _sys.path:
            _sys.path.insert(0, _dpe_dir)

        glue_dir = data_root / "multinli" / "glue_data" / "MNLI"
        features = []
        for fname in [
            "cached_train_bert-base-uncased_128_mnli",
            "cached_dev_bert-base-uncased_128_mnli",
            "cached_dev_bert-base-uncased_128_mnli-mm",
        ]:
            features += torch.load(glue_dir / fname, weights_only=False)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        masks = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segments = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        return torch.stack((input_ids, masks, segments), dim=2)

    if name == "civilcomments":
        cache_path = data_root / "civilcomments" / "cached_tokens_bert-base-uncased_220.pt"
        if cache_path.exists():
            print(f"[subpop] loading cached tokens from {cache_path}")
            return torch.load(cache_path, weights_only=True)
        from transformers import BertTokenizer
        text_df = pd.read_csv(data_root / "civilcomments" / "civilcomments_fine.csv")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        all_tokens = tokenizer(
            list(text_df["comment_text"]),
            padding="max_length",
            truncation=True,
            max_length=220,
            return_tensors="pt",
        )
        result = torch.stack((
            all_tokens["input_ids"],
            all_tokens["attention_mask"],
            all_tokens["token_type_ids"],
        ), dim=2)
        torch.save(result, cache_path)
        print(f"[subpop] cached tokens saved to {cache_path}")
        return result

    raise ValueError(f"Unknown text dataset: {name}")


# -------------------------------------------------------------------------
# Samplers
# -------------------------------------------------------------------------

class GroupBalancedSampler(Sampler):
    """Uniform sampling across groups (not classes).

    Each minibatch has roughly equal counts from every group, so the
    GroupDRO loss sees all groups in every step. Crucial for minority
    groups whose base rate is <1%.
    """

    def __init__(self, group_idx: np.ndarray, batch_size: int,
                 num_batches: int, seed: int = 42):
        self.group_idx = np.asarray(group_idx)
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.rng = np.random.default_rng(seed)
        self.groups = np.unique(self.group_idx)
        self.per_group_idx = {
            int(g): np.where(self.group_idx == g)[0] for g in self.groups
        }

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            per_group_take = max(1, self.batch_size // len(self.groups))
            for g in self.groups:
                pool = self.per_group_idx[int(g)]
                pick = self.rng.choice(pool, size=per_group_take, replace=True)
                batch.extend(pick.tolist())
            self.rng.shuffle(batch)
            yield batch[: self.batch_size]

    def __len__(self) -> int:
        return self.num_batches


# -------------------------------------------------------------------------
# GroupDRO loss (Sagawa et al. 2020, Eqn 2)
# -------------------------------------------------------------------------

class GroupDROLoss:
    """Online exponentiated gradient on per-group weights.

    q_g(t+1) proportional to q_g(t) * exp(eta * L_g(t)), then renormalized.
    Final loss = sum_g q_g * L_g(per_sample_loss averaged within group).
    """

    def __init__(self, num_groups: int, step_size: float = 0.01,
                 device: str = "cuda"):
        self.num_groups = num_groups
        self.step_size = step_size
        self.q = torch.ones(num_groups, device=device) / num_groups
        self.device = device

    def __call__(self, per_sample_loss: torch.Tensor,
                 group_idx: torch.Tensor) -> torch.Tensor:
        group_losses = torch.zeros(self.num_groups, device=self.device)
        counts = torch.zeros(self.num_groups, device=self.device)
        for g in range(self.num_groups):
            mask = (group_idx == g)
            if mask.any():
                group_losses[g] = per_sample_loss[mask].mean()
                counts[g] = mask.sum()
        # Update q only on groups actually present in this minibatch.
        present = counts > 0
        with torch.no_grad():
            self.q[present] = self.q[present] * torch.exp(
                self.step_size * group_losses[present].detach()
            )
            self.q = self.q / self.q.sum()
        return (self.q * group_losses).sum()


# -------------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------------

def per_group_accuracy(preds: np.ndarray, labels: np.ndarray,
                       groups: np.ndarray, num_groups: int) -> Dict[int, float]:
    """Accuracy within each group. Returns {group_idx: acc in [0,1]}."""
    out: Dict[int, float] = {}
    for g in range(num_groups):
        mask = groups == g
        if not mask.any():
            out[g] = float("nan")
            continue
        out[g] = float((preds[mask] == labels[mask]).mean())
    return out


def per_group_f1(preds: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                 num_groups: int, num_labels: int) -> Dict[int, float]:
    """Macro-F1 within each group. Returns {group_idx: f1 in [0,1]}."""
    out: Dict[int, float] = {}
    for g in range(num_groups):
        mask = groups == g
        if not mask.any():
            out[g] = float("nan")
            continue
        gp = preds[mask]
        gl = labels[mask]
        f1s = []
        for c in range(num_labels):
            tp = int(((gp == c) & (gl == c)).sum())
            fp = int(((gp == c) & (gl != c)).sum())
            fn = int(((gp != c) & (gl == c)).sum())
            denom = 2 * tp + fp + fn
            f1s.append(0.0 if denom == 0 else (2 * tp) / denom)
        out[g] = float(np.mean(f1s))
    return out


def worst_group(scores: Dict[int, float]) -> Tuple[int, float]:
    """Return (group_idx, value) of the worst-scoring group (ignores NaN)."""
    valid = {g: v for g, v in scores.items() if not np.isnan(v)}
    if not valid:
        return -1, float("nan")
    g_worst = min(valid, key=valid.get)
    return g_worst, valid[g_worst]


def group_gap(scores: Dict[int, float]) -> float:
    """max_group - min_group. Ignores NaN groups."""
    valid = [v for v in scores.values() if not np.isnan(v)]
    if len(valid) < 2:
        return float("nan")
    return max(valid) - min(valid)


# -------------------------------------------------------------------------
# Evaluation helper
# -------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(model: torch.nn.Module, loader,
                         device: str = "cuda",
                         return_logits: bool = False,
                         ) -> Tuple[np.ndarray, ...]:
    """Return (preds, labels, groups[, logits]) as numpy arrays.

    `model(x)` must return logits of shape (B, num_labels).
    `loader` yields (x, label_idx, group_idx) per batch.
    """
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_groups: List[int] = []
    all_logits: List[np.ndarray] = []
    for x, label, group in loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(label.numpy().tolist())
        all_groups.extend(group.numpy().tolist())
        if return_logits:
            all_logits.append(logits.cpu().float().numpy())
    result = (np.asarray(all_preds), np.asarray(all_labels), np.asarray(all_groups))
    if return_logits:
        result = result + (np.concatenate(all_logits, axis=0),)
    return result


@torch.no_grad()
def collect_preds_and_losses(
    model: torch.nn.Module, loader, device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Like collect_predictions, but also returns per-sample CE loss.

    Used by BRIDGE-DRO's merge-time objective: we need per-group
    cross-entropy (not just accuracy) to run Sagawa 2020's adversarial
    group-weight dynamics over the lambda search.
    """
    import torch.nn.functional as F
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_groups: List[int] = []
    all_losses: List[float] = []
    for x, label, group in loader:
        x = x.to(device)
        label_t = label.to(device)
        logits = model(x)
        per_sample = F.cross_entropy(logits, label_t, reduction="none")
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(label.numpy().tolist())
        all_groups.extend(group.numpy().tolist())
        all_losses.extend(per_sample.detach().cpu().numpy().tolist())
    return (np.asarray(all_preds), np.asarray(all_labels),
            np.asarray(all_groups), np.asarray(all_losses))


def per_group_loss(losses: np.ndarray, groups: np.ndarray,
                   num_groups: int) -> Dict[int, float]:
    """Mean CE within each group. Returns {group_idx: ce}. Groups with
    zero samples get 0.0 (they do not contribute to the DRO objective)."""
    out: Dict[int, float] = {}
    for g in range(num_groups):
        mask = groups == g
        out[g] = float(losses[mask].mean()) if mask.any() else 0.0
    return out


def dro_weighted_loss(pg_loss: Dict[int, float], step_size: float = 0.01
                      ) -> Tuple[float, Dict[int, float]]:
    """Sagawa 2020 robust loss in closed form (stateless per call).

    Because per-group losses on the full val split are deterministic at
    merge time (no minibatch noise), the exponentiated-gradient dynamics
    reduce to a single softmax-weighted sum:

        q_g       = softmax(eta * L_g)
        L_robust  = sum_g q_g * L_g

    As eta grows, q concentrates on the worst group; as eta -> 0 it is
    uniform (plain mean-loss). Returns (L_robust, q) with q as a dict
    keyed by group_idx for logging.
    """
    keys = sorted(pg_loss.keys())
    L = np.asarray([pg_loss[g] for g in keys], dtype=np.float64)
    logits = step_size * L
    logits = logits - logits.max()
    q = np.exp(logits)
    q = q / q.sum()
    return float((q * L).sum()), {int(g): float(qi) for g, qi in zip(keys, q)}


def balanced_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(labels, preds))


def compute_auroc(preds: np.ndarray, labels: np.ndarray,
                  num_labels: int, logits: Optional[np.ndarray] = None) -> Dict:
    """AUROC (OVR and OVO). Uses softmax(logits) if provided, else one-hot preds."""
    from sklearn.metrics import roc_auc_score
    out: Dict[str, float] = {}
    if logits is not None and num_labels >= 2:
        probs = _softmax(logits)
        try:
            out["auroc_ovr"] = float(roc_auc_score(
                labels, probs, multi_class="ovr", average="macro"))
        except ValueError:
            out["auroc_ovr"] = float("nan")
        if num_labels > 2:
            try:
                out["auroc_ovo"] = float(roc_auc_score(
                    labels, probs, multi_class="ovo", average="macro"))
            except ValueError:
                out["auroc_ovo"] = float("nan")
    return out


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def summarize_split(preds: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                    num_groups: int, num_labels: int,
                    logits: Optional[np.ndarray] = None) -> Dict:
    """Compact metrics dict suitable for JSON logging."""
    pg_acc = per_group_accuracy(preds, labels, groups, num_groups)
    pg_f1 = per_group_f1(preds, labels, groups, num_groups, num_labels)
    g_worst_acc, v_worst_acc = worst_group(pg_acc)
    g_worst_f1, v_worst_f1 = worst_group(pg_f1)
    result = {
        "overall_acc": float((preds == labels).mean()),
        "balanced_acc": balanced_accuracy(preds, labels),
        "per_group_acc": {str(g): v for g, v in pg_acc.items()},
        "per_group_f1": {str(g): v for g, v in pg_f1.items()},
        "worst_group_acc": {"group_idx": int(g_worst_acc), "value": v_worst_acc},
        "worst_group_f1": {"group_idx": int(g_worst_f1), "value": v_worst_f1},
        "group_gap_acc": group_gap(pg_acc),
        "group_gap_f1": group_gap(pg_f1),
    }
    auroc = compute_auroc(preds, labels, num_labels, logits)
    result.update(auroc)
    return result
