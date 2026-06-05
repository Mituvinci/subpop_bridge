"""Backbone factory for the subpopulation pipeline.

Provides a uniform interface so the ERM and BRIDGE stages can swap between
CLIP ViT-B/32 (default, used for all current Table 2 numbers) and
torchvision ResNet-50 (for DPE-style apples-to-apples comparisons,
since DPE / SubpopBench use ResNet-50 backbones).

Each backbone exposes the same surface as train_erm's CLIPVisualWithHead:
    .visual                 â€” feature extractor returning (B, embed_dim)
    .head                   â€” nn.Linear(embed_dim, num_labels)
    .train_preprocess(img)  â€” torchvision-style transform
    .val_preprocess(img)    â€” torchvision-style transform
    .forward(x)             â€” (B, num_labels) logits
    .save(path)             â€” torch.save({"visual": ..., "head": ...})

Plus, exposed by the factory itself:
    embed_dim               â€” int
    block_regex             â€” regex string for bucketing params by block

Block regexes:
    clip_vit_b32 : r"transformer\\.resblocks\\.\\d+"
    resnet50     : r"layer[1-4]\\.\\d+"     (each Bottleneck = one block)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn


@dataclass
class BackboneSpec:
    name: str
    embed_dim: int
    block_regex: str


_SPECS = {
    "clip_vit_b32": BackboneSpec(
        name="clip_vit_b32", embed_dim=512,
        block_regex=r"transformer\.resblocks\.\d+",
    ),
    "resnet50": BackboneSpec(
        name="resnet50", embed_dim=2048,
        block_regex=r"layer[1-4]\.\d+",
    ),
    "bert-base-uncased": BackboneSpec(
        name="bert-base-uncased", embed_dim=768,
        block_regex=r"encoder\.layer\.\d+",
    ),
}


def spec(name: str) -> BackboneSpec:
    if name not in _SPECS:
        raise ValueError(f"Unknown backbone {name!r}. "
                         f"Choices: {sorted(_SPECS)}")
    return _SPECS[name]


# ---------------------------------------------------------------------
# ResNet-50 wrapper (torchvision IMAGENET1K_V2 weights)
# ---------------------------------------------------------------------

class ResNet50WithHead(nn.Module):
    """torchvision ResNet-50 (IMAGENET1K_V2) feature extractor + linear
    classification head.

    The visual tower is the full ResNet-50 with the final fc replaced by
    nn.Identity, returning a (B, 2048) feature vector. The head is a
    fresh Linear(2048, num_labels). The visual and head state_dicts are
    saved separately so the encoder can be reused (frozen) downstream.

    Pretrained weights cache: by default torchvision downloads to
    ~/.cache/torch/hub/checkpoints. Compute nodes have no internet, so
    pre-download on the login node:

        wget -P ~/.cache/torch/hub/checkpoints \\
            https://download.pytorch.org/models/resnet50-11ad3fa6.pth
    """

    def __init__(self, num_labels: int, pretrained_path: Optional[str] = None,
                 use_pretrained: bool = True):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        if use_pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V2
        else:
            weights = None
        backbone = resnet50(weights=weights)
        backbone.fc = nn.Identity()
        self.visual = backbone
        self.head = nn.Linear(2048, num_labels)
        from torchvision.transforms import v2
        import torch as _torch
        _norm = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        _dpe_eval = v2.Compose([
            v2.Resize((256, 256), antialias=True),
            v2.CenterCrop((224, 224)),
            v2.ToImage(), v2.ToDtype(_torch.float32, scale=True),
            v2.Normalize(**_norm),
        ])
        self.val_preprocess = _dpe_eval
        self.train_preprocess = _dpe_eval

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
        torch.save({"visual": self.visual.state_dict(),
                    "head": self.head.state_dict()}, path)


# ---------------------------------------------------------------------
# BERT wrapper (bert-base-uncased, for text datasets)
# ---------------------------------------------------------------------

class BertWithHead(nn.Module):
    """HuggingFace bert-base-uncased feature extractor + linear head.

    Input is (B, seq_len, 2 or 3) int tensor where [:,:,0] = input_ids,
    [:,:,1] = attention_mask, [:,:,2] = token_type_ids (optional).
    Output of .visual is the pooler_output (B, 768).

    Matches DPE's BertFeatureWrapper interface exactly.
    """

    def __init__(self, num_labels: int, pretrained_path: Optional[str] = None,
                 dropout: float = 0.0):
        super().__init__()
        from transformers import BertModel
        bert = BertModel.from_pretrained("bert-base-uncased")
        self.visual = bert
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(768, num_labels)
        self.train_preprocess = None
        self.val_preprocess = None

        if pretrained_path is not None:
            state = torch.load(pretrained_path, map_location="cpu")
            if "visual" in state:
                self.visual.load_state_dict(state["visual"])
            if "head" in state:
                self.head.load_state_dict(state["head"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kwargs = {
            "input_ids": x[:, :, 0],
            "attention_mask": x[:, :, 1],
        }
        if x.shape[-1] == 3:
            kwargs["token_type_ids"] = x[:, :, 2]
        output = self.visual(**kwargs)
        if hasattr(output, "pooler_output"):
            feats = self.dropout(output.pooler_output)
        else:
            feats = self.dropout(output.last_hidden_state[:, 0, :])
        return self.head(feats)

    def save(self, path: Path):
        torch.save({"visual": self.visual.state_dict(),
                    "head": self.head.state_dict()}, path)


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------

def build_backbone(name: str, num_labels: int,
                    pretrained_path: Optional[str] = None,
                    use_pretrained: bool = True,
                    dropout: float = 0.0) -> nn.Module:
    """Return a backbone+head module for `name`.

    For clip_vit_b32 we delegate to s19's CLIPVisualWithHead so the
    existing module is the single source of truth; for resnet50 we use
    the local ResNet50WithHead above.
    """
    if name == "clip_vit_b32":
        import importlib, sys
        from pathlib import Path as _P
        sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
        s19 = importlib.import_module("train_erm")
        return s19.CLIPVisualWithHead(num_labels=num_labels,
                                       pretrained_path=pretrained_path)
    if name == "resnet50":
        return ResNet50WithHead(num_labels=num_labels,
                                 pretrained_path=pretrained_path,
                                 use_pretrained=use_pretrained)
    if name == "bert-base-uncased":
        return BertWithHead(num_labels=num_labels,
                            pretrained_path=pretrained_path,
                            dropout=dropout)
    raise ValueError(f"Unknown backbone: {name}")


# Convenience: preserve the EMBED_DIM constant used by the BRIDGE code.
def embed_dim(name: str) -> int:
    return spec(name).embed_dim


def block_regex(name: str) -> str:
    return spec(name).block_regex
