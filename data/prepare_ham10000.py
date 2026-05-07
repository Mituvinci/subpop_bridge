"""
s18_ham10000_prep.py -- Prepare HAM10000 for the subpop pipeline.

Label:  0=benign (nv, bkl, df, vasc), 1=malignant (mel, bcc, akiec)
Attr:   0=female, 1=male
Groups: label x attr => 4 groups
  0 = benign_female
  1 = benign_male
  2 = malignant_female
  3 = malignant_male

Split:  70/15/15 stratified by group.

Emits:
  data/manifests/ham10000_manifest.csv
  data/manifests/ham10000_labels.json
  data/manifests/ham10000_groups.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "HAM10000"
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
IMAGE_DIR = DATA_DIR / "images"

MALIGNANT = {"mel", "bcc", "akiec"}
BENIGN = {"nv", "bkl", "df", "vasc"}

def main():
    meta = pd.read_csv(DATA_DIR / "HAM10000_metadata")
    print(f"Raw metadata: {len(meta)} rows")

    # Drop unknown sex
    meta = meta[meta["sex"].isin(["male", "female"])].copy()
    print(f"After dropping unknown sex: {len(meta)} rows")

    # Drop duplicate lesion_ids: keep first image per lesion
    meta = meta.drop_duplicates(subset="lesion_id", keep="first").copy()
    print(f"After dedup by lesion_id: {len(meta)} rows")

    # Binary label
    meta["label_idx"] = meta["dx"].apply(
        lambda d: 1 if d in MALIGNANT else 0
    )

    # Binary attribute
    meta["attr"] = (meta["sex"] == "male").astype(int)

    # Group = label * 2 + attr
    meta["group_idx"] = meta["label_idx"] * 2 + meta["attr"]

    # Image path
    meta["image_path"] = meta["image_id"].apply(
        lambda x: str(IMAGE_DIR / f"{x}.jpg")
    )

    # Verify images exist
    missing = meta[~meta["image_path"].apply(lambda p: Path(p).exists())]
    if len(missing) > 0:
        print(f"WARNING: {len(missing)} images not found, dropping them")
        meta = meta[meta["image_path"].apply(lambda p: Path(p).exists())]

    # Stratified split: 70/15/15 by group
    train_df, temp_df = train_test_split(
        meta, test_size=0.30, random_state=42, stratify=meta["group_idx"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["group_idx"]
    )

    train_df = train_df.copy(); train_df["split"] = "train"
    val_df = val_df.copy();     val_df["split"] = "val"
    test_df = test_df.copy();   test_df["split"] = "test"

    full = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full["dataset"] = "ham10000"
    full["source"] = "ham10000"

    manifest = full[["dataset", "image_path", "label_idx", "group_idx", "split", "source"]]

    print(f"\nSplit sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"Total: {len(manifest)}")
    print(f"\nGroup distribution:")
    group_names = {0: "benign_female", 1: "benign_male",
                   2: "malignant_female", 3: "malignant_male"}
    for split in ["train", "val", "test"]:
        print(f"  {split}:")
        sub = manifest[manifest["split"] == split]
        for g in sorted(sub["group_idx"].unique()):
            print(f"    group {g} ({group_names[g]}): {len(sub[sub['group_idx']==g])}")

    # Save
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_DIR / "ham10000_manifest.csv", index=False)

    labels = {"0": "benign", "1": "malignant"}
    groups = {"0": "benign_female", "1": "benign_male",
              "2": "malignant_female", "3": "malignant_male"}

    with open(MANIFEST_DIR / "ham10000_labels.json", "w") as f:
        json.dump(labels, f, indent=2)
    with open(MANIFEST_DIR / "ham10000_groups.json", "w") as f:
        json.dump(groups, f, indent=2)

    print(f"\nSaved to {MANIFEST_DIR}/ham10000_*")

if __name__ == "__main__":
    main()
