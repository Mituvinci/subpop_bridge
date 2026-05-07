"""
s18_fitzpatrick_prep.py -- Prepare Fitzpatrick skin tone dataset for the subpop pipeline.

Source:  data/dermatology/unified/unified_dataset.csv
         (Fitzpatrick17k + DDI + Mafi-Bangladesh + DermaCon-IN)
         All sources are shuffled together and split randomly (no source-based split).

Label:  7 skin conditions (multi-class):
        0=basal_cell_carcinoma, 1=melanoma, 2=mycosis_fungoides,
        3=nevus, 4=pyogenic_granuloma, 5=seborrheic_keratosis,
        6=squamous_cell_carcinoma
Attr:   0=light (Fitzpatrick I-III), 1=dark (Fitzpatrick IV-VI)
        Mafi-Bangladesh and DermaCon-IN are always dark.
Groups: label x attr => 14 groups (7 conditions x 2 skin tones)

Split:  70/15/15 stratified by group (re-split from scratch, ignoring existing split column).

Emits:
  data/manifests/fitzpatrick_manifest.csv
  data/manifests/fitzpatrick_labels.json
  data/manifests/fitzpatrick_groups.json
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNIFIED_CSV = PROJECT_ROOT / "data" / "dermatology" / "unified" / "unified_dataset.csv"
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"

CONDITIONS = sorted([
    "basal_cell_carcinoma",
    "melanoma",
    "mycosis_fungoides",
    "nevus",
    "pyogenic_granuloma",
    "seborrheic_keratosis",
    "squamous_cell_carcinoma",
])
COND_TO_IDX = {c: i for i, c in enumerate(CONDITIONS)}


def main():
    df = pd.read_csv(UNIFIED_CSV)
    print(f"Raw unified dataset: {len(df)} rows")

    missing = df[~df["image_path"].apply(lambda p: Path(p).exists())]
    if len(missing) > 0:
        print(f"WARNING: {len(missing)} images not found, dropping them")
        df = df[df["image_path"].apply(lambda p: Path(p).exists())].copy()

    df["label_idx"] = df["condition"].map(COND_TO_IDX)
    assert df["label_idx"].notna().all(), "Unknown conditions found"
    df["label_idx"] = df["label_idx"].astype(int)

    df["attr"] = (df["skin_tone_group"] == "dark").astype(int)
    # group = label * 2 + attr  => 14 groups (0..13)
    df["group_idx"] = df["label_idx"] * 2 + df["attr"]

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["group_idx"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["group_idx"]
    )

    train_df = train_df.copy(); train_df["split"] = "train"
    val_df = val_df.copy();     val_df["split"] = "val"
    test_df = test_df.copy();   test_df["split"] = "test"

    full = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full["dataset"] = "fitzpatrick"
    full["source_orig"] = full["source"]

    manifest = full[["dataset", "image_path", "label_idx", "group_idx",
                      "split", "source_orig"]].copy()
    manifest.rename(columns={"source_orig": "source"}, inplace=True)

    print(f"\nSplit sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"Total: {len(manifest)}")
    print(f"Classes: {len(CONDITIONS)}, Attributes: 2, Groups: {df['group_idx'].nunique()}")

    print(f"\nGroup distribution:")
    for split in ["train", "val", "test"]:
        print(f"  {split}:")
        sub = manifest[manifest["split"] == split]
        for g in sorted(sub["group_idx"].unique()):
            cond = CONDITIONS[g // 2]
            tone = "dark" if g % 2 == 1 else "light"
            print(f"    group {g:2d} ({cond}_{tone}): {len(sub[sub['group_idx']==g])}")

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_DIR / "fitzpatrick_manifest.csv", index=False)

    labels = {str(i): c for i, c in enumerate(CONDITIONS)}
    groups = {}
    for g in range(len(CONDITIONS) * 2):
        cond = CONDITIONS[g // 2]
        tone = "dark" if g % 2 == 1 else "light"
        groups[str(g)] = f"{cond}_{tone}"

    with open(MANIFEST_DIR / "fitzpatrick_labels.json", "w") as f:
        json.dump(labels, f, indent=2)
    with open(MANIFEST_DIR / "fitzpatrick_groups.json", "w") as f:
        json.dump(groups, f, indent=2)

    print(f"\nSaved to {MANIFEST_DIR}/fitzpatrick_*")


if __name__ == "__main__":
    main()
