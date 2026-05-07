"""
s18_subpop_data_prep.py - Unify the three sub-population fairness datasets
into a single manifest schema so s19 (trainers) can treat them uniformly.

Datasets (paper Table B):
    1. Dermatology : data/dermatology/unified/unified_dataset.csv
                     Label = condition. Group = {light, dark} Fitzpatrick.
                     s19 unions synthetic dark-skin augmentation at load
                     time via s13d_synthetic_manifest.py.
                     ISIC is deliberately excluded per CLAUDE.md
                     (clinical-images-only scope).

    2. CelebA      : WILDS-style setup (Sagawa 2020).
                     Label = Blond_Hair. Group = 2*label + Male, so:
                        0=dark_hair_female (majority)
                        1=dark_hair_male    (majority)
                        2=blond_hair_female
                        3=blond_hair_male   (minority; ~0.85% of train)

    3. Waterbirds  : WILDS Waterbirds-v1.0 layout.
                     Label = y (0=landbird, 1=waterbird).
                     Group = 2*y + place, so:
                        0=landbird_on_land  (majority for y=0)
                        1=landbird_on_water (minority for y=0)
                        2=waterbird_on_land (minority for y=1)
                        3=waterbird_on_water(majority for y=1)

Output (under data/manifests/):
    derm_manifest.csv, celeba_manifest.csv, waterbirds_manifest.csv
        Shared columns:
            dataset        : "derm" | "celeba" | "waterbirds"
            image_path     : absolute path on disk
            label_idx      : integer class index
            group_idx      : integer group index
            split          : "train" | "val" | "test"
            source         : sub-source identifier (e.g., fitzpatrick17k)

    derm_labels.json, celeba_labels.json, waterbirds_labels.json
        { idx_str : label_name }

    derm_groups.json, celeba_groups.json, waterbirds_groups.json
        { idx_str : group_name }

Why label/group maps live next to the manifest:
    s19 trains a classifier head of size len(labels). s20's BRIDGE
    objective needs to know which group index is the "worst" group.
    Keeping the maps co-located with the manifest prevents drift.

Usage:
    python experiments/s18_subpop_data_prep.py
    python experiments/s18_subpop_data_prep.py --datasets celeba
    python experiments/s18_subpop_data_prep.py --verify-exists

Author: BRIDGE / NeurIPS 2026 submission.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
MANIFEST_DIR = DATA_ROOT / "manifests"

DERM_CSV = DATA_ROOT / "dermatology" / "unified" / "unified_dataset.csv"
CELEBA_DIR = DATA_ROOT / "celebA_v1.0"
WATERBIRDS_DIR = DATA_ROOT / "waterbirds_v1.0"

METASHIFT_CSV = DATA_ROOT / "metashift" / "metadata_metashift.csv"
IMAGENETBG_CSV = DATA_ROOT / "backgrounds_challenge" / "metadata.csv"
LIVING17_CSV = DATA_ROOT / "breeds" / "metadata_living17.csv"
BREEDS_INFO_DIR = DATA_ROOT / "breeds"
CHEXPERT_CSV = DATA_ROOT / "chexpert" / "subpop_bench_meta" / "metadata_no_finding.csv"


DERM_GROUP_MAP = {"light": 0, "dark": 1}
CELEBA_GROUP_MAP = {
    0: "dark_hair_female",
    1: "dark_hair_male",
    2: "blond_hair_female",
    3: "blond_hair_male",
}
WATERBIRDS_GROUP_MAP = {
    0: "landbird_on_land",
    1: "landbird_on_water",
    2: "waterbird_on_land",
    3: "waterbird_on_water",
}
CELEBA_LABEL_MAP = {0: "not_blond", 1: "blond"}
WATERBIRDS_LABEL_MAP = {0: "landbird", 1: "waterbird"}

# MetaShift: SubpopBench MetaShift Cat-vs-Dog uses y in {0=dog, 1=cat},
# a in {0=outdoor, 1=indoor}. group_idx = 2*y + a follows the
# Sagawa/SubpopBench convention. Train cells on disk: dog-outdoor (g=0)
# and cat-indoor (g=3); test cells: dog-indoor (g=1) and cat-outdoor
# (g=2). SubpopBench then mixes all four cells and randomly splits
# 65/10/25 train/val/test (seeded 42), so every group appears in every
# split. Worst group at test time is always one of {1, 2}.
METASHIFT_LABEL_MAP = {0: "dog", 1: "cat"}
METASHIFT_GROUP_MAP = {
    0: "dog_outdoor",
    1: "dog_indoor",
    2: "cat_outdoor",
    3: "cat_indoor",
}

# ImageNetBG: 9 super-classes from MadryLab IN-9. SubpopBench uses
# only `y` (no separate attribute axis), so group_idx = label_idx.
IMAGENETBG_LABEL_MAP = {
    0: "dog",
    1: "bird",
    2: "wheeled_vehicle",
    3: "reptile",
    4: "carnivore",
    5: "insect",
    6: "musical_instrument",
    7: "primate",
    8: "fish",
}
IMAGENETBG_GROUP_MAP = dict(IMAGENETBG_LABEL_MAP)

# Living17: 17 BREEDS super-classes. SubpopBench uses y as label and
# a=0 (no separate group axis); group_idx = label_idx. Pretty names
# come from data/breeds/dataset_class_info.json after setup_breeds.
LIVING17_DEFAULT_LABEL_MAP = {i: f"living17_class_{i}" for i in range(17)}

# CheXpert: binary (No Finding vs Finding), 6 groups = Sex x Race.
# SubpopBench's metadata generator creates a = {0..5} from Sex x ethnicity.
# CivilComments (fine): binary toxicity, 8 identity attributes, 16 groups.
# DPE uses CivilCommentsFine (granularity="fine").
CIVILCOMMENTS_CSV = DATA_ROOT / "civilcomments" / "metadata_civilcomments_fine.csv"
CIVILCOMMENTS_LABEL_MAP = {0: "non_toxic", 1: "toxic"}
CIVILCOMMENTS_GROUP_MAP = {i: f"group_{i}" for i in range(16)}

# MultiNLI: 3-way NLI (entailment/contradiction/neutral), 2 attributes, 6 groups.
MULTINLI_CSV = DATA_ROOT / "multinli" / "metadata_multinli.csv"
MULTINLI_LABEL_MAP = {0: "contradiction", 1: "entailment", 2: "neutral"}
MULTINLI_GROUP_MAP = {
    0: "contradiction_no_negation",
    1: "contradiction_negation",
    2: "entailment_no_negation",
    3: "entailment_negation",
    4: "neutral_no_negation",
    5: "neutral_negation",
}

CHEXPERT_LABEL_MAP = {0: "finding", 1: "no_finding"}
CHEXPERT_GROUP_MAP = {
    0: "male_white",
    1: "female_white",
    2: "male_black",
    3: "female_black",
    4: "male_other",
    5: "female_other",
}


def _save(manifest: pd.DataFrame, labels: dict, groups: dict, stem: str):
    """Write a manifest triple (CSV + labels.json + groups.json)."""
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFEST_DIR / f"{stem}_manifest.csv"
    labels_path = MANIFEST_DIR / f"{stem}_labels.json"
    groups_path = MANIFEST_DIR / f"{stem}_groups.json"

    manifest = manifest[[
        "dataset", "image_path", "label_idx", "group_idx", "split", "source",
    ]]
    manifest.to_csv(manifest_path, index=False)

    # JSON keys must be strings (json spec). Store idx as str so
    # round-tripping through json.load gives the same dict either way.
    with open(labels_path, "w") as f:
        json.dump({str(k): v for k, v in labels.items()}, f, indent=2,
                  sort_keys=True)
    with open(groups_path, "w") as f:
        json.dump({str(k): v for k, v in groups.items()}, f, indent=2,
                  sort_keys=True)

    print(f"[s18:{stem}] rows={len(manifest)}  manifest={manifest_path}")
    _print_split_group_table(manifest, stem)


def _print_split_group_table(manifest: pd.DataFrame, stem: str):
    """Per-(split, group) counts to sanity-check class/group priors."""
    counts = (
        manifest.groupby(["split", "group_idx"])
        .size().unstack(fill_value=0)
    )
    print(f"[s18:{stem}] split x group:")
    for line in counts.to_string().splitlines():
        print(f"          {line}")


# ---------------------------------------------------------------------------
# Dermatology
# ---------------------------------------------------------------------------

def build_derm_manifest(verify_exists: bool = False,
                         mix_sources: bool = False) -> pd.DataFrame:
    if not DERM_CSV.is_file():
        sys.exit(f"Missing derm unified CSV: {DERM_CSV}")
    df = pd.read_csv(DERM_CSV)

    # Deterministic label encoding: sort condition names alphabetically.
    conditions = sorted(df["condition"].dropna().unique())
    label_map = {i: name for i, name in enumerate(conditions)}
    name_to_idx = {v: k for k, v in label_map.items()}
    df["label_idx"] = df["condition"].map(name_to_idx)

    df["group_idx"] = df["skin_tone_group"].map(DERM_GROUP_MAP)
    if df["group_idx"].isna().any():
        bad = df[df["group_idx"].isna()]["skin_tone_group"].unique()
        sys.exit(f"Derm: unknown skin_tone_group values: {bad}")

    df = df.reset_index(drop=True)

    if mix_sources:
        # Mix all four sources (Fitzpatrick17k, DermaCon-IN, Mafi-Bangladesh,
        # DDI) into one pool and re-split 70/15/15 stratified by
        # (group_idx, label_idx). This converts the derm track from a
        # cross-site domain-generalization problem (train=3-sites,
        # test=DDI-only) into a proper in-distribution subpop setup, where
        # every split sees images from every source. Only use this when
        # the scientific question is "can BRIDGE reduce the light/dark
        # performance gap?" and NOT "can BRIDGE generalize across clinical
        # sites?".
        _SEED = 42
        _TRAIN = 0.70
        _VAL = 0.15

        def _shuffle_and_assign(g: pd.DataFrame) -> pd.DataFrame:
            n_g = len(g)
            n_train = int(round(n_g * _TRAIN))
            n_val = int(round(n_g * _VAL))
            n_test = n_g - n_train - n_val
            splits = (["train"] * n_train
                       + ["val"] * n_val
                       + ["test"] * n_test)
            g = g.sample(frac=1.0, random_state=_SEED)
            g = g.assign(split=splits)
            return g

        df = (df.groupby(["group_idx", "label_idx"], group_keys=False)
                .apply(_shuffle_and_assign)
                .reset_index(drop=True))
        if df["split"].isna().any():
            sys.exit("Derm mix: some rows failed to receive a split assignment")
    else:
        # Original behaviour: DDI is held out as test, 15% of the train
        # pool is carved out as val stratified by (group, label).
        df["split"] = df["split"].replace({"val": "test"})
        _VAL_FRAC = 0.15
        _VAL_SEED = 42
        train_df = df[df["split"] == "train"]
        val_idx = (
            train_df
            .groupby(["group_idx", "label_idx"], group_keys=False)
            .apply(lambda g: g.sample(frac=_VAL_FRAC, random_state=_VAL_SEED),
                   include_groups=False)
            .index
        )
        df.loc[val_idx, "split"] = "val"

    if verify_exists:
        missing = [p for p in df["image_path"].head(50) if not Path(p).is_file()]
        if missing:
            print(f"[s18:derm] WARNING: {len(missing)} missing "
                  f"image_paths in first 50 rows (sample: {missing[0]})")

    df["dataset"] = "derm"
    df = df.rename(columns={"source": "source"})
    df["label_idx"] = df["label_idx"].astype(int)
    df["group_idx"] = df["group_idx"].astype(int)

    _save(df, labels=label_map, groups={v: k for k, v in DERM_GROUP_MAP.items()},
          stem="derm")
    return df


# ---------------------------------------------------------------------------
# CelebA (Sagawa 2020 style: label=Blond_Hair, group=2*label+Male)
# ---------------------------------------------------------------------------

def build_celeba_manifest(verify_exists: bool = False) -> pd.DataFrame:
    attr_path = CELEBA_DIR / "list_attr_celeba.csv"
    part_path = CELEBA_DIR / "list_eval_partition.csv"
    img_dir = CELEBA_DIR / "img_align_celeba"
    for p in (attr_path, part_path, img_dir):
        if not p.exists():
            sys.exit(f"Missing CelebA input: {p}")

    attr = pd.read_csv(attr_path)
    part = pd.read_csv(part_path)
    df = attr.merge(part, on="image_id")

    # Attrs are encoded as {-1, +1} in CelebA. Binarize to {0, 1}.
    df["label_idx"] = (df["Blond_Hair"] == 1).astype(int)
    df["male"] = (df["Male"] == 1).astype(int)
    df["group_idx"] = (2 * df["label_idx"] + df["male"]).astype(int)

    # CelebA eval partition is 0=train, 1=val, 2=test.
    df["split"] = df["partition"].map({0: "train", 1: "val", 2: "test"})
    if df["split"].isna().any():
        sys.exit("CelebA: unexpected partition value")

    df["image_path"] = df["image_id"].apply(lambda n: str(img_dir / n))
    if verify_exists:
        missing = [p for p in df["image_path"].head(20) if not Path(p).is_file()]
        if missing:
            print(f"[s18:celeba] WARNING: {len(missing)}/20 sampled "
                  f"image_paths missing (sample: {missing[0]})")

    df["dataset"] = "celeba"
    df["source"] = "celeba"

    _save(df, labels=CELEBA_LABEL_MAP, groups=CELEBA_GROUP_MAP, stem="celeba")
    return df


# ---------------------------------------------------------------------------
# Waterbirds (WILDS Waterbirds-v1.0: y, place, split columns pre-baked)
# ---------------------------------------------------------------------------

def build_waterbirds_manifest(verify_exists: bool = False) -> pd.DataFrame:
    meta_path = WATERBIRDS_DIR / "metadata.csv"
    if not meta_path.is_file():
        sys.exit(f"Missing Waterbirds metadata: {meta_path}")
    df = pd.read_csv(meta_path)

    df["label_idx"] = df["y"].astype(int)
    df["group_idx"] = (2 * df["y"] + df["place"]).astype(int)

    df["split"] = df["split"].map({0: "train", 1: "val", 2: "test"})
    if df["split"].isna().any():
        sys.exit("Waterbirds: unexpected split value")

    df["image_path"] = df["img_filename"].apply(
        lambda n: str(WATERBIRDS_DIR / n)
    )
    if verify_exists:
        missing = [p for p in df["image_path"].head(20) if not Path(p).is_file()]
        if missing:
            print(f"[s18:waterbirds] WARNING: {len(missing)}/20 sampled "
                  f"image_paths missing (sample: {missing[0]})")

    df["dataset"] = "waterbirds"
    df["source"] = "waterbirds"

    _save(df, labels=WATERBIRDS_LABEL_MAP, groups=WATERBIRDS_GROUP_MAP,
          stem="waterbirds")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# MetaShift (SubpopBench Cat-vs-Dog, AG shift, 4 groups)
# ---------------------------------------------------------------------------

def build_metashift_manifest(verify_exists: bool = False) -> pd.DataFrame:
    if not METASHIFT_CSV.is_file():
        sys.exit(f"Missing MetaShift CSV: {METASHIFT_CSV}. "
                 "Run SubpopBench's metadata generator first.")
    df = pd.read_csv(METASHIFT_CSV)
    # SubpopBench cols: filename, y (0=dog,1=cat), a (0=outdoor,1=indoor), split (0/1/2).
    df["label_idx"] = df["y"].astype(int)
    df["group_idx"] = (2 * df["y"] + df["a"]).astype(int)
    df["split"] = df["split"].map({0: "train", 1: "val", 2: "test"})
    if df["split"].isna().any():
        sys.exit("MetaShift: unexpected split value")
    df["image_path"] = df["filename"].astype(str)
    df["dataset"] = "metashift"
    df["source"] = "metashift"

    if verify_exists:
        missing = [p for p in df["image_path"].head(20)
                   if not Path(p).is_file()]
        if missing:
            print(f"[s18:metashift] WARNING: {len(missing)}/20 sampled "
                  f"image_paths missing (sample: {missing[0]})")

    _save(df, labels=METASHIFT_LABEL_MAP, groups=METASHIFT_GROUP_MAP,
          stem="metashift")
    return df


# ---------------------------------------------------------------------------
# ImageNetBG (SC shift, 9 super-classes; group_idx = label_idx)
# ---------------------------------------------------------------------------

def build_imagenetbg_manifest(verify_exists: bool = False) -> pd.DataFrame:
    if not IMAGENETBG_CSV.is_file():
        sys.exit(f"Missing ImageNetBG CSV: {IMAGENETBG_CSV}. "
                 "Run SubpopBench's metadata generator first.")
    df = pd.read_csv(IMAGENETBG_CSV)
    # SubpopBench cols: split (str), filename, y (0..8), a (always 0).
    # split values: train, val, test, mixed_rand, only_fg, no_fg.
    # DPE evaluates on mixed_rand (not test), so we keep it alongside
    # train/val/test.  The pipeline trains on train, tunes on val, and
    # evaluates on mixed_rand to match DPE's protocol.
    keep_splits = {"train", "val", "test", "mixed_rand"}
    df = df[df["split"].isin(keep_splits)].reset_index(drop=True)

    df["label_idx"] = df["y"].astype(int)
    df["group_idx"] = df["y"].astype(int)   # a is uninformative here
    df["image_path"] = df["filename"].astype(str)
    df["dataset"] = "imagenetbg"
    df["source"] = "imagenetbg"

    if verify_exists:
        missing = [p for p in df["image_path"].head(20)
                   if not Path(p).is_file()]
        if missing:
            print(f"[s18:imagenetbg] WARNING: {len(missing)}/20 sampled "
                  f"image_paths missing (sample: {missing[0]})")

    _save(df, labels=IMAGENETBG_LABEL_MAP, groups=IMAGENETBG_GROUP_MAP,
          stem="imagenetbg")
    return df


# ---------------------------------------------------------------------------
# Living17 (AG shift, 17 BREEDS super-classes; group_idx = label_idx)
# ---------------------------------------------------------------------------

def _living17_label_map() -> dict:
    """Pretty names from BREEDS dataset_class_info.json if present,
    else generic 'living17_class_<i>' fallbacks. The pretty names are
    a list of (label_id, [list-of-imagenet-leaf-names], wordnet_id);
    we just take the wordnet display name."""
    info_path = BREEDS_INFO_DIR / "dataset_class_info.json"
    fallback = dict(LIVING17_DEFAULT_LABEL_MAP)
    if not info_path.is_file():
        return fallback
    try:
        data = json.loads(info_path.read_text())
        out = dict(fallback)
        # data is a list of [label_id, label_name, wnid] tuples.
        for entry in data:
            if isinstance(entry, list) and len(entry) >= 2:
                idx = int(entry[0])
                name = str(entry[1]).strip().replace(" ", "_")
                if 0 <= idx < 17:
                    out[idx] = name
        return out
    except Exception:
        return fallback


def build_living17_manifest(verify_exists: bool = False) -> pd.DataFrame:
    if not LIVING17_CSV.is_file():
        sys.exit(f"Missing Living17 CSV: {LIVING17_CSV}. "
                 "Run data/_gen_breeds_metadata.py first.")
    df = pd.read_csv(LIVING17_CSV)
    # SubpopBench cols: filename, y (0..16), split (0/1/2/3), a (always 0).
    # split: 0=train, 1=val, 2=test (source sub-class held-out),
    # 3=zs (zero-shot target sub-classes; AG mechanism).
    # DPE evaluates on zs (not test), so we keep all four splits.
    keep_splits = {0, 1, 2, 3}
    df = df[df["split"].isin(keep_splits)].reset_index(drop=True)

    df["label_idx"] = df["y"].astype(int)
    df["group_idx"] = df["y"].astype(int)   # a is uninformative here
    df["split"] = df["split"].map({0: "train", 1: "val", 2: "test", 3: "zs"})
    if df["split"].isna().any():
        sys.exit("Living17: unexpected split value after filter")
    df["image_path"] = df["filename"].astype(str)
    df["dataset"] = "living17"
    df["source"] = "living17"

    if verify_exists:
        missing = [p for p in df["image_path"].head(20)
                   if not Path(p).is_file()]
        if missing:
            print(f"[s18:living17] WARNING: {len(missing)}/20 sampled "
                  f"image_paths missing (sample: {missing[0]})")

    label_map = _living17_label_map()
    group_map = dict(label_map)
    _save(df, labels=label_map, groups=group_map, stem="living17")
    return df


# ---------------------------------------------------------------------------
# CheXpert (AI shift, binary label, 6 groups = Sex x Race)
# ---------------------------------------------------------------------------

def build_chexpert_manifest(verify_exists: bool = False) -> pd.DataFrame:
    if not CHEXPERT_CSV.is_file():
        sys.exit(f"Missing CheXpert metadata: {CHEXPERT_CSV}. "
                 "Run SubpopBench's metadata generator first: "
                 "cd Other_methods/SubpopBench && "
                 "python -m subpopbench.scripts.download chexpert --data_path data/")
    df = pd.read_csv(CHEXPERT_CSV)
    # SubpopBench cols: filename, y (0=finding, 1=no_finding),
    # a (0..5 = Sex x Race), split (0/1/2).
    df["label_idx"] = df["y"].astype(int)
    df["group_idx"] = df["a"].astype(int)
    df["split"] = df["split"].map({0: "train", 1: "val", 2: "test"})
    if df["split"].isna().any():
        sys.exit("CheXpert: unexpected split value")
    df["image_path"] = df["filename"].astype(str)
    df["dataset"] = "chexpert"
    df["source"] = "chexpert"

    if verify_exists:
        missing = [p for p in df["image_path"].head(20)
                   if not Path(p).is_file()]
        if missing:
            print(f"[s18:chexpert] WARNING: {len(missing)}/20 sampled "
                  f"image_paths missing (sample: {missing[0]})")

    _save(df, labels=CHEXPERT_LABEL_MAP, groups=CHEXPERT_GROUP_MAP,
          stem="chexpert")
    return df


def _build_text_manifest(csv_path, label_map, group_map, stem,
                         verify_exists=False):
    if not csv_path.is_file():
        sys.exit(f"Missing metadata: {csv_path}. Run SubpopBench download first.")
    df = pd.read_csv(csv_path)
    df["label_idx"] = df["y"].astype(int)
    df["group_idx"] = (df["y"].astype(int) * len(set(df["a"])) + df["a"].astype(int))
    df["split"] = df["split"].map({0: "train", 1: "val", 2: "test"})
    if df["split"].isna().any():
        sys.exit(f"{stem}: unexpected split value")
    df["image_path"] = df["filename"].astype(str)
    df["dataset"] = stem
    df["source"] = stem
    _save(df, labels=label_map, groups=group_map, stem=stem)
    return df


def build_civilcomments_manifest(verify_exists: bool = False) -> pd.DataFrame:
    return _build_text_manifest(
        CIVILCOMMENTS_CSV, CIVILCOMMENTS_LABEL_MAP, CIVILCOMMENTS_GROUP_MAP,
        "civilcomments", verify_exists)


def build_multinli_manifest(verify_exists: bool = False) -> pd.DataFrame:
    return _build_text_manifest(
        MULTINLI_CSV, MULTINLI_LABEL_MAP, MULTINLI_GROUP_MAP,
        "multinli", verify_exists)


BUILDERS = {
    "derm": build_derm_manifest,
    "celeba": build_celeba_manifest,
    "waterbirds": build_waterbirds_manifest,
    "metashift": build_metashift_manifest,
    "imagenetbg": build_imagenetbg_manifest,
    "living17": build_living17_manifest,
    "chexpert": build_chexpert_manifest,
    "civilcomments": build_civilcomments_manifest,
    "multinli": build_multinli_manifest,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default=",".join(BUILDERS.keys()),
                    help="Subset of: " + ", ".join(BUILDERS.keys()))
    ap.add_argument("--verify-exists", action="store_true",
                    help="Sample 20-50 paths per dataset and stat them. "
                         "Does not stat all paths (expensive on networked FS).")
    ap.add_argument("--derm-mix-sources", action="store_true",
                    help="For the derm manifest, mix all four sources "
                         "(Fitzpatrick17k + DermaCon-IN + Mafi-Bangladesh "
                         "+ DDI) into one pool and re-split 70/15/15 "
                         "stratified by (group, label). Without this "
                         "flag, DDI is held out as test (cross-site "
                         "domain generalization setup).")
    args = ap.parse_args()

    requested = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for d in requested:
        if d not in BUILDERS:
            sys.exit(f"Unknown dataset '{d}'. Known: {list(BUILDERS.keys())}")

    print(f"[s18] manifest dir: {MANIFEST_DIR}")
    for name in requested:
        print("=" * 60)
        print(f"[s18] building {name} manifest...")
        kwargs = {"verify_exists": args.verify_exists}
        if name == "derm":
            kwargs["mix_sources"] = args.derm_mix_sources
        BUILDERS[name](**kwargs)


if __name__ == "__main__":
    main()
