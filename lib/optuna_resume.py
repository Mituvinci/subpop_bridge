"""Persistent-storage + replay helpers for BRIDGE Optuna studies.

Two resume flows are supported:

  --resume-dir DIR        Reuse an existing run directory. Opens the SQLite DB
                          already inside it; Optuna loads prior trials natively
                          via load_if_exists=True. This is the normal
                          walltime-restart flow.

  --resume-from JSONL     One-shot import of an older trials.jsonl (written by
                          a pre-persistence run) into a fresh study via
                          add_trial(). Use this exactly once to migrate the
                          pre-SQLite runs; after that the SQLite DB is the
                          source of truth.

Both flags are optional. Default behaviour (neither flag set) creates a new
timestamped run with a fresh SQLite DB, matching the old behaviour byte-for-
byte except that the DB now exists.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import optuna
from optuna.distributions import FloatDistribution
from optuna.trial import TrialState, create_trial


def create_or_resume_study(
    db_path: Path,
    study_name: str,
    direction: str,
    sampler,
    resume_from_jsonl: Optional[Path] = None,
    lambda_key: str = "lambdas",
) -> optuna.Study:
    """Create a SQLite-backed Optuna study, optionally seeded from older trials.

    db_path
        Absolute path to the SQLite file. Parent dir is created if needed.
    study_name
        Stable name for the study. Must match across resumes.
    direction
        "minimize" or "maximize".
    sampler
        An Optuna sampler (TPESampler etc.).
    resume_from_jsonl
        If provided, read trial records from this file and add each to the
        study via add_trial() - but only if the study is currently empty.
        Each record must have keys: "trial" (int), lambda_key (dict[str, float]),
        "objective" (float).
    lambda_key
        The JSON field holding the {param_name: value} dict.

    Returns
        An Optuna study ready for study.optimize().
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{db_path}"

    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        storage=storage_url,
        study_name=study_name,
        load_if_exists=True,
    )

    existing = study.get_trials(deepcopy=False)
    existing_count = len(existing)

    if resume_from_jsonl is None:
        if existing_count:
            print(f"[resume] study '{study_name}' already has {existing_count} "
                  f"trial(s) in {db_path}; continuing.")
        return study

    if existing_count > 0:
        print(f"[resume] study '{study_name}' already has {existing_count} "
              f"trial(s); skipping --resume-from replay to avoid duplicates.")
        return study

    jsonl_path = Path(resume_from_jsonl)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"--resume-from jsonl not found: {jsonl_path}")

    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    records.sort(key=lambda r: r.get("trial", -1))

    added = 0
    for rec in records:
        params = rec.get(lambda_key)
        if not isinstance(params, dict):
            continue
        obj = rec.get("objective")
        if obj is None:
            continue
        distributions = {k: FloatDistribution(0.0, 1.0) for k in params.keys()}
        trial = create_trial(
            params=params,
            distributions=distributions,
            value=float(obj),
            state=TrialState.COMPLETE,
        )
        study.add_trial(trial)
        added += 1

    print(f"[resume] replayed {added} trials from {jsonl_path} "
          f"into study '{study_name}'.")
    return study


def remaining_trials(study: optuna.Study, target_total: int) -> int:
    """Return how many more trials to run so the study ends with target_total.

    Makes --n-trials idempotent across resumes: passing the same value on
    every resubmit targets the same total, rather than adding n_trials new
    trials each time.
    """
    have = len(study.get_trials(deepcopy=False))
    return max(0, target_total - have)
