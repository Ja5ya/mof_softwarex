#!/usr/bin/env python3
"""
Build per-diagnosis psychiatric pickles and install a shared normal pool.

Reads preprocessed multi-lead ECG arrays (NumPy), a manifest CSV, and an
optional demographics table. All paths, column names, lead count, signal
length, and diagnosis codes are defined in a YAML configuration file.

Outputs (under output_root):
  shared/recs_normal.pkl
  datasets/icd10_<CODE>_vs_normal/recs_psych.pkl
  datasets/icd10_<CODE>_vs_normal/recs_normal.pkl  (symlink)
  datasets/build_manifest.json
  datasets/pipeline_config.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ecg_utils import adjust_signal_length, normalize_gender, validate_lead_count
from pipeline_config import PipelineConfig, add_config_arg, load_config


def load_demographics(cfg: PipelineConfig) -> dict[str, dict]:
    demo_path = cfg.input.demographics_path
    cols = cfg.input.demographics_columns
    if not demo_path or not os.path.isfile(demo_path):
        return {}

    if demo_path.lower().endswith((".xlsx", ".xls")):
        demo = pd.read_excel(demo_path, dtype=str)
    else:
        demo = pd.read_csv(demo_path, dtype=str)

    demo.columns = demo.columns.str.strip()
    lookup: dict[str, dict] = {}
    for _, row in demo.iterrows():
        ecg_id = str(row[cols.ecg_id_column]).strip()
        entry: dict = {
            "gender": normalize_gender(row.get(cols.sex_column)),
            "diagnosis": str(row.get(cols.diagnosis_column, "")).strip(),
        }
        if cols.age_column and cols.age_column in row:
            try:
                entry["age"] = float(row[cols.age_column])
            except (TypeError, ValueError):
                entry["age"] = np.nan
        else:
            entry["age"] = np.nan
        lookup[ecg_id] = entry
    return lookup


def discover_codes(manifest: pd.DataFrame, cfg: PipelineConfig) -> list[str]:
    col = cfg.input.manifest_columns.diagnosis_column
    counts = manifest[col].value_counts()
    codes = [
        str(code)
        for code, n in counts.items()
        if n >= cfg.diagnoses.min_count and str(code).strip()
        and str(code).upper() != "NOT FOUND"
    ]
    return sorted(codes)


def resolve_diagnosis_codes(manifest: pd.DataFrame, cfg: PipelineConfig) -> list[str]:
    if cfg.diagnoses.auto_discover:
        codes = discover_codes(manifest, cfg)
        if not codes:
            raise SystemExit(
                f"No diagnoses met min_count={cfg.diagnoses.min_count} in manifest."
            )
        return codes
    codes = [c.strip() for c in cfg.diagnoses.codes if c.strip()]
    if not codes:
        raise SystemExit("Set diagnoses.codes or enable diagnoses.auto_discover.")
    return codes


def validate_normal_pool_config(cfg: PipelineConfig, positive_codes: list[str]) -> None:
    np_cfg = cfg.normal_pool
    if np_cfg.cases_only and np_cfg.negative_diagnosis:
        raise SystemExit(
            "normal_pool.cases_only and normal_pool.negative_diagnosis "
            "cannot both be set."
        )
    if not np_cfg.negative_diagnosis:
        return
    if np_cfg.pkl or np_cfg.external_data_root:
        raise SystemExit(
            "normal_pool.negative_diagnosis cannot be combined with "
            "normal_pool.pkl or normal_pool.external_data_root."
        )
    neg = np_cfg.negative_diagnosis.strip()
    if neg in positive_codes:
        raise SystemExit(
            f"negative_diagnosis {neg!r} is also listed in diagnoses.codes."
        )


def resolve_normal_pkl(cfg: PipelineConfig) -> str:
    if cfg.normal_pool.pkl:
        path = cfg.normal_pool.pkl
        if os.path.isfile(path):
            return path
        raise SystemExit(f"normal_pool.pkl not found: {path}")

    root = cfg.normal_pool.external_data_root
    if root:
        candidate = os.path.join(
            root,
            cfg.normal_pool.external_shared_subdir,
            cfg.normal_pool.external_pkl_name,
        )
        if os.path.isfile(candidate):
            return candidate

    raise SystemExit(
        "Normal controls are required for case-vs-normal training.\n"
        "Set normal_pool.pkl or normal_pool.external_data_root in the config."
    )


def install_normal_pool(normal_src: str, shared_dir: str) -> str:
    os.makedirs(shared_dir, exist_ok=True)
    dst = os.path.join(shared_dir, "recs_normal.pkl")
    if os.path.isfile(dst):
        print(f"Reusing existing {dst}")
        return dst
    print(f"Installing normal pool: {normal_src} -> {dst}")
    shutil.copy2(normal_src, dst)
    return dst


def write_normal_pool(recs_normal: list[dict], shared_dir: str) -> str:
    os.makedirs(shared_dir, exist_ok=True)
    dst = os.path.join(shared_dir, "recs_normal.pkl")
    with open(dst, "wb") as f:
        pickle.dump(recs_normal, f, protocol=4)
    print(f"Wrote negative pool: {dst} ({len(recs_normal)} records)")
    return dst


def load_psych_records(
    manifest: pd.DataFrame,
    demo_lookup: dict[str, dict],
    psych_code: str,
    cfg: PipelineConfig,
    *,
    label: int = 1,
) -> list[dict]:
    mcols = cfg.input.manifest_columns
    records = []
    subset = manifest[manifest[mcols.diagnosis_column] == psych_code]

    for _, row in subset.iterrows():
        ecg_id = str(row[mcols.ecg_id_column])
        npy_path = os.path.join(
            cfg.input.artifacts_dir,
            f"{ecg_id}{cfg.input.npy_suffix}",
        )
        if not os.path.isfile(npy_path):
            print(f"  WARNING: missing {npy_path}")
            continue

        sig = np.load(npy_path)
        if not validate_lead_count(sig, cfg.ecg.n_leads):
            print(
                f"  WARNING: expected {cfg.ecg.n_leads} leads, "
                f"got shape {sig.shape} for {ecg_id}"
            )
            continue

        demo = demo_lookup.get(ecg_id, {})
        subject_id = int(ecg_id) if ecg_id.isdigit() else hash(ecg_id) % (10**9)
        records.append(
            {
                "signal": adjust_signal_length(sig, cfg.ecg),
                "subject_id": subject_id,
                "ecg_id": ecg_id,
                "age": demo.get("age", np.nan),
                "gender": demo.get("gender"),
                "label": label,
            }
        )
    return records


def build_negative_pool_from_diagnosis(
    manifest: pd.DataFrame,
    demo_lookup: dict[str, dict],
    cfg: PipelineConfig,
) -> tuple[list[dict], str]:
    neg_code = cfg.normal_pool.negative_diagnosis
    if not neg_code:
        raise ValueError("negative_diagnosis is not set")
    neg_code = neg_code.strip()
    mcol = cfg.input.manifest_columns.diagnosis_column
    n_manifest = int((manifest[mcol] == neg_code).sum())
    if n_manifest == 0:
        raise SystemExit(
            f"negative_diagnosis {neg_code!r} has no rows in manifest."
        )
    print(
        f"\nNegative class from diagnosis {neg_code} "
        f"({n_manifest} rows in manifest)"
    )
    recs = load_psych_records(
        manifest, demo_lookup, neg_code, cfg, label=0
    )
    if not recs:
        raise SystemExit(
            f"No records loaded for negative_diagnosis {neg_code!r} "
            "(check .npy files under artifacts_dir)."
        )
    return recs, f"diagnosis:{neg_code}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    add_config_arg(p)
    p.add_argument(
        "--list-codes",
        action="store_true",
        help="Print diagnosis codes from manifest and exit.",
    )
    p.add_argument(
        "--cases-only",
        action="store_true",
        help="Build psychiatric pickles only; skip normal pool (overrides config).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if not cfg.needs_dataset_build():
        raise SystemExit(
            "data.mode is 'prebuilt' — dataset build is skipped. "
            "Point data.prebuilt_root at your pickles and start from training:\n"
            "  python src/train_multi_f_array.py --config <config> --task-id 0"
        )

    out_root = os.path.abspath(cfg.output_root)
    manifest_path = cfg.manifest_path()
    shared_dir = os.path.join(out_root, "shared")
    ds_root = os.path.join(out_root, "datasets")
    os.makedirs(ds_root, exist_ok=True)

    if not os.path.isfile(manifest_path):
        raise SystemExit(f"Missing manifest: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    codes = resolve_diagnosis_codes(manifest, cfg)
    cfg.diagnoses.codes = codes
    validate_normal_pool_config(cfg, codes)

    if args.list_codes:
        col = cfg.input.manifest_columns.diagnosis_column
        print(manifest[col].value_counts().to_string())
        print(f"\nSelected codes ({len(codes)}): {codes}")
        return

    demo_lookup = load_demographics(cfg)
    print(f"Dataset   : {cfg.name}")
    print(f"Artifacts : {cfg.input.artifacts_dir}")
    print(f"Manifest  : {manifest_path}")
    print(f"Output    : {out_root}")
    print(f"ECG       : {cfg.ecg.n_leads} leads, len {cfg.ecg.signal_len}")
    print(f"Codes ({len(codes)}): {codes}")
    print(f"Manifest rows: {len(manifest)}  |  demo rows: {len(demo_lookup)}")

    cases_only = args.cases_only or cfg.normal_pool.cases_only
    neg_diag = (cfg.normal_pool.negative_diagnosis or "").strip() or None
    normal_src: str | None = None
    normal_pkl: str | None = None
    recs_normal: list[dict] = []

    if cases_only:
        print("Cases-only mode: skipping normal pool (training/XAI need normals later).")
    elif neg_diag:
        recs_normal, normal_src = build_negative_pool_from_diagnosis(
            manifest, demo_lookup, cfg
        )
        normal_pkl = write_normal_pool(recs_normal, shared_dir)
        print(
            f"Negative pool: {len(recs_normal)} records "
            f"(label 0, diagnosis {neg_diag})"
        )
    else:
        normal_src = resolve_normal_pkl(cfg)
        normal_pkl = install_normal_pool(normal_src, shared_dir)
        with open(normal_pkl, "rb") as f:
            recs_normal = pickle.load(f)
        print(f"Normal pool: {len(recs_normal)} records from {normal_src}")

    build_manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": cfg.name,
        "config_path": cfg.config_path,
        "n_leads": cfg.ecg.n_leads,
        "signal_len": cfg.ecg.signal_len,
        "cases_only": cases_only,
        "negative_diagnosis": neg_diag,
        "normal_source": normal_src,
        "n_normal_loaded": len(recs_normal),
        "codes": [],
    }

    mcol = cfg.input.manifest_columns.diagnosis_column
    for psych_code in codes:
        sub = cfg.dataset_subdir(psych_code)
        out_dir = os.path.join(ds_root, sub)
        os.makedirs(out_dir, exist_ok=True)

        n_cases = int((manifest[mcol] == psych_code).sum())
        print(f"\n{psych_code}: {n_cases} rows in manifest")

        psych_path = os.path.join(out_dir, "recs_psych.pkl")
        if os.path.isfile(psych_path):
            with open(psych_path, "rb") as f:
                recs_psych = pickle.load(f)
            print(f"  Reusing {psych_path} ({len(recs_psych)} records)")
        else:
            recs_psych = load_psych_records(
                manifest, demo_lookup, psych_code, cfg
            )
            with open(psych_path, "wb") as f:
                pickle.dump(recs_psych, f, protocol=4)
            print(f"  Wrote {psych_path} ({len(recs_psych)} records)")

        if not cases_only and normal_pkl:
            link_target = os.path.relpath(normal_pkl, start=out_dir)
            normal_link = os.path.join(out_dir, "recs_normal.pkl")
            Path(normal_link).unlink(missing_ok=True)
            os.symlink(link_target, normal_link)

        build_manifest["codes"].append(
            {
                "psych_code": psych_code,
                "dataset_dir": sub,
                "n_psych_manifest": n_cases,
                "n_psych_loaded": len(recs_psych),
                "n_normal_shared": len(recs_normal),
            }
        )

    cfg.save_snapshot(os.path.join(ds_root, "pipeline_config.json"))
    with open(os.path.join(ds_root, "build_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(build_manifest, f, indent=2)

    print(f"\nWrote {ds_root}/pipeline_config.json")
    print(f"Wrote {ds_root}/build_manifest.json")
    print(f"Sweep tasks after build: {cfg.sweep_task_count()}")
    print("Done.")


if __name__ == "__main__":
    main()
