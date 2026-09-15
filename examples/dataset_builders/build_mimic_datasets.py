#!/usr/bin/env python3
"""
MIMIC-IV ECG dataset builder (Layer 1, optional).

Builds recs_psych.pkl + recs_normal.pkl per ICD-10 F* code from MIMIC-IV-ECG
WFDB waveforms and ICD extension labels. Copied into this repository for
SoftwareX self-containment; the parent NeurIPS repo is not required at runtime.

Usage:
  source configs/env.mimic_build.example   # edit paths first
  python examples/dataset_builders/build_mimic_datasets.py \\
    --output-root /path/to/prebuilt_run \\
    --codes F329,F419,F17200,F17210,F0390,F1010

Then point MOF prebuilt config at the output directory:
  data.prebuilt_root: /path/to/prebuilt_run
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import pickle
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

warnings.filterwarnings("ignore")

# ── Same constants as notebook Part C ─────────────────────────────────────
N_LEADS = 12
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SIGNAL_LEN = 5000

EXPECTED_POLARITY = {
    "I":   +1,
    "II":  +1,
    "III": +1,
    "aVR": -1,
    "aVL": +1,
    "aVF": +1,
    "V1":  -1,
    "V2":  -1,
    "V3":  +1,
    "V4":  +1,
    "V5":  +1,
    "V6":  +1,
}

# Part A: report classification (same lists as notebook)
REPORT_COLS = [f"report_{i}" for i in range(18)]
NORMAL_PHRASES = [
    "normal sinus rhythm",
    "normal ecg",
    "normal tracing",
]
ABNORMAL_KEYWORDS = [
    "fibrillation",
    "flutter",
    "tachycardia",
    "bradycardia",
    "block",
    "infarct",
    "ischemi",
    "hypertrophy",
    "elevation",
    "depression",
    "inversion",
    "prolonged",
    "wide",
    "aberrant",
    "paced",
    "pacemaker",
    "arrhythmia",
    "ectopic",
    "premature",
    "extrasystole",
    "bigeminy",
    "trigeminy",
    "escape",
    "accessory",
    "delta",
    "wolff",
    "pre-excitation",
    "axis deviation",
    "hemiblock",
    "fascicular",
    "abnormal",
    "abnormality",
    "pattern",
    "change",
    "low voltage",
    "poor r-wave",
    "poor r wave",
]
UNCERTAINTY_WORDS = [
    "possible",
    "probable",
    "likely",
    "borderline",
    "questionable",
    "equivocal",
    "suspected",
    "uncertain",
    "consider",
    "cannot exclude",
    "cannot rule out",
]


def extract_codes(x):
    if pd.isna(x) or x == "":
        return []
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except Exception:
        return []


def classify_record(row, report_cols: list[str]) -> str:
    reports = []
    for col in report_cols:
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            cleaned = str(val).lower().strip()
            if cleaned.endswith("."):
                cleaned = cleaned[:-1].strip()
            if cleaned.startswith("- "):
                cleaned = cleaned[2:].strip()
            if cleaned.startswith("summary:"):
                cleaned = cleaned[8:].strip()
            reports.append(cleaned)
    if not reports:
        return "unknown"
    full_text = " ".join(reports)
    has_normal = any(p in full_text for p in NORMAL_PHRASES)
    has_abnorm = any(k in full_text for k in ABNORMAL_KEYWORDS)
    has_uncert = any(u in full_text for u in UNCERTAINTY_WORDS)
    if has_uncert:
        return "uncertain"
    if has_normal and not has_abnorm:
        return "normal"
    return "abnormal"


def fft_lowpass(sig, fs=500, cutoff=60):
    freqs = np.fft.rfftfreq(len(sig), d=1 / fs)
    coeffs = np.fft.rfft(sig)
    coeffs[freqs > cutoff] = 0
    return np.fft.irfft(coeffs, n=len(sig))


def polarity_correct(sig, lead_name):
    if lead_name == "II":
        centred = sig - np.mean(sig)
        return -sig if np.mean(centred**99) < 0 else sig
    p95 = np.percentile(sig, 95)
    p5  = np.percentile(sig, 5)
    mp  = np.mean(sig[sig >= p95])
    mt  = np.mean(sig[sig <= p5])
    dom = +1 if abs(mp) >= abs(mt) else -1
    exp = EXPECTED_POLARITY.get(lead_name, +1)
    return -sig if dom != exp else sig


def standardize(sig):
    mu, sd = np.mean(sig), np.std(sig)
    res = (sig - mu) / sd if sd > 0 else sig - mu
    return np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)


def pad_or_trim(sig, length=SIGNAL_LEN):
    if len(sig) >= length:
        return sig[:length]
    return np.pad(sig, (0, length - len(sig)), mode="constant")


def preprocess_lead(raw, lead_name, fs):
    s = polarity_correct(raw.astype(np.float32), lead_name)
    s = fft_lowpass(s, fs=fs, cutoff=60)
    s = standardize(s)
    return pad_or_trim(s, SIGNAL_LEN).astype(np.float32)


def load_all_leads(row, base_ecg_dir: str):
    rel = row["file_name"]
    if "files/" in rel:
        rel = rel.split("files/", 1)[1]
    record_dir = os.path.join(base_ecg_dir, os.path.dirname(rel))
    if not os.path.isdir(record_dir):
        return None, None
    candidates = {}
    for f in os.listdir(record_dir):
        root, ext = os.path.splitext(f)
        if ext.lower() in (".hea", ".dat"):
            candidates.setdefault(root, set()).add(ext.lower())
    for root, exts in candidates.items():
        if ".hea" in exts and ".dat" in exts:
            sig, fields = wfdb.rdsamp(os.path.join(record_dir, root))
            return sig, fields.get("fs", 500)
    return None, None


def is_signal_good(sig) -> bool:
    if sig is None:
        return False
    count_bad_leads = 0
    for lead_idx in range(sig.shape[1]):
        v = float(np.var(sig[:, lead_idx]))
        if v >= 2.0:
            return False
        if v >= 0.5:
            count_bad_leads += 1
    return count_bad_leads < 2


def load_and_filter(row, base_ecg_dir: str):
    sig, fs = load_all_leads(row, base_ecg_dir)
    if sig is None or sig.shape[1] < N_LEADS:
        return None, None
    if not is_signal_good(sig):
        return None, None
    leads = []
    for li, ln in enumerate(LEAD_NAMES):
        leads.append(preprocess_lead(sig[:, li], ln, fs))
    return np.stack(leads, axis=0), fs


def load_group(df_group, label: int, base_ecg_dir: str, label_name: str):
    records = []
    total = len(df_group)
    rejected_quality = 0
    for idx, (_, row) in enumerate(df_group.iterrows()):
        if idx % 300 == 0:
            print(f"  {label_name}: {idx}/{total}", end="\r")
        sig, fs = load_and_filter(row, base_ecg_dir)
        if sig is None:
            rejected_quality += 1
            continue
        records.append(
            {
                "signal":     sig,
                "subject_id": int(row["subject_id"]),
                "age":        row.get("age", np.nan),
                "gender":     row.get("gender", None),
                "label":      label,
            }
        )
    print(f"  {label_name}: loaded={len(records)}, rejected={rejected_quality}    ")
    return records


def dataset_subdir(psych_code: str) -> str:
    return f"icd10_{psych_code}_vs_normal"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Run root (creates shared/, datasets/).",
    )
    p.add_argument(
        "--codes",
        type=str,
        default="F329,F419,F17200,F0390,F17210,F1010",
        help="Comma-separated ICD-10 codes (exact match, sole F* in all_diag_all).",
    )
    p.add_argument(
        "--meas-path",
        type=str,
        default=os.environ.get("TRIAL6_MEASUREMENTS_CSV", ""),
        help="Machine measurements CSV (or set TRIAL6_MEASUREMENTS_CSV).",
    )
    p.add_argument(
        "--labels-path",
        type=str,
        default=os.environ.get("TRIAL6_LABELS_CSV", ""),
        help="ICD-10 labels / metadata CSV (or set TRIAL6_LABELS_CSV).",
    )
    p.add_argument(
        "--base-ecg-dir",
        type=str,
        default=os.environ.get("TRIAL6_WFDB_ROOT", ""),
        help="WFDB root directory (or set TRIAL6_WFDB_ROOT).",
    )
    p.add_argument(
        "--max-codes",
        type=int,
        default=10,
        help="Exclude normal patients with more than this many ICD codes (default: 10).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    missing = [
        name
        for name, val in (
            ("--meas-path / TRIAL6_MEASUREMENTS_CSV", args.meas_path),
            ("--labels-path / TRIAL6_LABELS_CSV", args.labels_path),
            ("--base-ecg-dir / TRIAL6_WFDB_ROOT", args.base_ecg_dir),
        )
        if not (val or "").strip()
    ]
    if missing:
        raise SystemExit(
            "Missing required inputs: " + ", ".join(missing) + ". "
            "Pass CLI flags or set the TRIAL6_* environment variables."
        )
    codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    if not codes:
        raise SystemExit("No codes in --codes")

    out_root   = os.path.abspath(args.output_root)
    shared_dir = os.path.join(out_root, "shared")
    ds_root    = os.path.join(out_root, "datasets")
    os.makedirs(shared_dir, exist_ok=True)
    os.makedirs(ds_root, exist_ok=True)

    # ── Load and classify ECG reports ─────────────────────────────────────
    print("Loading machine_measurements...")
    meas = pd.read_csv(args.meas_path)
    report_cols = [c for c in REPORT_COLS if c in meas.columns]
    print("Classifying ECG reports...")
    meas["ecg_class"] = meas.apply(
        lambda r: classify_record(r, report_cols), axis=1
    )

    normal_meas = (
        meas[meas["ecg_class"] == "normal"]
        .sort_values(["subject_id", "study_id"])
        .groupby("subject_id", as_index=False)
        .first()
    )
    print(f"Unique patients with purely normal ECG report: {len(normal_meas)}")

    # ── Load labels CSV ────────────────────────────────────────────────────
    print("Loading labels CSV (all study rows)...")
    labels_full = pd.read_csv(args.labels_path)

    df_all = (
        labels_full.sort_values(["subject_id", "study_id"])
        .groupby("subject_id", as_index=False)
        .first()
    )

    normal_df = normal_meas[["subject_id", "study_id"]].merge(
        labels_full[["subject_id", "study_id", "file_name", "age", "gender"]],
        on=["subject_id", "study_id"],
        how="inner",
    )
    normal_df["group"] = "NORMAL"
    print(f"Normal rows with file_name:                    {len(normal_df)}")

    # Pre-compute per-patient code lists once (used by all filter steps)
    all_diag = df_all["all_diag_all"].apply(extract_codes)
    all_diag_lookup = df_all.set_index("subject_id")["all_diag_all"].apply(extract_codes)

    f_codes = all_diag.apply(
        lambda c: [str(x) for x in c if str(x).startswith("F")]
    )

    # ── Step 1: Remove any F* psychiatric code ────────────────────────────
    normal_df = normal_df[
        ~normal_df["subject_id"].isin(
            df_all[f_codes.apply(len) > 0]["subject_id"]
        )
    ].copy()
    print(f"Normal after removing any F* psychiatric codes: {len(normal_df)}")

    # ── Step 2: Remove any I* circulatory/cardiac code ────────────────────
    i_codes_per_patient = (
        df_all[["subject_id", "all_diag_all"]]
        .set_index("subject_id")["all_diag_all"]
        .apply(extract_codes)
        .apply(lambda c: [str(x) for x in c if str(x).startswith("I")])
    )
    cardiac_subjects = set(
        i_codes_per_patient[i_codes_per_patient.apply(len) > 0].index
    )
    normal_df = normal_df[~normal_df["subject_id"].isin(cardiac_subjects)].copy()
    print(f"Normal after removing any I* cardiac codes:     {len(normal_df)}")

    # ── Step 3: Remove any G* nervous system code ─────────────────────────
    g_codes_per_patient = (
        df_all[["subject_id", "all_diag_all"]]
        .set_index("subject_id")["all_diag_all"]
        .apply(extract_codes)
        .apply(lambda c: [str(x) for x in c if str(x).startswith("G")])
    )
    neuro_subjects = set(
        g_codes_per_patient[g_codes_per_patient.apply(len) > 0].index
    )
    normal_df = normal_df[~normal_df["subject_id"].isin(neuro_subjects)].copy()
    print(f"Normal after removing any G* nervous system codes: {len(normal_df)}")

    # ── Step 4: Remove high comorbidity burden (> max_codes total) ────────
    high_burden_subjects = set(
        all_diag_lookup[
            all_diag_lookup.index.isin(normal_df["subject_id"]) &
            (all_diag_lookup.apply(len) > args.max_codes)
        ].index
    )
    normal_df = normal_df[~normal_df["subject_id"].isin(high_burden_subjects)].copy()
    print(f"Normal after removing >{args.max_codes} code burden:            {len(normal_df)}")
    print(f"\nFinal clean normal pool: {len(normal_df)} patients")

    # ── Build or reuse recs_normal.pkl ────────────────────────────────────
    normal_pkl = os.path.join(shared_dir, "recs_normal.pkl")
    if os.path.isfile(normal_pkl):
        print(f"\nReusing existing {normal_pkl}")
        with open(normal_pkl, "rb") as f:
            recs_normal = pickle.load(f)
        print(f"  Normal records loaded: {len(recs_normal)}")
    else:
        print("\nLoading NORMAL signals (quality filter + preprocess)...")
        recs_normal = load_group(
            normal_df, label=0, base_ecg_dir=args.base_ecg_dir, label_name="NORMAL"
        )
        with open(normal_pkl, "wb") as f:
            pickle.dump(recs_normal, f, protocol=4)
        print(f"Wrote {normal_pkl}  ({len(recs_normal)} records)")

    # ── Build manifest ─────────────────────────────────────────────────────
    manifest = {
        "built_at_utc":  datetime.now(timezone.utc).isoformat(),
        "meas_path":     args.meas_path,
        "labels_path":   args.labels_path,
        "base_ecg_dir":  args.base_ecg_dir,
        "max_codes":     args.max_codes,
        "n_normal_pool": len(normal_df),
        "n_normal_loaded": len(recs_normal),
        "exclusion_steps": {
            "removed_F_codes":         "any F* in all_diag_all",
            "removed_I_codes":         "any I* in all_diag_all",
            "removed_G_codes":         "any G* in all_diag_all",
            "removed_high_burden":     f">{args.max_codes} total ICD codes",
        },
        "codes": [],
    }

    # ── Per-diagnosis psych pickles ────────────────────────────────────────
    for psych_code in codes:
        sub     = dataset_subdir(psych_code)
        out_dir = os.path.join(ds_root, sub)
        os.makedirs(out_dir, exist_ok=True)

        psych_mask = f_codes.apply(
            lambda fc: len(fc) == 1 and fc[0] == psych_code
        )
        df_psych = df_all[psych_mask][
            ["subject_id", "study_id", "file_name", "age", "gender"]
        ].copy()
        df_psych["group"] = psych_code
        print(f"\n{psych_code} (single F* label == {psych_code}): {len(df_psych)} patients")

        psych_path = os.path.join(out_dir, "recs_psych.pkl")
        if os.path.isfile(psych_path):
            print(f"  Reusing existing {psych_path}")
            with open(psych_path, "rb") as f:
                recs_psych = pickle.load(f)
            print(f"    loaded records: {len(recs_psych)}")
            psych_reused = True
        else:
            print(f"  Loading {psych_code} signals...")
            recs_psych = load_group(
                df_psych,
                label=1,
                base_ecg_dir=args.base_ecg_dir,
                label_name=psych_code,
            )
            with open(psych_path, "wb") as f:
                pickle.dump(recs_psych, f, protocol=4)
            psych_reused = False

        # Symlink recs_normal.pkl into the per-diagnosis directory
        link_target = os.path.relpath(normal_pkl, start=out_dir)
        normal_link = os.path.join(out_dir, "recs_normal.pkl")
        Path(normal_link).unlink(missing_ok=True)
        os.symlink(link_target, normal_link)

        entry = {
            "psych_code":      psych_code,
            "dataset_dir":     sub,
            "n_psych_loaded":  len(recs_psych),
            "n_normal_shared": len(recs_normal),
        }
        manifest["codes"].append(entry)
        status = "Reused" if psych_reused else "Saved"
        print(f"  {status} {psych_path}")
        print(f"  Symlink recs_normal.pkl -> {link_target}")

    # ── Write manifest ─────────────────────────────────────────────────────
    man_path = os.path.join(ds_root, "build_manifest.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest: {man_path}")
    print("\nDone.")
    print(f"  Normal pool (after all filters): {len(normal_df)}")
    print(f"  Normal records loaded from disk: {len(recs_normal)}")
    for entry in manifest["codes"]:
        print(f"  {entry['psych_code']}: {entry['n_psych_loaded']} psych records")


if __name__ == "__main__":
    main()