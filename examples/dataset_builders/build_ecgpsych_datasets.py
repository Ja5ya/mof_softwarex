#!/usr/bin/env python3
"""
Psychiatry-ECG (Kaggle / Tasci) dataset builder (Layer 1).

Builds the MOF pickle contract for one binary task:

  Schizophrenia (label=1)  vs  Bipolar + Depression (label=0)

Same acquisition source for both classes (no external healthy controls).
No subject IDs in the public release → beat-level records only.

Usage (from repo root):
  python examples/dataset_builders/build_ecgpsych_datasets.py \\
    --data-root "/path/to/Psychiatry_ECG" \\
    --output-root artifacts/prebuilt \\
    --signal-len 384

Then:
  source configs/env.ecgpsych
  # MOF_PREBUILT_ROOT must point at --output-root
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.io as sio

N_LEADS = 12
TASK_CODE = "SCZ"
CLASS_DIRS = {
    "Schizophrenia": ("positive", 1),
    "Bipolar": ("negative", 0),
    "Depression": ("negative", 0),
}


def dataset_subdir(psych_code: str) -> str:
    return f"icd10_{psych_code}_vs_normal"


def resample_beat(sig_t12: np.ndarray, target_len: int) -> np.ndarray:
    """(T, 12) -> (12, target_len) via linear interpolation per lead + z-score."""
    t, leads = sig_t12.shape
    if leads != N_LEADS:
        raise ValueError(f"Expected 12 leads, got {leads}")
    if t < 2:
        raise ValueError("Need at least 2 samples to interpolate")

    x_old = np.linspace(0.0, 1.0, t)
    x_new = np.linspace(0.0, 1.0, target_len)
    out = np.empty((N_LEADS, target_len), dtype=np.float32)
    for j in range(N_LEADS):
        y = np.interp(x_new, x_old, sig_t12[:, j].astype(np.float64))
        mu, sd = float(np.mean(y)), float(np.std(y))
        if sd > 0:
            y = (y - mu) / sd
        else:
            y = y - mu
        out[j] = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return out


def load_mat_beat(path: Path, expected_label: int) -> np.ndarray:
    """Return raw (T, 12) float array; check embedded label."""
    m = sio.loadmat(path)
    lab = int(np.squeeze(m["label"]))
    if lab != expected_label:
        raise ValueError(f"Label mismatch {path}: expected {expected_label}, mat has {lab}")
    sig = np.asarray(m["sinyal"], dtype=np.float64)
    if sig.ndim != 2 or sig.shape[1] != N_LEADS:
        raise ValueError(f"Expected (T, 12), got {sig.shape} for {path}")
    return sig


def folder_mat_label(folder: str) -> int:
    """Embedded MATLAB label: 1=Bipolar, 2=Depression, 3=Schizophrenia."""
    return {"Bipolar": 1, "Depression": 2, "Schizophrenia": 3}[folder]


def collect_records(
    data_root: Path,
    signal_len: int,
) -> tuple[list[dict], list[dict], dict]:
    positives: list[dict] = []
    negatives: list[dict] = []
    lengths: list[int] = []
    per_folder: dict[str, int] = {}

    for folder, (role, bin_label) in CLASS_DIRS.items():
        dpath = data_root / folder
        if not dpath.is_dir():
            raise FileNotFoundError(f"Missing class folder: {dpath}")
        mat_label = folder_mat_label(folder)
        paths = sorted(dpath.glob("*.mat"))
        per_folder[folder] = 0
        for i, p in enumerate(paths):
            raw = load_mat_beat(p, mat_label)
            lengths.append(raw.shape[0])
            sig = resample_beat(raw, signal_len)
            rec = {
                "signal": sig,
                "label": bin_label,
                "record_id": p.name,
                "source_class": folder,
                # No subject_id / age / gender in public release
            }
            if role == "positive":
                positives.append(rec)
            else:
                negatives.append(rec)
            per_folder[folder] += 1
            if (i + 1) % 500 == 0:
                print(f"  {folder}: {i + 1}/{len(paths)}", flush=True)
        print(f"  {folder}: loaded {per_folder[folder]} beats", flush=True)

    stats = {
        "per_folder": per_folder,
        "n_positive_scz": len(positives),
        "n_negative_other_psych": len(negatives),
        "raw_T_min": int(min(lengths)) if lengths else None,
        "raw_T_median": int(np.median(lengths)) if lengths else None,
        "raw_T_max": int(max(lengths)) if lengths else None,
        "signal_len": signal_len,
        "signal_shape": [N_LEADS, signal_len],
    }
    return positives, negatives, stats


def write_layout(
    output_root: Path,
    positives: list[dict],
    negatives: list[dict],
    stats: dict,
    data_root: Path,
) -> Path:
    output_root = output_root.resolve()
    shared = output_root / "shared"
    ds_dir = output_root / "datasets" / dataset_subdir(TASK_CODE)
    shared.mkdir(parents=True, exist_ok=True)
    ds_dir.mkdir(parents=True, exist_ok=True)

    normal_pkl = shared / "recs_normal.pkl"
    psych_pkl = ds_dir / "recs_psych.pkl"
    link_pkl = ds_dir / "recs_normal.pkl"

    with open(normal_pkl, "wb") as f:
        pickle.dump(negatives, f, protocol=4)
    with open(psych_pkl, "wb") as f:
        pickle.dump(positives, f, protocol=4)

    if link_pkl.exists() or link_pkl.is_symlink():
        link_pkl.unlink()
    os.symlink(os.path.relpath(normal_pkl, start=ds_dir), link_pkl)

    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "SCZ_vs_other_psych",
        "positive_class": "Schizophrenia",
        "negative_classes": ["Bipolar", "Depression"],
        "note": (
            "Negative pool is other psychiatric diagnoses from the same Kaggle "
            "release, not healthy ECG-normal controls. No subject IDs available."
        ),
        "data_root": str(data_root),
        "task_code": TASK_CODE,
        "dataset_subdir": dataset_subdir(TASK_CODE),
        **stats,
        "n_normal_written": len(negatives),
        "n_psych_written": len(positives),
    }
    with open(output_root / "datasets" / "build_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {psych_pkl} ({len(positives)} records)")
    print(f"Wrote {normal_pkl} ({len(negatives)} records)")
    print(f"Symlink {link_pkl} -> {normal_pkl}")
    return ds_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/path/to/Psychiatry_ECG"),
        help="Root with Bipolar/, Depression/, Schizophrenia/ folders",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            os.environ.get(
                "MOF_PREBUILT_ROOT",
                str(Path(__file__).resolve().parents[2] / "artifacts" / "prebuilt"),
            )
        ),
        help="MOF prebuilt root (shared/ + datasets/)",
    )
    parser.add_argument(
        "--signal-len",
        type=int,
        default=384,
        help="Resampled beat length (must match configs/ecgpsych.yaml ecg.signal_len)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Skip build; only validate an existing output-root layout",
    )
    args = parser.parse_args()

    repo_src = Path(__file__).resolve().parents[2] / "src"
    sys.path.insert(0, str(repo_src))
    from mof.contracts import validate_dataset_layout

    if args.validate_only:
        ds = args.output_root / "datasets" / dataset_subdir(TASK_CODE)
        summary = validate_dataset_layout(ds, n_leads=N_LEADS)
        print(json.dumps(summary, indent=2))
        return 0

    if not args.data_root.is_dir():
        raise FileNotFoundError(f"--data-root not found: {args.data_root}")

    print(f"Data root:   {args.data_root}")
    print(f"Output root: {args.output_root}")
    print(f"signal_len:  {args.signal_len}")
    print("Loading and resampling beats...")
    positives, negatives, stats = collect_records(args.data_root, args.signal_len)
    print(
        f"Raw T min/median/max: {stats['raw_T_min']} / "
        f"{stats['raw_T_median']} / {stats['raw_T_max']}"
    )
    print(
        f"Binary counts: SCZ(+)={stats['n_positive_scz']}  "
        f"other_psych(-)={stats['n_negative_other_psych']}"
    )

    ds_dir = write_layout(
        args.output_root, positives, negatives, stats, args.data_root
    )
    summary = validate_dataset_layout(ds_dir, n_leads=N_LEADS)
    print("Validation OK:")
    print(json.dumps(summary, indent=2))
    print(
        "\nNote: with match_gender/match_age disabled, demographic_match will "
        f"pair min(n_psych, n_normal) = {min(len(positives), len(negatives))} "
        "balanced pairs (extra SCZ beats unused)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
