"""Data contracts between user dataset builders and the MOF pipeline."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

RECORD_REQUIRED_KEYS = frozenset({"signal", "label"})
RECORD_OPTIONAL_KEYS = frozenset({"gender", "age", "subject_id", "record_id"})

POSITIVE_PKL_NAME = "recs_psych.pkl"
NORMAL_PKL_NAME = "recs_normal.pkl"


def validate_record(record: dict[str, Any], n_leads: int | None = None) -> None:
    """Raise ValueError if a pickle record does not match the ECG dataset contract."""
    missing = RECORD_REQUIRED_KEYS - record.keys()
    if missing:
        raise ValueError(f"Record missing required keys: {sorted(missing)}")

    signal = record["signal"]
    if not isinstance(signal, np.ndarray):
        raise ValueError(f"signal must be numpy.ndarray, got {type(signal)!r}")
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2-D (n_channels, n_samples), got shape {signal.shape}")
    if n_leads is not None and signal.shape[0] != n_leads:
        raise ValueError(f"expected {n_leads} channels, got {signal.shape[0]}")

    label = record["label"]
    if label not in (0, 1):
        raise ValueError(f"label must be 0 or 1, got {label!r}")


def validate_dataset_layout(
    dataset_path: Path,
    *,
    n_leads: int | None = None,
    sample_records: int = 3,
    positive_name: str = POSITIVE_PKL_NAME,
    require_normal: bool = True,
) -> dict[str, Any]:
    """
    Validate a case-vs-normal dataset directory.

    Returns summary dict with record counts and signal shape.
    """
    dataset_path = Path(dataset_path)
    positive_pkl = dataset_path / positive_name
    if not positive_pkl.is_file():
        raise FileNotFoundError(f"Missing positive pickle: {positive_pkl}")

    normal_pkl = dataset_path / NORMAL_PKL_NAME
    if require_normal and not normal_pkl.is_file() and not normal_pkl.is_symlink():
        raise FileNotFoundError(f"Missing normal pickle: {normal_pkl}")

    with open(positive_pkl, "rb") as f:
        positives = pickle.load(f)
    if not positives:
        raise ValueError(f"No records in {positive_pkl}")

    for rec in positives[:sample_records]:
        validate_record(rec, n_leads=n_leads)

    summary: dict[str, Any] = {
        "positive_pkl": str(positive_pkl),
        "n_positive": len(positives),
        "signal_shape": tuple(positives[0]["signal"].shape),
        "signal_dtype": str(positives[0]["signal"].dtype),
    }

    if normal_pkl.is_file() or normal_pkl.is_symlink():
        resolved = normal_pkl.resolve()
        with open(resolved, "rb") as f:
            normals = pickle.load(f)
        for rec in normals[: min(sample_records, len(normals))]:
            validate_record(rec, n_leads=n_leads)
        summary["normal_pkl"] = str(resolved)
        summary["n_normal"] = len(normals)

    return summary
