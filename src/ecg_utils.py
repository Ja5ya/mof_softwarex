"""Shared ECG signal and demographic-matching utilities."""

from __future__ import annotations

import numpy as np

from pipeline_config import DemographicMatchingConfig, ECGConfig


def adjust_signal_length(
    signal: np.ndarray,
    ecg: ECGConfig,
) -> np.ndarray:
    """Resize (n_leads, n_samples) to ecg.signal_len."""
    target = ecg.signal_len
    n_leads, n_samples = signal.shape
    if n_samples == target:
        return signal.astype(np.float32)

    policy = ecg.length_policy
    if n_samples > target:
        if policy == "truncate_start":
            return signal[:, :target].astype(np.float32)
        # default: center_crop
        start = (n_samples - target) // 2
        return signal[:, start : start + target].astype(np.float32)

    if policy == "pad":
        pad = target - n_samples
        return np.pad(signal, ((0, 0), (0, pad)), mode="constant").astype(np.float32)

    raise ValueError(
        f"Signal length {n_samples} < target {target} and length_policy={policy!r} "
        "does not allow padding."
    )


def validate_lead_count(signal: np.ndarray, n_leads: int) -> bool:
    return signal.ndim == 2 and signal.shape[0] == n_leads


def normalize_gender(sex: str | None) -> str | None:
    if sex is None or (isinstance(sex, float) and np.isnan(sex)):
        return None
    s = str(sex).strip().upper()
    if s in {"F", "FEMALE", "W"}:
        return "F"
    if s in {"M", "MALE"}:
        return "M"
    return s or None


def _ages_compatible(
    rec_n: dict,
    rec_p: dict,
    cfg: DemographicMatchingConfig,
) -> bool:
    if not cfg.match_age:
        return True
    age_n, age_p = rec_n.get("age"), rec_p.get("age")
    if age_n is None or age_p is None:
        return True
    try:
        fn, fp = float(age_n), float(age_p)
    except (TypeError, ValueError):
        return True
    if np.isnan(fn) or np.isnan(fp):
        return True
    return abs(fn - fp) <= cfg.age_tolerance


def demographic_match(
    recs_psych: list[dict],
    recs_normal: list[dict],
    cfg: DemographicMatchingConfig,
) -> tuple[list[dict], list[dict]]:
    """Match psychiatric cases to normal controls without replacement."""
    rng = np.random.default_rng(cfg.random_state)
    pool = list(range(len(recs_normal)))
    matched_p, matched_n = [], []

    for rec_p in recs_psych:
        candidates = []
        for i in pool:
            if cfg.match_gender and recs_normal[i].get("gender") != rec_p.get("gender"):
                continue
            if not _ages_compatible(recs_normal[i], rec_p, cfg):
                continue
            candidates.append(i)
        if not candidates:
            continue
        chosen = int(rng.choice(candidates))
        pool.remove(chosen)
        matched_p.append(rec_p)
        matched_n.append(recs_normal[chosen])

    return matched_p, matched_n
