"""Pluggable multi-objective scoring on pipeline artifact tables."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-12
DEFAULT_PERF_METRICS = ("acc", "precision", "recall", "specificity")
# Legacy column name in training/bootstrap CSVs; override via analysis.task_column (e.g. task_id).
TASK_COLUMN = "psych_code"
LEGACY_TASK_COLUMN = TASK_COLUMN


@dataclass
class FPConfig:
    metrics: list[str] = field(
        default_factory=lambda: [f"test_{m}_mean" for m in DEFAULT_PERF_METRICS]
    )
    weights: list[float] | None = None
    normalize: str = "per_task_minmax"
    task_column: str = TASK_COLUMN


@dataclass
class FSConfig:
    method: str = "inv_std"
    metrics: list[str] = field(
        default_factory=lambda: [f"test_{m}_std" for m in DEFAULT_PERF_METRICS]
    )
    mean_metrics: list[str] = field(
        default_factory=lambda: [f"test_{m}_mean" for m in DEFAULT_PERF_METRICS]
    )
    task_column: str = TASK_COLUMN


@dataclass
class FEConfig:
    components: list[str] = field(
        default_factory=lambda: [
            "continuity_valid",
            "compactness_valid",
            "contrastivity_valid",
        ]
    )
    weights: list[float] | None = None


@dataclass
class ObjectivesConfig:
    f_P: FPConfig = field(default_factory=FPConfig)
    f_S: FSConfig = field(default_factory=FSConfig)
    f_E: FEConfig = field(default_factory=FEConfig)


def _weighted_mean(values: pd.DataFrame, weights: list[float] | None) -> pd.Series:
    if weights is None:
        return values.mean(axis=1)
    if len(weights) != len(values.columns):
        raise ValueError("weights length must match number of metrics")
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return values.mul(w, axis=1).sum(axis=1)


def compute_fp_fs_scores(
    bs_df: pd.DataFrame,
    codes: list[str] | None = None,
    cfg: ObjectivesConfig | None = None,
    f_s_method: str | None = None,
) -> pd.DataFrame:
    """Add f_P_raw, f_S_raw, f_P, f_S using configurable formulas."""
    cfg = cfg or ObjectivesConfig()
    fp_cfg = cfg.f_P
    fs_cfg = cfg.f_S
    method = f_s_method or fs_cfg.method

    out = bs_df.copy()
    mean_cols = [c for c in fp_cfg.metrics if c in out.columns]
    std_cols = [c for c in fs_cfg.metrics if c in out.columns]
    if not mean_cols:
        raise ValueError(f"No f_P metrics found in dataframe: {fp_cfg.metrics}")
    if not std_cols:
        raise ValueError(f"No f_S metrics found in dataframe: {fs_cfg.metrics}")

    out["f_P_raw"] = _weighted_mean(out[mean_cols], fp_cfg.weights)

    if method == "inv_std":
        out["f_S_raw"] = _weighted_mean(out[std_cols], None)
        higher_is_better = False
    elif method == "cv":
        mean_cols_fs = [c for c in fs_cfg.mean_metrics if c in out.columns]
        cv_vals = out[std_cols].values / (out[mean_cols_fs].values + EPS)
        out["f_S_raw"] = pd.DataFrame(cv_vals, index=out.index, columns=std_cols).mean(axis=1)
        higher_is_better = False
    elif method == "recip_clipped":
        recip = 1.0 / (out[std_cols] + EPS)
        clip_val = recip.quantile(0.95)
        out["f_S_raw"] = recip.clip(upper=clip_val, axis=1).mean(axis=1)
        higher_is_better = True
    else:
        raise ValueError(f"Unknown f_S method: {method}")

    task_col = fp_cfg.task_column
    codes = codes or sorted(out[task_col].dropna().unique())
    out["f_P"] = np.nan
    out["f_S"] = np.nan

    if fp_cfg.normalize == "per_task_minmax":
        for task in codes:
            mask = out[task_col] == task
            if not mask.any():
                continue
            if mask.sum() < 2:
                out.loc[mask, "f_P"] = 0.5
                out.loc[mask, "f_S"] = 0.5
                continue
            fp_min = out.loc[mask, "f_P_raw"].min()
            fp_max = out.loc[mask, "f_P_raw"].max()
            out.loc[mask, "f_P"] = (
                out.loc[mask, "f_P_raw"] - fp_min
            ) / (fp_max - fp_min + EPS)
            fs_min = out.loc[mask, "f_S_raw"].min()
            fs_max = out.loc[mask, "f_S_raw"].max()
            normalised = (
                out.loc[mask, "f_S_raw"] - fs_min
            ) / (fs_max - fs_min + EPS)
            out.loc[mask, "f_S"] = (
                normalised if higher_is_better else 1.0 - normalised
            )
    elif fp_cfg.normalize == "none":
        out["f_P"] = out["f_P_raw"]
        out["f_S"] = out["f_S_raw"] if higher_is_better else -out["f_S_raw"]
    else:
        raise ValueError(f"Unknown normalize mode: {fp_cfg.normalize}")

    return out


def apply_fe_scores(
    df: pd.DataFrame,
    cfg: ObjectivesConfig | None = None,
) -> pd.DataFrame:
    """Add or overwrite ``f_E`` as a weighted mean of configured component columns."""
    cfg = cfg or ObjectivesConfig()
    fe_cfg = cfg.f_E
    cols = [c for c in fe_cfg.components if c in df.columns]
    if not cols:
        if "f_E" in df.columns:
            return df.copy()
        raise ValueError(
            f"No f_E components found in dataframe. "
            f"Expected any of {fe_cfg.components}; got columns {list(df.columns)}"
        )

    out = df.copy()
    out["f_E"] = _weighted_mean(out[cols], fe_cfg.weights)
    return out


def compute_f_E(
    continuity_val: float,
    compactness_val: float,
    contrastivity_val: float,
    cfg: FEConfig | None = None,
) -> float:
    """Scalar helper for the default ECG-XAI component names."""
    row = apply_fe_scores(
        pd.DataFrame(
            [
                {
                    "continuity_valid": continuity_val,
                    "compactness_valid": compactness_val,
                    "contrastivity_valid": contrastivity_val,
                }
            ]
        ),
        ObjectivesConfig(f_E=cfg or FEConfig()),
    )
    return float(row["f_E"].iloc[0])


def apply_objectives(
    df: pd.DataFrame,
    cfg: ObjectivesConfig | None = None,
    *,
    codes: list[str] | None = None,
) -> pd.DataFrame:
    """Apply f_P, f_S, and f_E using YAML-configured formulas."""
    scored = compute_fp_fs_scores(df, codes=codes, cfg=cfg)
    return apply_fe_scores(scored, cfg)


def objectives_from_dict(raw: dict[str, Any] | None) -> ObjectivesConfig:
    if not raw:
        return ObjectivesConfig()

    fp_raw = raw.get("f_P", {})
    fs_raw = raw.get("f_S", {})
    fe_raw = raw.get("f_E", {})

    return ObjectivesConfig(
        f_P=FPConfig(
            metrics=fp_raw.get(
                "metrics",
                [f"test_{m}_mean" for m in DEFAULT_PERF_METRICS],
            ),
            weights=fp_raw.get("weights"),
            normalize=fp_raw.get("normalize", "per_task_minmax"),
            task_column=fp_raw.get("task_column", TASK_COLUMN),
        ),
        f_S=FSConfig(
            method=fs_raw.get("method", "inv_std"),
            metrics=fs_raw.get(
                "metrics",
                [f"test_{m}_std" for m in DEFAULT_PERF_METRICS],
            ),
            mean_metrics=fs_raw.get(
                "mean_metrics",
                [f"test_{m}_mean" for m in DEFAULT_PERF_METRICS],
            ),
            task_column=fs_raw.get("task_column", TASK_COLUMN),
        ),
        f_E=FEConfig(
            components=fe_raw.get(
                "components",
                ["continuity_valid", "compactness_valid", "contrastivity_valid"],
            ),
            weights=fe_raw.get("weights"),
        ),
    )
