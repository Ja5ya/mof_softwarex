"""
Multi-objective analysis helpers for MOF-ECG (stages 4–5).

Bootstrap stability (f_P, f_S), merge of XAI f_E results, and feasibility-frontier
computations. Used by notebooks/ and callable from scripts.

All paths and sweep grids are resolved from PipelineConfig (YAML or snapshot).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

from mof.objectives import (
    LEGACY_TASK_COLUMN,
    apply_fe_scores,
    compute_fp_fs_scores,
    objectives_from_dict,
)
from mof.paths import resolve_artifact_paths
from pipeline_config import PipelineConfig, load_config, load_config_or_snapshot

EPS = 1e-12
PERF_METRICS = ("acc", "precision", "recall", "specificity")
JOIN_KEY_SUFFIX = ("architecture", "learning_rate", "dropout_rate")


def _task_column(cfg: PipelineConfig) -> str:
    return cfg.analysis.task_column


def join_keys(cfg: PipelineConfig) -> tuple[str, ...]:
    return (_task_column(cfg), *JOIN_KEY_SUFFIX)


def _normalize_task_column(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Rename legacy ``psych_code`` to configured ``analysis.task_column`` when needed."""
    task_col = _task_column(cfg)
    if task_col == LEGACY_TASK_COLUMN:
        return df
    if LEGACY_TASK_COLUMN in df.columns and task_col not in df.columns:
        return df.rename(columns={LEGACY_TASK_COLUMN: task_col})
    return df


def resolve_paths(cfg: PipelineConfig) -> dict[str, Path]:
    """Backward-compatible alias for artifact path resolution."""
    return resolve_artifact_paths(cfg)


def lr_tag(lr: float) -> str:
    return f"lr{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def dr_tag(dr: float) -> str:
    return f"dr{dr}"


def sweep_run_dir(cfg: PipelineConfig, code: str, arch: str, lr: float, dr: float) -> Path:
    paths = resolve_paths(cfg)
    return paths["sweep_root"] / code / arch / f"{lr_tag(lr)}_{dr_tag(dr)}"


def discover_completed_runs(cfg: PipelineConfig) -> tuple[list[dict[str, Any]], list[tuple], int]:
    """Return (completed configs with paths, missing grid tuples)."""
    completed: list[dict[str, Any]] = []
    missing: list[tuple] = []
    grid = cfg.build_sweep_grid()
    expected = len(grid)

    for entry in grid:
        code, arch, lr, dr = (
            entry["psych_code"],
            entry["architecture"],
            entry["learning_rate"],
            entry["dropout_rate"],
        )
        run_dir = sweep_run_dir(cfg, code, arch, lr, dr)
        npz_path = run_dir / "test_predictions.npz"
        metrics_path = run_dir / "metrics.json"
        if npz_path.is_file() and metrics_path.is_file():
            completed.append(
                {
                    **entry,
                    "run_dir": str(run_dir),
                    "npz_path": str(npz_path),
                    "metrics_path": str(metrics_path),
                }
            )
        else:
            missing.append((code, arch, lr, dr))

    return completed, missing, expected


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    if len(np.unique(y_true)) < 2:
        return {m: float("nan") for m in PERF_METRICS}
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity_score(y_true, y_pred),
    }


def bootstrap_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
    threshold: float = 0.5,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n_samp = len(y_true)
    buckets = {m: [] for m in PERF_METRICS}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_samp, size=n_samp)
        mets = compute_metrics(y_true[idx], y_score[idx], threshold=threshold)
        for key, val in mets.items():
            buckets[key].append(val)

    out: dict[str, float] = {}
    for key, vals in buckets.items():
        arr = np.array(vals, dtype=float)
        out[f"{key}_mean"] = float(np.nanmean(arr))
        out[f"{key}_std"] = float(np.nanstd(arr))
    return out


def run_bootstrap(
    cfg: PipelineConfig,
    n_bootstrap: int = 1000,
    random_state: int | None = None,
    threshold: float | None = None,
    progress_every: int = 30,
) -> pd.DataFrame:
    """Bootstrap all completed sweep runs; write bootstrap_results.csv."""
    random_state = cfg.training.random_state if random_state is None else random_state
    threshold = cfg.training.threshold if threshold is None else threshold

    completed, missing, expected = discover_completed_runs(cfg)
    print(f"Completed runs: {len(completed)} / {expected}")
    if missing:
        print(f"Missing runs: {len(missing)} (first: {missing[0]})")

    rows: list[dict[str, Any]] = []
    for i, entry in enumerate(completed):
        if progress_every and (i == 0 or (i + 1) % progress_every == 0):
            print(
                f"  [{i + 1}/{len(completed)}] {entry['psych_code']} | "
                f"{entry['architecture']} | lr={entry['learning_rate']} | "
                f"dr={entry['dropout_rate']}"
            )

        data = np.load(entry["npz_path"])
        y_true = data["y_true"].astype(int)
        y_score = data["y_score"].astype(float)

        with open(entry["metrics_path"], encoding="utf-8") as f:
            metrics = json.load(f)

        bs = bootstrap_metrics(
            y_true,
            y_score,
            n_bootstrap=n_bootstrap,
            seed=random_state,
            threshold=threshold,
        )

        rows.append(
            {
                "psych_code": entry["psych_code"],
                "architecture": entry["architecture"],
                "learning_rate": entry["learning_rate"],
                "dropout_rate": entry["dropout_rate"],
                "epochs_ran": metrics.get("epochs_ran", np.nan),
                "n_test": int(len(y_true)),
                "test_acc_point": metrics["test"]["acc"],
                "test_precision_point": metrics["test"]["precision"],
                "test_recall_point": metrics["test"]["recall"],
                "test_specificity_point": metrics["test"]["specificity"],
                **{f"test_{k}": v for k, v in bs.items()},
            }
        )

    df = pd.DataFrame(rows)
    paths = resolve_paths(cfg)
    paths["sweep_root"].mkdir(parents=True, exist_ok=True)
    df.to_csv(paths["bootstrap_csv"], index=False)
    print(f"Saved: {paths['bootstrap_csv']}")
    return df


def _objectives_cfg(cfg: PipelineConfig):
    obj_cfg = objectives_from_dict(cfg.objectives_raw)
    task_col = _task_column(cfg)
    if obj_cfg.f_P.task_column == LEGACY_TASK_COLUMN and task_col != LEGACY_TASK_COLUMN:
        obj_cfg.f_P.task_column = task_col
        obj_cfg.f_S.task_column = task_col
    return obj_cfg


def load_objective_inputs(
    cfg: PipelineConfig,
    bootstrap_path: Path | None = None,
    auto_merge_fe: bool = True,
) -> pd.DataFrame:
    """Join bootstrap raw metrics with XAI columns before applying f_P/f_S/f_E."""
    paths = resolve_paths(cfg)
    boot_path = bootstrap_path or paths["bootstrap_csv"]
    if not boot_path.is_file():
        raise FileNotFoundError(f"Bootstrap CSV not found: {boot_path}")

    boot = pd.read_csv(boot_path)
    boot = _normalize_task_column(boot, cfg)
    try:
        fe_df = load_fe_table(cfg, auto_merge=auto_merge_fe)
        df = boot.merge(fe_df, on=list(join_keys(cfg)), how="left")
    except FileNotFoundError:
        df = boot.copy()

    paths["analysis_dir"].mkdir(parents=True, exist_ok=True)
    df.to_csv(paths["objective_inputs_csv"], index=False)
    return df.reset_index(drop=True)


def parse_lr_dr(run_str: str) -> tuple[float | None, float | None]:
    try:
        parts = str(run_str).split("_dr")
        return float(parts[0].replace("lr", "")), float(parts[1])
    except Exception:
        return None, None


def merge_sweep_fe_csvs(cfg: PipelineConfig, write_summary: bool = True) -> Path | None:
    """Merge per-code sweep_fe_<CODE>.csv into sweep_fe_summary.csv."""
    paths = resolve_paths(cfg)
    fe_dir = paths["fe_dir"]
    if not fe_dir.is_dir():
        print(f"No XAI sweep directory: {fe_dir}")
        return None

    sources = sorted(fe_dir.glob("sweep_fe_*.csv"))
    sources = [p for p in sources if p.name != "sweep_fe_summary.csv"]
    if not sources:
        print(f"No per-code f_E CSVs in {fe_dir}")
        return None

    combined = pd.concat([pd.read_csv(p) for p in sources], ignore_index=True)
    out = paths["fe_summary_csv"]
    if write_summary:
        out.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out, index=False)
        print(f"Merged {len(sources)} files → {out} ({len(combined)} rows)")
    return out


def load_fe_table(cfg: PipelineConfig, auto_merge: bool = True) -> pd.DataFrame:
    paths = resolve_paths(cfg)
    summary = paths["fe_summary_csv"]
    if not summary.is_file() and auto_merge:
        merge_sweep_fe_csvs(cfg)
    if not summary.is_file():
        raise FileNotFoundError(
            f"f_E summary not found: {summary}. Run run_xai.py --hyper_sweep_fe first."
        )

    fe_raw = pd.read_csv(summary)
    parsed = fe_raw["sweep_run"].apply(parse_lr_dr)
    fe_raw[["learning_rate", "dropout_rate"]] = pd.DataFrame(
        parsed.tolist(), index=fe_raw.index
    )
    task_col = _task_column(cfg)
    fe_raw = fe_raw.rename(columns={"code": task_col, "sweep_arch": "architecture"})
    drop_cols = [c for c in ("sweep_run",) if c in fe_raw.columns]
    out = fe_raw.drop(columns=drop_cols)
    out = apply_fe_scores(out, _objectives_cfg(cfg))
    component_cols = _objectives_cfg(cfg).f_E.components
    has_components = [c for c in component_cols if c in out.columns]
    if has_components:
        out = out.dropna(subset=has_components)
    elif "f_E" in out.columns:
        out = out.dropna(subset=["f_E"])
    keys = join_keys(cfg)
    front = [c for c in keys if c in out.columns]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest]


def load_three_objective_dataset(
    cfg: PipelineConfig,
    recall_min: float | None = None,
    auto_merge_fe: bool = True,
    bootstrap_path: Path | None = None,
) -> pd.DataFrame:
    """Join bootstrap results with f_E; filter by minimum recall."""
    paths = resolve_paths(cfg)
    boot_path = bootstrap_path or paths["bootstrap_csv"]
    if not boot_path.is_file():
        raise FileNotFoundError(
            f"Bootstrap CSV not found: {boot_path}. Run notebook 01 or run_bootstrap()."
        )

    boot = pd.read_csv(boot_path)
    boot = _normalize_task_column(boot, cfg)
    obj_cfg = _objectives_cfg(cfg)
    boot = compute_fp_fs_scores(boot, codes=cfg.diagnoses.codes, cfg=obj_cfg)

    fe_df = load_fe_table(cfg, auto_merge=auto_merge_fe)
    df = boot.merge(fe_df, on=list(join_keys(cfg)), how="left")
    df = apply_fe_scores(df, obj_cfg)
    df = df.dropna(subset=["f_P", "f_S", "f_E"]).copy()

    recall_min = cfg.analysis.recall_min if recall_min is None else recall_min
    recall_metric = cfg.analysis.recall_metric
    if recall_min is not None and recall_metric in df.columns:
        df = df[df[recall_metric] >= recall_min].copy()

    paths["analysis_dir"].mkdir(parents=True, exist_ok=True)
    load_objective_inputs(cfg, bootstrap_path=boot_path, auto_merge_fe=auto_merge_fe)
    df.to_csv(paths["three_obj_csv"], index=False)
    return df.reset_index(drop=True)


def adaptive_delta(series: pd.Series, method: str = "median") -> float:
    unique_vals = np.sort(series.dropna().unique())
    if len(unique_vals) < 2:
        return 0.02
    gaps = np.diff(unique_vals)
    delta = float(np.median(gaps) if method == "median" else np.mean(gaps))
    return max(delta, 1e-6)


def is_feasible(df_code: pd.DataFrame, k_s: float, k_e: float) -> bool:
    return int(((df_code["f_S"] >= k_s) & (df_code["f_E"] >= k_e)).sum()) > 0


def compute_feasibility_frontier(
    df_code: pd.DataFrame,
    n_grid: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ks_vals = np.linspace(df_code["f_S"].min(), df_code["f_S"].max(), n_grid)
    ke_candidates = np.sort(df_code["f_E"].unique())

    records = []
    for ks in ks_vals:
        max_ke = np.nan
        for ke in reversed(ke_candidates):
            if is_feasible(df_code, ks, ke):
                max_ke = ke
                break
        records.append({"K_S": ks, "max_K_E": max_ke})

    frontier_df = pd.DataFrame(records)
    frontier_df["feasible"] = ~frontier_df["max_K_E"].isna()

    pareto = frontier_df.dropna(subset=["max_K_E"]).copy()
    pareto = pareto.sort_values("K_S", ascending=False).reset_index(drop=True)
    pareto["running_max_KE"] = pareto["max_K_E"].cummax()
    pareto["on_frontier"] = pareto["max_K_E"] >= pareto["running_max_KE"]
    return frontier_df, pareto[pareto["on_frontier"]].sort_values("K_S")


def reliable_shadow_price(
    df_code: pd.DataFrame,
    k_s: float,
    k_e: float,
    n_min: int = 3,
) -> dict[str, Any] | None:
    feas = df_code[(df_code["f_S"] >= k_s) & (df_code["f_E"] >= k_e)]
    if len(feas) == 0:
        return None

    delta_s = adaptive_delta(df_code["f_S"])
    delta_e = adaptive_delta(df_code["f_E"])

    best = feas.loc[feas["f_P"].idxmax()]
    fp_now = float(feas["f_P"].max())

    slack_s = float(best["f_S"]) - k_s
    slack_e = float(best["f_E"]) - k_e

    feas_s = df_code[(df_code["f_S"] >= k_s + delta_s) & (df_code["f_E"] >= k_e)]
    feas_e = df_code[(df_code["f_S"] >= k_s) & (df_code["f_E"] >= k_e + delta_e)]

    lam_s = (
        max(0.0, -(float(feas_s["f_P"].max()) - fp_now) / delta_s)
        if len(feas_s) > 0
        else float("inf")
    )
    lam_e = (
        max(0.0, -(float(feas_e["f_P"].max()) - fp_now) / delta_e)
        if len(feas_e) > 0
        else float("inf")
    )

    return {
        "fP_star": fp_now,
        "fS_star": float(best["f_S"]),
        "fE_star": float(best["f_E"]),
        "lambda_S": lam_s,
        "lambda_E": lam_e,
        "delta_S": delta_s,
        "delta_E": delta_e,
        "slack_S": slack_s,
        "slack_E": slack_e,
        "n_feasible": len(feas),
        "n_feas_S": len(feas_s),
        "n_feas_E": len(feas_e),
        "reliable_S": (slack_s > delta_s) and (len(feas_s) >= n_min),
        "reliable_E": (slack_e > delta_e) and (len(feas_e) >= n_min),
        "best": best,
    }


def pareto_front(df: pd.DataFrame, col_x: str = "f_P", col_y: str = "f_S") -> pd.Series:
    """Boolean mask for non-dominated points (maximise both columns)."""
    vals = df[[col_x, col_y]].values
    n = len(vals)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            if (vals[j, 0] >= vals[i, 0] and vals[j, 1] >= vals[i, 1]) and (
                vals[j, 0] > vals[i, 0] or vals[j, 1] > vals[i, 1]
            ):
                dominated[i] = True
                break
    return ~dominated


def run_bootstrap_pipeline(
    cfg: PipelineConfig,
    n_bootstrap: int = 1000,
    random_state: int | None = None,
    threshold: float | None = None,
    f_s_method: str = "inv_std",
) -> pd.DataFrame:
    """
    Stage 4 entry point: bootstrap all completed sweep runs and write both CSVs.

    Writes:
      hyper_sweep/bootstrap_results.csv
      hyper_sweep/bootstrap_results_with_scores.csv
    """
    bs_df = run_bootstrap(
        cfg,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        threshold=threshold,
    )
    scored = compute_fp_fs_scores(
        bs_df,
        codes=cfg.diagnoses.codes,
        cfg=_objectives_cfg(cfg),
        f_s_method=f_s_method,
    )
    paths = resolve_paths(cfg)
    scored.to_csv(paths["bootstrap_scored_csv"], index=False)
    print(f"Saved: {paths['bootstrap_scored_csv']}")
    return scored


def load_bootstrap_scored(
    cfg: PipelineConfig,
    *,
    force: bool = False,
    n_bootstrap: int = 1000,
    f_s_method: str = "inv_std",
) -> pd.DataFrame:
    """
    Load cached bootstrap scores, or compute them if missing.

    Reads ``hyper_sweep/bootstrap_results_with_scores.csv`` when present (unless
    ``force=True``). Falls back to scoring ``bootstrap_results.csv``, then to a
    full bootstrap pass over ``test_predictions.npz`` files.
    """
    paths = resolve_paths(cfg)
    scored_path = paths["bootstrap_scored_csv"]
    raw_path = paths["bootstrap_csv"]

    if not force and scored_path.is_file():
        df = pd.read_csv(scored_path)
        if {"f_P_raw", "f_S_raw", "f_P", "f_S"}.issubset(df.columns):
            print(f"Loaded cached scores: {scored_path} ({len(df)} rows)")
            return df

    if not force and raw_path.is_file():
        print(f"Loaded raw bootstrap: {raw_path} — computing f_P / f_S")
        bs_df = pd.read_csv(raw_path)
        scored = compute_fp_fs_scores(
            bs_df,
            codes=cfg.diagnoses.codes,
            cfg=_objectives_cfg(cfg),
            f_s_method=f_s_method,
        )
        scored.to_csv(scored_path, index=False)
        print(f"Saved: {scored_path}")
        return scored

    print("No cached bootstrap CSV found — running full bootstrap pass")
    return run_bootstrap_pipeline(
        cfg,
        n_bootstrap=n_bootstrap,
        f_s_method=f_s_method,
    )


def load_analysis_config(
    config_path: str | None = None,
    data_root: str | None = None,
) -> PipelineConfig:
    """Load YAML or datasets/pipeline_config.json snapshot."""
    env_root = data_root or os.environ.get("MOF_DATA_ROOT")
    return load_config_or_snapshot(config_path, data_root=env_root)


def cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument(
        "command",
        choices=["merge-fe", "bootstrap", "status"],
        help="Analysis command.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args()

    cfg = load_analysis_config(args.config, args.data_root)
    if args.data_root:
        cfg.output_root = args.data_root

    if args.command == "merge-fe":
        merge_sweep_fe_csvs(cfg)
    elif args.command == "bootstrap":
        run_bootstrap_pipeline(cfg, n_bootstrap=args.n_bootstrap)
    elif args.command == "status":
        completed, missing, expected = discover_completed_runs(cfg)
        paths = resolve_paths(cfg)
        print(f"config:      {cfg.config_path or cfg.name}")
        print(f"data_root:   {paths['data_root']}")
        print(f"codes:       {cfg.diagnoses.codes}")
        print(f"sweep:       {len(completed)} / {expected} completed")
        print(f"bootstrap:   {paths['bootstrap_csv'].is_file()}")
        print(f"f_E summary: {paths['fe_summary_csv'].is_file()}")


if __name__ == "__main__":
    cli()
