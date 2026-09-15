"""Path resolution for MOF pipeline stages."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline_config import PipelineConfig


def data_source_root(cfg: PipelineConfig) -> Path:
    """Root where pre-built dataset pickles are read from."""
    if cfg.data.mode == "prebuilt":
        if not cfg.data.prebuilt_root:
            raise ValueError(
                "data.mode is 'prebuilt' but data.prebuilt_root is not set."
            )
        return Path(cfg.data.prebuilt_root)
    return Path(cfg.output_root)


def artifacts_root(cfg: PipelineConfig) -> Path:
    """Writable run root for hyper_sweep, xai_results, and analysis outputs."""
    return Path(cfg.output_root)


def dataset_dir(cfg: PipelineConfig, task_id: str) -> Path:
    subdir = cfg.dataset_subdir(task_id)
    return data_source_root(cfg) / "datasets" / subdir


def shared_dir(cfg: PipelineConfig) -> Path:
    return data_source_root(cfg) / "shared"


def resolve_artifact_paths(cfg: PipelineConfig) -> dict[str, Path]:
    """Standard MOF artifact locations under the writable run root."""
    root = artifacts_root(cfg)
    return {
        "data_root": root,
        "sweep_root": root / "hyper_sweep",
        "xai_root": root / "xai_results",
        "fe_dir": root / "xai_results" / "hyper_sweep_fe",
        "analysis_dir": root / "analysis",
        "bootstrap_csv": root / "hyper_sweep" / "bootstrap_results.csv",
        "bootstrap_scored_csv": root / "hyper_sweep" / "bootstrap_results_with_scores.csv",
        "fe_summary_csv": root / "xai_results" / "hyper_sweep_fe" / "sweep_fe_summary.csv",
        "three_obj_csv": root / "analysis" / "three_objective_dataset.csv",
        "objective_inputs_csv": root / "analysis" / "objective_inputs.csv",
    }
