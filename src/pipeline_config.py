"""
Configuration loader for the multi-objective framework (MOF) ECG use case.

Dataset construction can be external (prebuilt pickles) or in-repo (build mode).
All paths, task codes, and objective formulas are declared in YAML.

Example:
    python src/build_datasets.py --config configs/romania.yaml
    python src/train_multi_f_array.py --config configs/mimic_f329.yaml --task-id 0
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "mimic_f329.yaml"


def _expand_path(value: str | None, base: Path | None = None) -> str | None:
    if value is None or value == "":
        return None
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(expanded)
    if not path.is_absolute() and base is not None:
        path = (base / path).resolve()
    return str(path)


@dataclass
class DataConfig:
    # build: run in-repo build_datasets.py (manifest + npy)
    # prebuilt: read pickles from data.prebuilt_root (user-built dataset)
    mode: str = "build"
    prebuilt_root: str | None = None


@dataclass
class ECGConfig:
    n_leads: int = 12
    signal_len: int = 5000
    source_len: int | None = None
    length_policy: str = "center_crop"  # center_crop | truncate_start | pad


@dataclass
class DemographicsColumns:
    ecg_id_column: str = "ECG ID"
    sex_column: str = "sex"
    diagnosis_column: str = "main psychiatric diagnosis"
    age_column: str | None = None


@dataclass
class ManifestColumns:
    ecg_id_column: str = "ecg_id"
    diagnosis_column: str = "diagnosis"


@dataclass
class InputConfig:
    artifacts_dir: str = ""
    manifest_filename: str = "manifest.csv"
    npy_suffix: str = "_processed.npy"
    demographics_path: str = ""
    manifest_columns: ManifestColumns = field(default_factory=ManifestColumns)
    demographics_columns: DemographicsColumns = field(default_factory=DemographicsColumns)


@dataclass
class NormalPoolConfig:
    pkl: str | None = None
    external_data_root: str | None = None
    external_shared_subdir: str = "shared"
    external_pkl_name: str = "recs_normal.pkl"
    # Build case pickles only; skip normal pool until controls are available.
    cases_only: bool = False
    # Smoke-test / cross-diagnosis mode: use another ICD code as label-0 negatives.
    negative_diagnosis: str | None = None


@dataclass
class DiagnosesConfig:
    codes: list[str] = field(default_factory=list)
    auto_discover: bool = False
    min_count: int = 1


@dataclass
class DemographicMatchingConfig:
    age_tolerance: int = 5
    match_gender: bool = True
    match_age: bool = True
    random_state: int = 42


@dataclass
class TrainingConfig:
    learning_rates: list[float] = field(
        default_factory=lambda: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    )
    dropout_rates: list[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    )
    architectures: list[str] = field(default_factory=lambda: ["st_cnn", "resnet1d"])
    batch_size: int = 32
    epochs: int = 50
    random_state: int = 42
    threshold: float = 0.5


@dataclass
class XAIConfig:
    n_samples: int = 200
    shap_bg: int = 50
    best_configs: dict[str, dict[str, str]] = field(default_factory=dict)
    default_best: dict[str, str] = field(
        default_factory=lambda: {"arch": "st_cnn", "lr": "1e-4", "dr": "0.1"}
    )


@dataclass
class AnalysisConfig:
    """MOF analysis stage settings (frontier dataset assembly)."""
    task_column: str = "psych_code"  # alias as task_id for non-psychiatry cohorts
    recall_min: float | None = 0.30
    recall_metric: str = "test_recall_mean"


@dataclass
class PipelineConfig:
    name: str = "dataset"
    output_root: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    ecg: ECGConfig = field(default_factory=ECGConfig)
    input: InputConfig = field(default_factory=InputConfig)
    normal_pool: NormalPoolConfig = field(default_factory=NormalPoolConfig)
    diagnoses: DiagnosesConfig = field(default_factory=DiagnosesConfig)
    demographic_matching: DemographicMatchingConfig = field(
        default_factory=DemographicMatchingConfig
    )
    training: TrainingConfig = field(default_factory=TrainingConfig)
    xai: XAIConfig = field(default_factory=XAIConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    objectives_raw: dict[str, Any] = field(default_factory=dict)
    config_path: str = ""

    def needs_dataset_build(self) -> bool:
        return self.data.mode == "build"

    def dataset_subdir(self, psych_code: str) -> str:
        return f"icd10_{psych_code}_vs_normal"

    def negative_class_label(self) -> str:
        if self.normal_pool.negative_diagnosis:
            return self.normal_pool.negative_diagnosis
        return "Normal"

    def manifest_path(self) -> str:
        return str(
            Path(self.input.artifacts_dir) / self.input.manifest_filename
        )

    def sweep_task_count(self) -> int:
        return (
            len(self.diagnoses.codes)
            * len(self.training.architectures)
            * len(self.training.learning_rates)
            * len(self.training.dropout_rates)
        )

    def sweep_last_task_id(self) -> int:
        n = self.sweep_task_count()
        if n == 0:
            raise ValueError("Sweep grid is empty — set diagnoses.codes in config.")
        return n - 1

    def build_sweep_grid(self) -> list[dict[str, Any]]:
        grid = []
        for code, arch, lr, dr in product(
            self.diagnoses.codes,
            self.training.architectures,
            self.training.learning_rates,
            self.training.dropout_rates,
        ):
            grid.append(
                {
                    "psych_code": code,
                    "architecture": arch,
                    "learning_rate": lr,
                    "dropout_rate": dr,
                }
            )
        return grid

    def resolve_task(self, task_id: int) -> dict[str, Any]:
        grid = self.build_sweep_grid()
        if task_id < 0 or task_id >= len(grid):
            raise ValueError(
                f"task_id {task_id} out of range [0, {len(grid) - 1}]"
            )
        return {"task_id": task_id, **grid[task_id]}

    def best_xai_config(self, code: str) -> dict[str, str]:
        if code in self.xai.best_configs:
            return dict(self.xai.best_configs[code])
        return dict(self.xai.default_best)

    def to_manifest_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("config_path", None)
        return data

    def save_snapshot(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_manifest_dict(), f, indent=2)


def _nested(raw: dict, key: str, cls: type, defaults: Any = None):
    block = raw.get(key, defaults or {})
    if block is None:
        block = {}
    if cls in (ManifestColumns, DemographicsColumns):
        return cls(**{k: v for k, v in block.items() if k in cls.__dataclass_fields__})
    return cls(**block)


def load_config(path: str | Path | None = None) -> PipelineConfig:
    """Load YAML config; falls back to MOF_CONFIG then configs/romania.yaml."""
    if path is None:
        path = os.environ.get("MOF_CONFIG", str(_DEFAULT_CONFIG))
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    base = config_path.parent
    ecg_raw = raw.get("ecg", {})
    input_raw = dict(raw.get("input", {}))
    manifest_cols = input_raw.pop("manifest_columns", {})
    demo_cols = input_raw.pop("demographics_columns", {})

    output_root = _expand_path(
        raw.get("output_root")
        or os.environ.get("MOF_DATA_ROOT", ""),
        base,
    )
    artifacts_dir = _expand_path(input_raw.get("artifacts_dir", ""), base)
    if not artifacts_dir and output_root:
        artifacts_dir = output_root

    demo_path = _expand_path(input_raw.get("demographics_path", ""), base)
    normal_raw = raw.get("normal_pool", {})
    normal_pkl = _expand_path(
        normal_raw.get("pkl") or os.environ.get("MOF_NORMAL_PKL"), base
    )
    external_root = _expand_path(
        normal_raw.get("external_data_root")
        or os.environ.get("MOF_NORMAL_DATA_ROOT"),
        base,
    )
    data_raw = raw.get("data", {})
    prebuilt_root = _expand_path(
        data_raw.get("prebuilt_root") or os.environ.get("MOF_PREBUILT_ROOT"),
        base,
    )

    cfg = PipelineConfig(
        name=raw.get("name", "dataset"),
        output_root=output_root or "",
        data=DataConfig(
            mode=str(data_raw.get("mode", "build")),
            prebuilt_root=prebuilt_root,
        ),
        ecg=_nested(raw, "ecg", ECGConfig),
        input=InputConfig(
            artifacts_dir=artifacts_dir or "",
            manifest_filename=input_raw.get("manifest_filename", "manifest.csv"),
            npy_suffix=input_raw.get("npy_suffix", "_processed.npy"),
            demographics_path=demo_path or "",
            manifest_columns=_nested(
                {"manifest_columns": manifest_cols}, "manifest_columns", ManifestColumns
            ),
            demographics_columns=_nested(
                {"demographics_columns": demo_cols},
                "demographics_columns",
                DemographicsColumns,
            ),
        ),
        normal_pool=NormalPoolConfig(
            pkl=normal_pkl,
            external_data_root=external_root,
            external_shared_subdir=normal_raw.get(
                "external_shared_subdir", "shared"
            ),
            external_pkl_name=normal_raw.get(
                "external_pkl_name", "recs_normal.pkl"
            ),
            cases_only=bool(normal_raw.get("cases_only", False)),
            negative_diagnosis=normal_raw.get("negative_diagnosis") or None,
        ),
        diagnoses=_nested(raw, "diagnoses", DiagnosesConfig),
        demographic_matching=_nested(
            raw, "demographic_matching", DemographicMatchingConfig
        ),
        training=_nested(raw, "training", TrainingConfig),
        xai=_nested(raw, "xai", XAIConfig),
        analysis=_nested(raw, "analysis", AnalysisConfig),
        objectives_raw=dict(raw.get("objectives", {})),
        config_path=str(config_path),
    )

    if cfg.data.mode not in {"build", "prebuilt"}:
        raise ValueError(f"data.mode must be 'build' or 'prebuilt', got {cfg.data.mode!r}")
    if cfg.data.mode == "prebuilt" and not cfg.data.prebuilt_root:
        raise ValueError(
            "data.mode is 'prebuilt' but data.prebuilt_root / MOF_PREBUILT_ROOT is not set."
        )

    if not cfg.output_root:
        raise ValueError(
            "output_root must be set in config or via MOF_DATA_ROOT environment variable."
        )
    if not cfg.input.artifacts_dir:
        cfg.input.artifacts_dir = cfg.output_root

    return cfg


def load_config_or_snapshot(
    config_path: str | Path | None = None,
    data_root: str | Path | None = None,
) -> PipelineConfig:
    """
    Prefer datasets/pipeline_config.json under data_root (written at build time),
    otherwise load YAML config.
    """
    if data_root:
        snapshot = Path(data_root) / "datasets" / "pipeline_config.json"
        if snapshot.is_file():
            with open(snapshot, encoding="utf-8") as f:
                raw = json.load(f)
            return _config_from_dict(raw, config_path=str(snapshot))
    return load_config(config_path)


def _config_from_dict(raw: dict[str, Any], config_path: str = "") -> PipelineConfig:
    input_raw = dict(raw.get("input", {}))
    manifest_cols = input_raw.pop("manifest_columns", {})
    demo_cols = input_raw.pop("demographics_columns", {})
    normal_raw = raw.get("normal_pool", {})

    return PipelineConfig(
        name=raw.get("name", "dataset"),
        output_root=raw.get("output_root", ""),
        ecg=ECGConfig(**raw.get("ecg", {})),
        input=InputConfig(
            manifest_columns=ManifestColumns(**manifest_cols),
            demographics_columns=DemographicsColumns(**demo_cols),
            **{k: v for k, v in input_raw.items() if k in InputConfig.__dataclass_fields__},
        ),
        normal_pool=NormalPoolConfig(**normal_raw),
        diagnoses=DiagnosesConfig(**raw.get("diagnoses", {})),
        demographic_matching=DemographicMatchingConfig(
            **raw.get("demographic_matching", {})
        ),
        training=TrainingConfig(**raw.get("training", {})),
        xai=XAIConfig(**raw.get("xai", {})),
        analysis=AnalysisConfig(**raw.get("analysis", {})),
        data=DataConfig(**raw.get("data", {})),
        objectives_raw=dict(raw.get("objectives", {})),
        config_path=config_path,
    )


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config path (default: MOF_CONFIG or configs/mimic_f329.yaml).",
    )


def resolve_config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return load_config(getattr(args, "config", None))


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "command",
        choices=["print", "sweep-count", "sweep-last-id", "codes"],
        help="Command to run.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.command == "print":
        print(json.dumps(cfg.to_manifest_dict(), indent=2))
    elif args.command == "sweep-count":
        print(cfg.sweep_task_count())
    elif args.command == "sweep-last-id":
        print(cfg.sweep_last_task_id())
    elif args.command == "codes":
        for code in cfg.diagnoses.codes:
            print(code)


if __name__ == "__main__":
    cli()
