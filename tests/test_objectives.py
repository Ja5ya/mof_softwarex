"""Tests for pluggable objective scoring."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mof.objectives import (  # noqa: E402
    ObjectivesConfig,
    FPConfig,
    FEConfig,
    apply_fe_scores,
    compute_fp_fs_scores,
    objectives_from_dict,
)
from mof_analysis import load_three_objective_dataset  # noqa: E402
from pipeline_config import load_config  # noqa: E402


def _bootstrap_row(**overrides):
    base = {
        "psych_code": "A",
        "architecture": "st_cnn",
        "learning_rate": 1e-4,
        "dropout_rate": 0.1,
        "test_acc_mean": 0.6,
        "test_precision_mean": 0.6,
        "test_recall_mean": 0.6,
        "test_specificity_mean": 0.6,
        "test_acc_std": 0.1,
        "test_precision_std": 0.1,
        "test_recall_std": 0.1,
        "test_specificity_std": 0.1,
    }
    base.update(overrides)
    return base


class TestObjectives(unittest.TestCase):
    def test_custom_fp_weights_change_f_p_raw(self):
        df = pd.DataFrame(
            [
                _bootstrap_row(test_recall_mean=1.0, test_specificity_mean=0.0),
                _bootstrap_row(test_recall_mean=0.0, test_specificity_mean=1.0),
            ]
        )
        default_cfg = ObjectivesConfig(
            f_P=FPConfig(
                metrics=["test_recall_mean", "test_specificity_mean"],
                normalize="none",
            )
        )
        weighted_cfg = ObjectivesConfig(
            f_P=FPConfig(
                metrics=["test_recall_mean", "test_specificity_mean"],
                weights=[1.0, 0.0],
                normalize="none",
            )
        )
        default_scores = compute_fp_fs_scores(df, codes=["A"], cfg=default_cfg)
        weighted_scores = compute_fp_fs_scores(df, codes=["A"], cfg=weighted_cfg)

        self.assertAlmostEqual(default_scores.loc[0, "f_P_raw"], 0.5)
        self.assertAlmostEqual(weighted_scores.loc[0, "f_P_raw"], 1.0)
        self.assertAlmostEqual(weighted_scores.loc[1, "f_P_raw"], 0.0)

    def test_apply_fe_scores_custom_weights(self):
        df = pd.DataFrame(
            [
                {
                    "continuity_valid": 1.0,
                    "compactness_valid": 0.0,
                    "contrastivity_valid": 0.0,
                }
            ]
        )
        default_out = apply_fe_scores(df, ObjectivesConfig())
        custom_out = apply_fe_scores(
            df,
            ObjectivesConfig(
                f_E=FEConfig(
                    components=[
                        "continuity_valid",
                        "compactness_valid",
                        "contrastivity_valid",
                    ],
                    weights=[1.0, 0.0, 0.0],
                )
            ),
        )
        self.assertAlmostEqual(default_out.loc[0, "f_E"], 1.0 / 3.0)
        self.assertAlmostEqual(custom_out.loc[0, "f_E"], 1.0)

    def test_objectives_from_dict_reads_fe_weights(self):
        cfg = objectives_from_dict(
            {
                "f_E": {
                    "components": ["continuity_valid", "compactness_valid"],
                    "weights": [0.75, 0.25],
                }
            }
        )
        self.assertEqual(cfg.f_E.weights, [0.75, 0.25])

    def test_load_three_objective_dataset_applies_yaml_objectives(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(
                Path(__file__).resolve().parents[1] / "configs" / "romania.yaml"
            )
            cfg.output_root = tmp
            cfg.diagnoses.codes = ["A"]
            cfg.objectives_raw = {
                "f_P": {
                    "metrics": ["test_recall_mean", "test_specificity_mean"],
                    "weights": [1.0, 0.0],
                    "normalize": "none",
                },
                "f_S": {"method": "inv_std"},
                "f_E": {
                    "components": [
                        "continuity_valid",
                        "compactness_valid",
                        "contrastivity_valid",
                    ],
                    "weights": [1.0, 0.0, 0.0],
                },
            }
            cfg.analysis.recall_min = None

            sweep = Path(tmp) / "hyper_sweep"
            sweep.mkdir(parents=True)
            pd.DataFrame(
                [
                    _bootstrap_row(test_recall_mean=0.8, test_specificity_mean=0.2),
                    _bootstrap_row(
                        test_recall_mean=0.2,
                        test_specificity_mean=0.8,
                        learning_rate=5e-4,
                    ),
                ]
            ).to_csv(sweep / "bootstrap_results.csv", index=False)

            fe_dir = Path(tmp) / "xai_results" / "hyper_sweep_fe"
            fe_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "code": "A",
                        "sweep_arch": "st_cnn",
                        "sweep_run": "lr1e-4_dr0.1",
                        "continuity_valid": 0.9,
                        "compactness_valid": 0.1,
                        "contrastivity_valid": 0.1,
                        "f_E": 0.37,
                    },
                    {
                        "code": "A",
                        "sweep_arch": "st_cnn",
                        "sweep_run": "lr5e-4_dr0.1",
                        "continuity_valid": 0.1,
                        "compactness_valid": 0.9,
                        "contrastivity_valid": 0.9,
                        "f_E": 0.63,
                    },
                ]
            ).to_csv(fe_dir / "sweep_fe_A.csv", index=False)

            df = load_three_objective_dataset(cfg)

            self.assertEqual(len(df), 2)
            self.assertAlmostEqual(df.loc[df["test_recall_mean"] == 0.8, "f_P"].iloc[0], 0.8)
            self.assertAlmostEqual(df.loc[df["test_recall_mean"] == 0.2, "f_P"].iloc[0], 0.2)
            self.assertAlmostEqual(
                df.loc[df["learning_rate"] == 1e-4, "f_E"].iloc[0], 0.9
            )
            self.assertAlmostEqual(
                df.loc[df["learning_rate"] == 5e-4, "f_E"].iloc[0], 0.1
            )
            self.assertTrue((Path(tmp) / "analysis" / "three_objective_dataset.csv").is_file())


if __name__ == "__main__":
    unittest.main()
