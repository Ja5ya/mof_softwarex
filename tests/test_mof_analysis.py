"""Tests for multi-objective analysis helpers."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mof_analysis import (  # noqa: E402
    compute_fp_fs_scores,
    compute_metrics,
    discover_completed_runs,
    load_bootstrap_scored,
    lr_tag,
    merge_sweep_fe_csvs,
    parse_lr_dr,
    resolve_paths,
)
from pipeline_config import load_config  # noqa: E402


class TestMofAnalysis(unittest.TestCase):
    def test_lr_tag_matches_training(self):
        self.assertEqual(lr_tag(1e-4), "lr1e-4")
        self.assertEqual(lr_tag(5e-5), "lr5e-5")

    def test_parse_lr_dr(self):
        self.assertEqual(parse_lr_dr("lr1e-4_dr0.1"), (1e-4, 0.1))

    def test_compute_metrics_binary(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        mets = compute_metrics(y_true, y_score)
        self.assertEqual(mets["acc"], 1.0)
        self.assertEqual(mets["recall"], 1.0)

    def test_compute_fp_fs_scores_per_code_normalisation(self):
        df = compute_fp_fs_scores(
            pd.DataFrame(
                [
                    {
                        "psych_code": "A",
                        "test_acc_mean": 0.6,
                        "test_precision_mean": 0.6,
                        "test_recall_mean": 0.6,
                        "test_specificity_mean": 0.6,
                        "test_acc_std": 0.1,
                        "test_precision_std": 0.1,
                        "test_recall_std": 0.1,
                        "test_specificity_std": 0.1,
                    },
                    {
                        "psych_code": "A",
                        "test_acc_mean": 0.8,
                        "test_precision_mean": 0.8,
                        "test_recall_mean": 0.8,
                        "test_specificity_mean": 0.8,
                        "test_acc_std": 0.05,
                        "test_precision_std": 0.05,
                        "test_recall_std": 0.05,
                        "test_specificity_std": 0.05,
                    },
                ]
            ),
            codes=["A"],
        )
        self.assertAlmostEqual(df["f_P"].max(), 1.0)
        self.assertAlmostEqual(df["f_S"].max(), 1.0)

    def test_romania_config_paths(self):
        cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "romania.yaml")
        paths = resolve_paths(cfg)
        self.assertEqual(paths["sweep_root"].name, "hyper_sweep")
        completed, missing, expected = discover_completed_runs(cfg)
        self.assertEqual(expected, cfg.sweep_task_count())

    def test_merge_sweep_fe_csvs(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "romania.yaml")
            cfg.output_root = tmp
            fe_dir = Path(tmp) / "xai_results" / "hyper_sweep_fe"
            fe_dir.mkdir(parents=True)
            (fe_dir / "sweep_fe_A.csv").write_text(
                "code,sweep_arch,sweep_run,f_E\nA,st_cnn,lr1e-4_dr0.1,0.5\n"
            )
            out = merge_sweep_fe_csvs(cfg)
            self.assertIsNotNone(out)
            self.assertTrue((fe_dir / "sweep_fe_summary.csv").is_file())

    def test_load_bootstrap_scored_from_cache(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "romania.yaml")
            cfg.output_root = tmp
            sweep = Path(tmp) / "hyper_sweep"
            sweep.mkdir(parents=True)
            scored_path = sweep / "bootstrap_results_with_scores.csv"
            scored_path.write_text(
                "psych_code,f_P_raw,f_S_raw,f_P,f_S\n"
                "F32.10,0.5,0.1,0.5,0.5\n"
            )
            df = load_bootstrap_scored(cfg)
            self.assertEqual(len(df), 1)
            self.assertAlmostEqual(df.loc[0, "f_P"], 0.5)


if __name__ == "__main__":
    unittest.main()
