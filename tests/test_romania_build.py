"""Tests for Romania build-mode dataset integration."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline_config import load_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ROMANIA_CFG = REPO / "configs" / "romania.yaml"


class TestRomaniaBuild(unittest.TestCase):
    def test_romania_config_loads(self):
        cfg = load_config(ROMANIA_CFG)
        self.assertEqual(cfg.data.mode, "build")
        self.assertEqual(cfg.ecg.n_leads, 12)
        self.assertGreaterEqual(len(cfg.diagnoses.codes), 1)

    def test_romania_requires_build(self):
        cfg = load_config(ROMANIA_CFG)
        self.assertTrue(cfg.needs_dataset_build())

    def test_romania_manifest_when_available(self):
        cfg = load_config(ROMANIA_CFG)
        manifest = Path(cfg.manifest_path())
        if not manifest.is_file():
            self.skipTest(f"Romania manifest not available: {manifest}")
        self.assertGreater(manifest.stat().st_size, 0)

    def test_romania_sweep_grid(self):
        cfg = load_config(ROMANIA_CFG)
        self.assertEqual(
            cfg.sweep_task_count(),
            len(cfg.diagnoses.codes)
            * len(cfg.training.architectures)
            * len(cfg.training.learning_rates)
            * len(cfg.training.dropout_rates),
        )


if __name__ == "__main__":
    unittest.main()
