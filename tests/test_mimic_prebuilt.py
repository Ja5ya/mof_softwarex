"""Tests for MIMIC prebuilt dataset integration."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mof.contracts import validate_dataset_layout  # noqa: E402
from mof.paths import data_source_root, dataset_dir, shared_dir  # noqa: E402
from pipeline_config import load_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MIMIC_CFG = REPO / "configs" / "mimic_f329.yaml"
DEFAULT_PREBUILT = os.environ.get(
    "MOF_PREBUILT_ROOT",
    "/path/to/prebuilt_mimic_run",
)


class TestMimicPrebuilt(unittest.TestCase):
    def test_mimic_all_config_loads(self):
        import os

        os.environ["MOF_PREBUILT_ROOT"] = DEFAULT_PREBUILT
        os.environ["MOF_DATA_ROOT"] = "/tmp/mof_test_run"
        cfg = load_config(REPO / "configs" / "mimic.yaml")
        self.assertEqual(cfg.data.mode, "prebuilt")
        self.assertEqual(len(cfg.diagnoses.codes), 6)
        self.assertEqual(cfg.sweep_task_count(), 6)

    def test_mimic_config_loads(self):
        import os

        os.environ["MOF_PREBUILT_ROOT"] = DEFAULT_PREBUILT
        os.environ["MOF_DATA_ROOT"] = "/tmp/mof_test_run"
        cfg = load_config(MIMIC_CFG)
        self.assertEqual(cfg.data.mode, "prebuilt")
        self.assertEqual(cfg.diagnoses.codes, ["F329"])
        self.assertTrue(cfg.data.prebuilt_root)

    def test_mimic_paths_resolve(self):
        import os

        os.environ["MOF_PREBUILT_ROOT"] = DEFAULT_PREBUILT
        os.environ["MOF_DATA_ROOT"] = "/tmp/mof_test_run"
        if not Path(DEFAULT_PREBUILT).is_dir():
            self.skipTest("MOF_PREBUILT_ROOT not available on this machine")
        cfg = load_config(MIMIC_CFG)
        root = data_source_root(cfg)
        self.assertTrue(root.is_dir(), f"prebuilt_root missing: {root}")
        ds = dataset_dir(cfg, "F329")
        self.assertTrue((ds / "recs_psych.pkl").is_file())
        self.assertTrue(shared_dir(cfg).joinpath("recs_normal.pkl").is_file())

    def test_mimic_pickle_contract(self):
        import os

        os.environ["MOF_PREBUILT_ROOT"] = DEFAULT_PREBUILT
        cfg = load_config(MIMIC_CFG)
        if not Path(cfg.data.prebuilt_root or DEFAULT_PREBUILT).is_dir():
            self.skipTest("MIMIC prebuilt root not available on this machine")
        summary = validate_dataset_layout(
            dataset_dir(cfg, "F329"),
            n_leads=cfg.ecg.n_leads,
        )
        self.assertGreater(summary["n_positive"], 100)
        self.assertEqual(summary["signal_shape"][0], cfg.ecg.n_leads)
        self.assertEqual(summary["signal_shape"][1], cfg.ecg.signal_len)

    def test_prebuilt_skips_build(self):
        import os

        os.environ["MOF_PREBUILT_ROOT"] = DEFAULT_PREBUILT
        cfg = load_config(MIMIC_CFG)
        self.assertFalse(cfg.needs_dataset_build())


if __name__ == "__main__":
    unittest.main()
