import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline_config import load_config  # noqa: E402


class TestPipelineConfig(unittest.TestCase):
    def test_mimic_config_loads(self):
        os.environ["MOF_PREBUILT_ROOT"] = "/path/to/prebuilt_mimic_run"
        os.environ["MOF_DATA_ROOT"] = "/tmp/mof_test_run"
        cfg_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "mimic_f329.yaml"
        )
        cfg = load_config(cfg_path)
        self.assertEqual(cfg.data.mode, "prebuilt")
        self.assertEqual(cfg.diagnoses.codes, ["F329"])
        self.assertFalse(cfg.needs_dataset_build())

    def test_romania_config_loads(self):
        cfg_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "romania.yaml"
        )
        cfg = load_config(cfg_path)
        self.assertEqual(cfg.data.mode, "build")
        self.assertTrue(cfg.needs_dataset_build())
        self.assertEqual(cfg.ecg.n_leads, 12)
        self.assertGreaterEqual(len(cfg.diagnoses.codes), 1)
        self.assertEqual(
            cfg.sweep_task_count(),
            len(cfg.diagnoses.codes)
            * len(cfg.training.architectures)
            * len(cfg.training.learning_rates)
            * len(cfg.training.dropout_rates),
        )

    def test_template_has_required_sections(self):
        cfg_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "dataset.template.yaml"
        )
        cfg = load_config(cfg_path)
        self.assertEqual(cfg.data.mode, "build")
        self.assertEqual(cfg.ecg.n_leads, 12)
        self.assertFalse(cfg.diagnoses.auto_discover)


if __name__ == "__main__":
    unittest.main()
