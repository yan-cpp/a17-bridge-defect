from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a17.config import load_config, validate_config  # noqa: E402


class ConfigTests(unittest.TestCase):
    def test_configuration_is_valid_and_phase_one_has_three_classes(self) -> None:
        bundle = load_config(ROOT)
        warnings = validate_config(bundle)
        self.assertEqual(bundle.class_map["phase1_classes"], ["crack", "spalling", "exposed_rebar"])
        self.assertEqual(bundle.metrics["phase1_classes"], bundle.class_map["phase1_classes"])
        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
