from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from services import eligibility_service  # noqa: E402


class EligibilityServiceTests(unittest.TestCase):
    def test_check_eligibility_passes_for_valid_profile(self) -> None:
        result = eligibility_service.check_eligibility(
            age=35,
            has_preexisting_condition=False,
            currently_hospitalized=False,
        )
        self.assertTrue(result["eligible"])
        self.assertEqual(result["reasons"], [])
        self.assertEqual(len(result["evaluated_rules"]), 3)

    def test_check_eligibility_fails_for_age_outside_range(self) -> None:
        result = eligibility_service.check_eligibility(age=70)
        self.assertFalse(result["eligible"])
        self.assertTrue(any("between" in item for item in result["reasons"]))

    def test_check_eligibility_fails_for_preexisting_condition(self) -> None:
        result = eligibility_service.check_eligibility(
            age=35,
            has_preexisting_condition=True,
        )
        self.assertFalse(result["eligible"])
        self.assertTrue(
            any("pre-existing condition" in item for item in result["reasons"])
        )

    def test_check_eligibility_fails_for_hospitalized(self) -> None:
        result = eligibility_service.check_eligibility(
            age=35,
            currently_hospitalized=True,
        )
        self.assertFalse(result["eligible"])
        self.assertTrue(any("hospitalized" in item for item in result["reasons"]))

    def test_check_eligibility_rejects_negative_age(self) -> None:
        with self.assertRaises(ValueError):
            eligibility_service.check_eligibility(age=-1)


if __name__ == "__main__":
    unittest.main()

