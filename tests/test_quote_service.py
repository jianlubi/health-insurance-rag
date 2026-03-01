from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from services import quote_service  # noqa: E402


class QuoteServiceTests(unittest.TestCase):
    @patch("services.quote_service.get_rate_quote")
    def test_generate_quote_rejects_when_ineligible(self, mock_get_rate_quote) -> None:
        result = quote_service.generate_quote(
            age=70,
            smoker=False,
            riders=[],
            has_preexisting_condition=False,
            currently_hospitalized=False,
        )

        self.assertEqual(result["status"], "rejected")
        self.assertIsNone(result["rate_quote"])
        self.assertFalse(result["eligibility"]["eligible"])
        mock_get_rate_quote.assert_not_called()

    @patch("services.quote_service.get_rate_quote")
    def test_generate_quote_calls_rate_service_after_eligibility_pass(
        self, mock_get_rate_quote
    ) -> None:
        mock_get_rate_quote.return_value = {
            "policy_id": "demolife_critical_illness_policy",
            "age": 42,
            "smoker": False,
            "age_band": {"min": 31, "max": 45},
            "benefit_amount": 100000.0,
            "base_monthly_rate": 42.0,
            "scaled_base_monthly_rate": 42.0,
            "applied_riders": [],
            "unknown_riders": [],
            "total_loading_pct": 0.0,
            "monthly_premium": 42.0,
            "currency": "USD",
            "assumptions": {
                "base_benefit_amount": 100000.0,
                "benefit_scaling": "linear",
            },
        }

        result = quote_service.generate_quote(
            age=42,
            smoker=False,
            riders=["early_stage_cancer"],
            has_preexisting_condition=False,
            currently_hospitalized=False,
        )

        self.assertEqual(result["status"], "quoted")
        self.assertTrue(result["eligibility"]["eligible"])
        self.assertIsNotNone(result["rate_quote"])
        mock_get_rate_quote.assert_called_once()


if __name__ == "__main__":
    unittest.main()

