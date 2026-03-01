from __future__ import annotations

import os
import sys
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from assistant.orchestrator import route_question  # noqa: E402


class AssistantRouterTests(unittest.TestCase):
    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_routes_eligibility_question(self) -> None:
        decision = route_question(
            "Am I eligible if I am 42 years old?",
            model="gpt-4o-mini",
        )
        self.assertEqual(decision.route, "eligibility")
        self.assertEqual(decision.age, 42)

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_routes_quote_question_and_extracts_smoker(self) -> None:
        decision = route_question(
            "Can I get a premium quote? I am age 40 and non-smoker.",
            model="gpt-4o-mini",
        )
        self.assertEqual(decision.route, "quote")
        self.assertEqual(decision.age, 40)
        self.assertFalse(bool(decision.smoker))

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_routes_rate_question(self) -> None:
        decision = route_question(
            "Check the rate table for age 46 smoker with return of premium rider.",
            model="gpt-4o-mini",
        )
        self.assertEqual(decision.route, "rate")
        self.assertEqual(decision.age, 46)
        self.assertTrue(bool(decision.smoker))

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_routes_rag_question(self) -> None:
        decision = route_question(
            "What illnesses are covered by the policy?",
            model="gpt-4o-mini",
        )
        self.assertEqual(decision.route, "rag")


if __name__ == "__main__":
    unittest.main()
