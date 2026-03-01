from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from assistant.orchestrator import (  # noqa: E402
    RouteDecision,
    clear_session_profile,
    remember_session_profile,
    run_insurance_assistant,
)


class AssistantMemoryTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_session_profile("test-session")
        clear_session_profile("test-session-2")
        clear_session_profile("test-session-3")

    @patch("assistant.orchestrator.generate_quote")
    @patch(
        "assistant.orchestrator.route_question",
        return_value=RouteDecision(route="quote", smoker=False),
    )
    def test_quote_uses_age_from_session_memory(
        self,
        _mock_route_question,
        mock_generate_quote,
    ) -> None:
        remember_session_profile("test-session", {"age": 42})
        mock_generate_quote.return_value = {
            "status": "quoted",
            "message": "Quote generated successfully.",
            "eligibility": {
                "policy_id": "demolife_critical_illness_policy",
                "age": 42,
                "eligible": True,
                "reasons": [],
                "evaluated_rules": [],
            },
            "rate_quote": {"monthly_premium": 42.0},
        }

        result = run_insurance_assistant(
            question="How much is the basic plan?",
            model="gpt-4o-mini",
            session_id="test-session",
            include_chunks=False,
        )

        self.assertIn("Quote generated successfully", result["answer"])
        self.assertEqual(mock_generate_quote.call_count, 1)
        kwargs = mock_generate_quote.call_args.kwargs
        self.assertEqual(kwargs["age"], 42)
        self.assertFalse(kwargs["smoker"])

    @patch(
        "assistant.orchestrator.route_question",
        return_value=RouteDecision(route="quote"),
    )
    def test_quote_missing_fields_message_is_natural(self, _mock_route_question) -> None:
        result = run_insurance_assistant(
            question="How much is the basic plan?",
            model="gpt-4o-mini",
            session_id="test-session-2",
            include_chunks=False,
        )
        self.assertIn("I can generate that quote", result["answer"])
        self.assertIn("whether you are a smoker", result["answer"])
        self.assertIn("I'll run eligibility", result["answer"])

    @patch("assistant.orchestrator.generate_quote")
    @patch(
        "assistant.orchestrator.route_question",
        side_effect=[
            RouteDecision(route="quote", age=42),
            RouteDecision(route="rag", smoker=False),
        ],
    )
    def test_profile_followup_continues_pending_quote_route(
        self,
        _mock_route_question,
        mock_generate_quote,
    ) -> None:
        mock_generate_quote.return_value = {
            "status": "quoted",
            "message": "Quote generated successfully.",
            "eligibility": {
                "policy_id": "demolife_critical_illness_policy",
                "age": 42,
                "eligible": True,
                "reasons": [],
                "evaluated_rules": [],
            },
            "rate_quote": {"monthly_premium": 42.0},
        }

        first = run_insurance_assistant(
            question="How much is the basic plan?",
            model="gpt-4o-mini",
            session_id="test-session-3",
            include_chunks=False,
        )
        self.assertIn("I can generate that quote", first["answer"])
        self.assertEqual(first["route"], "quote")

        second = run_insurance_assistant(
            question="I am a non-smoker.",
            model="gpt-4o-mini",
            session_id="test-session-3",
            include_chunks=False,
        )
        self.assertIn("Quote generated successfully", second["answer"])
        self.assertEqual(second["route"], "quote")
        self.assertEqual(mock_generate_quote.call_count, 1)
        kwargs = mock_generate_quote.call_args.kwargs
        self.assertEqual(kwargs["age"], 42)
        self.assertFalse(kwargs["smoker"])


if __name__ == "__main__":
    unittest.main()
