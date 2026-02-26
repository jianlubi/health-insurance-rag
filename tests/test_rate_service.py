from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from services import rate_service  # noqa: E402


class FakeCursor:
    def __init__(
        self,
        *,
        fetchone_values: list[tuple | None] | None = None,
        fetchall_values: list[list[tuple]] | None = None,
    ) -> None:
        self._fetchone_values = list(fetchone_values or [])
        self._fetchall_values = list(fetchall_values or [])
        self.executed: list[tuple[str, tuple | None]] = []

    def execute(self, query: str, params: tuple | None = None) -> None:
        self.executed.append((query, params))

    def fetchone(self):
        if not self._fetchone_values:
            return None
        return self._fetchone_values.pop(0)

    def fetchall(self):
        if not self._fetchall_values:
            return []
        return self._fetchall_values.pop(0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor
        self.committed = False

    def cursor(self) -> FakeCursor:
        return self._cursor

    def commit(self) -> None:
        self.committed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class RateServiceTests(unittest.TestCase):
    def test_normalize_riders_dedupes_and_maps_aliases(self) -> None:
        normalized = rate_service.normalize_riders(
            [
                "Early-Stage Cancer Rider",
                "return of premium",
                "early_stage_cancer",
                "Return-Of-Premium Rider",
            ]
        )
        self.assertEqual(normalized, ["early_stage_cancer", "return_of_premium"])

    def test_coerce_smoker_accepts_supported_tokens(self) -> None:
        self.assertTrue(rate_service.coerce_smoker(True))
        self.assertTrue(rate_service.coerce_smoker("smoker"))
        self.assertFalse(rate_service.coerce_smoker(False))
        self.assertFalse(rate_service.coerce_smoker("non smoker"))

    def test_coerce_smoker_rejects_invalid_value(self) -> None:
        with self.assertRaises(ValueError):
            rate_service.coerce_smoker("maybe")

    @patch("services.rate_service.psycopg2.connect")
    def test_get_rate_quote_calculates_monthly_premium(self, mock_connect) -> None:
        fake_cursor = FakeCursor(
            fetchone_values=[(31, 45, 42.0)],
            fetchall_values=[[("early_stage_cancer", 0.15)]],
        )
        mock_connect.return_value = FakeConnection(fake_cursor)

        quote = rate_service.get_rate_quote(
            age=42,
            smoker=False,
            riders=["Early-Stage Cancer Rider", "mystery rider"],
            database_url="postgresql://fake",
        )

        self.assertEqual(quote["age_band"], {"min": 31, "max": 45})
        self.assertEqual(quote["base_monthly_rate"], 42.0)
        self.assertEqual(quote["total_loading_pct"], 0.15)
        self.assertEqual(quote["monthly_premium"], 48.3)
        self.assertEqual(quote["unknown_riders"], ["mystery"])

    @patch("services.rate_service.psycopg2.connect")
    def test_get_rate_quote_scales_for_benefit_amount(self, mock_connect) -> None:
        fake_cursor = FakeCursor(
            fetchone_values=[(31, 45, 60.0)],
            fetchall_values=[[("return_of_premium", 0.2)]],
        )
        mock_connect.return_value = FakeConnection(fake_cursor)

        quote = rate_service.get_rate_quote(
            age=42,
            smoker=True,
            riders=["return_of_premium"],
            benefit_amount=200000,
            database_url="postgresql://fake",
        )

        self.assertEqual(quote["scaled_base_monthly_rate"], 120.0)
        self.assertEqual(quote["monthly_premium"], 144.0)

    @patch("services.rate_service.psycopg2.connect")
    def test_get_rate_quote_rejects_age_outside_configured_ranges(self, mock_connect) -> None:
        fake_cursor = FakeCursor(fetchone_values=[None, (18, 65)])
        mock_connect.return_value = FakeConnection(fake_cursor)

        with self.assertRaises(ValueError) as ctx:
            rate_service.get_rate_quote(
                age=70,
                smoker=False,
                riders=[],
                database_url="postgresql://fake",
            )

        self.assertIn("supported range is 18-65", str(ctx.exception))

    @patch("services.rate_service.psycopg2.connect")
    def test_get_rate_quote_rejects_when_no_rates_configured(self, mock_connect) -> None:
        fake_cursor = FakeCursor(fetchone_values=[None, (None, None)])
        mock_connect.return_value = FakeConnection(fake_cursor)

        with self.assertRaises(ValueError) as ctx:
            rate_service.get_rate_quote(
                age=42,
                smoker=False,
                riders=[],
                database_url="postgresql://fake",
            )

        self.assertIn("no rates configured", str(ctx.exception))

    def test_get_rate_quote_rejects_invalid_inputs_before_db(self) -> None:
        with self.assertRaises(ValueError):
            rate_service.get_rate_quote(
                age=-1,
                smoker=False,
                riders=[],
                database_url="postgresql://fake",
            )

        with self.assertRaises(ValueError):
            rate_service.get_rate_quote(
                age=42,
                smoker=False,
                riders=[],
                benefit_amount=0,
                database_url="postgresql://fake",
            )

    @patch("services.rate_service.execute_batch")
    @patch("services.rate_service.psycopg2.connect")
    def test_seed_default_rates_upserts_bands_and_riders(
        self,
        mock_connect,
        mock_execute_batch,
    ) -> None:
        fake_cursor = FakeCursor()
        fake_connection = FakeConnection(fake_cursor)
        mock_connect.return_value = fake_connection

        rate_service.seed_default_rates(
            policy_id="test_policy",
            database_url="postgresql://fake",
        )

        self.assertTrue(fake_connection.committed)
        self.assertEqual(mock_execute_batch.call_count, 2)
        first_rows = mock_execute_batch.call_args_list[0][0][2]
        second_rows = mock_execute_batch.call_args_list[1][0][2]
        self.assertEqual(len(first_rows), 8)
        self.assertEqual(len(second_rows), 2)


if __name__ == "__main__":
    unittest.main()
