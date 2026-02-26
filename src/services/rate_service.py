from __future__ import annotations

import os
import re
from typing import Any

import psycopg2
from psycopg2.extras import execute_batch


DEFAULT_POLICY_ID = "demolife_critical_illness_policy"
BASE_BENEFIT_AMOUNT = 100000.0


_RIDER_ALIASES: dict[str, str] = {
    "early stage cancer": "early_stage_cancer",
    "early-stage cancer": "early_stage_cancer",
    "early_stage_cancer": "early_stage_cancer",
    "early stage cancer rider": "early_stage_cancer",
    "early-stage cancer rider": "early_stage_cancer",
    "return of premium": "return_of_premium",
    "return-of-premium": "return_of_premium",
    "return_of_premium": "return_of_premium",
    "return of premium rider": "return_of_premium",
    "return-of-premium rider": "return_of_premium",
}


def _require_database_url(database_url: str | None = None) -> str:
    resolved = database_url or os.getenv("DATABASE_URL")
    if not resolved:
        raise ValueError("DATABASE_URL is required")
    return resolved


def normalize_rider_code(rider: str) -> str:
    normalized = re.sub(r"\s+", " ", str(rider).strip().lower())
    if normalized in _RIDER_ALIASES:
        return _RIDER_ALIASES[normalized]
    normalized = normalized.replace("-", "_").replace(" ", "_")
    normalized = normalized.replace("_rider", "")
    return normalized


def normalize_riders(riders: list[str] | None) -> list[str]:
    if not riders:
        return []
    deduped: list[str] = []
    seen: set[str] = set()
    for rider in riders:
        code = normalize_rider_code(rider)
        if not code or code in seen:
            continue
        deduped.append(code)
        seen.add(code)
    return deduped


def coerce_smoker(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"smoker", "true", "yes", "y", "1"}:
            return True
        if token in {"non-smoker", "non smoker", "nonsmoker", "false", "no", "n", "0"}:
            return False
    raise ValueError("smoker must be a boolean or smoker/non-smoker string")


def ensure_rate_schema(*, database_url: str | None = None) -> None:
    resolved_database_url = _require_database_url(database_url)
    with psycopg2.connect(resolved_database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS policy_rate_bands (
                    policy_id TEXT NOT NULL,
                    age_min INTEGER NOT NULL,
                    age_max INTEGER NOT NULL,
                    smoker BOOLEAN NOT NULL,
                    base_monthly_rate NUMERIC(10,2) NOT NULL CHECK (base_monthly_rate >= 0),
                    PRIMARY KEY (policy_id, age_min, age_max, smoker),
                    CHECK (age_min <= age_max)
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS policy_rate_rider_loadings (
                    policy_id TEXT NOT NULL,
                    rider_code TEXT NOT NULL,
                    loading_pct NUMERIC(8,6) NOT NULL CHECK (loading_pct >= 0),
                    PRIMARY KEY (policy_id, rider_code)
                );
                """
            )
        conn.commit()


def seed_default_rates(
    *,
    database_url: str | None = None,
    policy_id: str = DEFAULT_POLICY_ID,
) -> None:
    ensure_rate_schema(database_url=database_url)
    resolved_database_url = _require_database_url(database_url)
    with psycopg2.connect(resolved_database_url) as conn:
        with conn.cursor() as cur:
            band_rows = [
                (policy_id, 18, 30, False, 28.0),
                (policy_id, 18, 30, True, 40.0),
                (policy_id, 31, 45, False, 42.0),
                (policy_id, 31, 45, True, 60.0),
                (policy_id, 46, 60, False, 70.0),
                (policy_id, 46, 60, True, 95.0),
                (policy_id, 61, 65, False, 110.0),
                (policy_id, 61, 65, True, 150.0),
            ]
            execute_batch(
                cur,
                """
                INSERT INTO policy_rate_bands (
                    policy_id,
                    age_min,
                    age_max,
                    smoker,
                    base_monthly_rate
                ) VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (policy_id, age_min, age_max, smoker)
                DO UPDATE SET
                    base_monthly_rate = EXCLUDED.base_monthly_rate;
                """,
                band_rows,
                page_size=100,
            )

            rider_rows = [
                (policy_id, "early_stage_cancer", 0.15),
                (policy_id, "return_of_premium", 0.20),
            ]
            execute_batch(
                cur,
                """
                INSERT INTO policy_rate_rider_loadings (
                    policy_id,
                    rider_code,
                    loading_pct
                ) VALUES (%s, %s, %s)
                ON CONFLICT (policy_id, rider_code)
                DO UPDATE SET
                    loading_pct = EXCLUDED.loading_pct;
                """,
                rider_rows,
                page_size=100,
            )
        conn.commit()


def get_rate_quote(
    *,
    age: int,
    smoker: bool,
    riders: list[str] | None = None,
    benefit_amount: float = BASE_BENEFIT_AMOUNT,
    policy_id: str = DEFAULT_POLICY_ID,
    database_url: str | None = None,
) -> dict[str, Any]:
    if age < 0:
        raise ValueError("age must be non-negative")
    if benefit_amount <= 0:
        raise ValueError("benefit_amount must be greater than 0")

    requested_riders = normalize_riders(riders)
    resolved_database_url = _require_database_url(database_url)
    with psycopg2.connect(resolved_database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    age_min,
                    age_max,
                    base_monthly_rate
                FROM policy_rate_bands
                WHERE policy_id = %s
                  AND smoker = %s
                  AND %s BETWEEN age_min AND age_max
                ORDER BY age_min DESC
                LIMIT 1;
                """,
                (policy_id, smoker, age),
            )
            band_row = cur.fetchone()

            if band_row is None:
                cur.execute(
                    """
                    SELECT MIN(age_min), MAX(age_max)
                    FROM policy_rate_bands
                    WHERE policy_id = %s;
                    """,
                    (policy_id,),
                )
                coverage_row = cur.fetchone()
                if coverage_row and coverage_row[0] is not None and coverage_row[1] is not None:
                    raise ValueError(
                        f"no rate found for age {age}; supported range is "
                        f"{int(coverage_row[0])}-{int(coverage_row[1])}"
                    )
                raise ValueError("no rates configured for requested policy")

            age_min = int(band_row[0])
            age_max = int(band_row[1])
            base_monthly_rate = float(band_row[2])

            loading_by_rider: dict[str, float] = {}
            if requested_riders:
                cur.execute(
                    """
                    SELECT rider_code, loading_pct
                    FROM policy_rate_rider_loadings
                    WHERE policy_id = %s
                      AND rider_code = ANY(%s);
                    """,
                    (policy_id, requested_riders),
                )
                for rider_code, loading_pct in cur.fetchall():
                    loading_by_rider[str(rider_code)] = float(loading_pct)

    unknown_riders = [r for r in requested_riders if r not in loading_by_rider]
    applied_riders = [
        {"rider": rider, "loading_pct": loading_by_rider[rider]}
        for rider in requested_riders
        if rider in loading_by_rider
    ]
    total_loading_pct = sum(float(r["loading_pct"]) for r in applied_riders)
    scaled_base_monthly_rate = base_monthly_rate * (float(benefit_amount) / BASE_BENEFIT_AMOUNT)
    monthly_premium = scaled_base_monthly_rate * (1.0 + total_loading_pct)

    return {
        "policy_id": policy_id,
        "age": int(age),
        "smoker": bool(smoker),
        "age_band": {"min": age_min, "max": age_max},
        "benefit_amount": float(benefit_amount),
        "base_monthly_rate": round(base_monthly_rate, 2),
        "scaled_base_monthly_rate": round(scaled_base_monthly_rate, 2),
        "applied_riders": applied_riders,
        "unknown_riders": unknown_riders,
        "total_loading_pct": round(total_loading_pct, 4),
        "monthly_premium": round(monthly_premium, 2),
        "currency": "USD",
        "assumptions": {
            "base_benefit_amount": BASE_BENEFIT_AMOUNT,
            "benefit_scaling": "linear",
        },
    }
