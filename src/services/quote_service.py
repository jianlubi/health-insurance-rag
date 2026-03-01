from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlencode

from services.eligibility_service import check_eligibility
from services.rate_service import (
    BASE_BENEFIT_AMOUNT,
    DEFAULT_POLICY_ID,
    get_rate_quote,
)


def _build_application_url(
    *,
    age: int,
    smoker: bool,
    riders: list[str],
    benefit_amount: float,
    policy_id: str,
    has_preexisting_condition: bool,
    currently_hospitalized: bool,
) -> str:
    base_url = os.getenv("APPLICATION_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    query = urlencode(
        {
            "age": age,
            "smoker": str(bool(smoker)).lower(),
            "riders": ",".join(riders),
            "benefit_amount": benefit_amount,
            "policy_id": policy_id,
            "has_preexisting_condition": str(bool(has_preexisting_condition)).lower(),
            "currently_hospitalized": str(bool(currently_hospitalized)).lower(),
        }
    )
    return f"{base_url}/application/complete?{query}"


def generate_quote(
    *,
    age: int,
    smoker: bool,
    riders: list[str] | None = None,
    benefit_amount: float = BASE_BENEFIT_AMOUNT,
    policy_id: str = DEFAULT_POLICY_ID,
    has_preexisting_condition: bool = False,
    currently_hospitalized: bool = False,
) -> dict[str, Any]:
    rider_list = riders or []
    eligibility = check_eligibility(
        age=age,
        has_preexisting_condition=has_preexisting_condition,
        currently_hospitalized=currently_hospitalized,
        policy_id=policy_id,
    )

    if not eligibility["eligible"]:
        return {
            "status": "rejected",
            "message": "Eligibility check failed; quote was not generated.",
            "eligibility": eligibility,
            "rate_quote": None,
            "application_url": None,
        }

    rate_quote = get_rate_quote(
        age=age,
        smoker=smoker,
        riders=rider_list,
        benefit_amount=benefit_amount,
        policy_id=policy_id,
    )
    application_url = _build_application_url(
        age=age,
        smoker=smoker,
        riders=rider_list,
        benefit_amount=benefit_amount,
        policy_id=policy_id,
        has_preexisting_condition=has_preexisting_condition,
        currently_hospitalized=currently_hospitalized,
    )
    return {
        "status": "quoted",
        "message": "Quote generated successfully.",
        "eligibility": eligibility,
        "rate_quote": rate_quote,
        "application_url": application_url,
    }
