from __future__ import annotations

from typing import Any

from services.eligibility_service import check_eligibility
from services.rate_service import (
    BASE_BENEFIT_AMOUNT,
    DEFAULT_POLICY_ID,
    get_rate_quote,
)


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
        }

    rate_quote = get_rate_quote(
        age=age,
        smoker=smoker,
        riders=riders or [],
        benefit_amount=benefit_amount,
        policy_id=policy_id,
    )
    return {
        "status": "quoted",
        "message": "Quote generated successfully.",
        "eligibility": eligibility,
        "rate_quote": rate_quote,
    }

