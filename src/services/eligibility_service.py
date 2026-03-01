from __future__ import annotations

from typing import Any

from services.rate_service import DEFAULT_POLICY_ID


MIN_ELIGIBLE_AGE = 18
MAX_ELIGIBLE_AGE = 65


def check_eligibility(
    *,
    age: int,
    has_preexisting_condition: bool = False,
    currently_hospitalized: bool = False,
    policy_id: str = DEFAULT_POLICY_ID,
) -> dict[str, Any]:
    if age < 0:
        raise ValueError("age must be non-negative")

    evaluated_rules: list[dict[str, Any]] = []
    reasons: list[str] = []

    age_rule_passed = MIN_ELIGIBLE_AGE <= int(age) <= MAX_ELIGIBLE_AGE
    evaluated_rules.append(
        {
            "rule": "age_range",
            "passed": age_rule_passed,
            "details": {"min": MIN_ELIGIBLE_AGE, "max": MAX_ELIGIBLE_AGE},
        }
    )
    if not age_rule_passed:
        reasons.append(
            f"age must be between {MIN_ELIGIBLE_AGE} and {MAX_ELIGIBLE_AGE} inclusive"
        )

    preexisting_rule_passed = not bool(has_preexisting_condition)
    evaluated_rules.append(
        {
            "rule": "preexisting_condition",
            "passed": preexisting_rule_passed,
            "details": {
                "has_preexisting_condition": bool(has_preexisting_condition),
            },
        }
    )
    if not preexisting_rule_passed:
        reasons.append("pre-existing condition is not eligible for this policy")

    hospitalization_rule_passed = not bool(currently_hospitalized)
    evaluated_rules.append(
        {
            "rule": "currently_hospitalized",
            "passed": hospitalization_rule_passed,
            "details": {
                "currently_hospitalized": bool(currently_hospitalized),
            },
        }
    )
    if not hospitalization_rule_passed:
        reasons.append("currently hospitalized applicants are not eligible")

    eligible = not reasons
    return {
        "policy_id": policy_id,
        "age": int(age),
        "eligible": eligible,
        "reasons": reasons,
        "evaluated_rules": evaluated_rules,
    }

