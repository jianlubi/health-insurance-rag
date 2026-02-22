from __future__ import annotations

import re


_PERSONAL_MARKER = re.compile(r"\b(i|my|me|we|our|us)\b", re.IGNORECASE)

_STRONG_PATTERNS: list[tuple[str, str]] = [
    (r"\bam i covered for this\b", "coverage_unspecified"),
    (r"\bhow much would i get\b", "benefit_unspecified"),
    (r"\bcan i get back on my plan\b", "reinstatement"),
]

_CLAIM_DENIAL_PATTERN = re.compile(
    r"\b(denied|denial|appeal|appeals|appealing)\b", re.IGNORECASE
)
_REINSTATEMENT_PATTERN = re.compile(
    r"\b(reinstate|reinstated|reinstatement|lapsed|get back on)\b", re.IGNORECASE
)
_CLAIM_FOLLOW_UP_PATTERN = re.compile(
    r"\b(what else do you need from me|move (this )?claim faster|what now|what can i do now)\b",
    re.IGNORECASE,
)


def classify_ambiguity_intent(question: str) -> str | None:
    q = question.strip().lower()
    if not q:
        return None

    for pattern, intent in _STRONG_PATTERNS:
        if re.search(pattern, q):
            return intent

    has_personal_context = bool(_PERSONAL_MARKER.search(q))
    if has_personal_context and _CLAIM_DENIAL_PATTERN.search(q):
        return "claim_denial"
    if has_personal_context and _REINSTATEMENT_PATTERN.search(q):
        return "reinstatement"
    if has_personal_context and _CLAIM_FOLLOW_UP_PATTERN.search(q):
        return "claim_follow_up"
    return None


def needs_clarification(question: str) -> bool:
    return classify_ambiguity_intent(question) is not None


def build_clarification_prompt(question: str) -> str:
    intent = classify_ambiguity_intent(question) or "general"

    if intent == "claim_denial":
        return (
            "I can help, but I need 2 details first:\n"
            "1. What denial reason was given: missing documents, ineligibility, "
            "waiting-period/exclusion, or suspected fraud?\n"
            "2. What date did you receive the denial letter or email?"
        )

    if intent == "reinstatement":
        return (
            "I can help, but I need 2 details first:\n"
            "1. Why did the policy end or lapse: missed premium, age limit, full payout, "
            "or another reason?\n"
            "2. What date did it lapse or terminate?"
        )

    if intent == "claim_follow_up":
        return (
            "I can help, but I need 2 details first:\n"
            "1. What is the current claim status: submitted, pending documents, under review, "
            "or denied?\n"
            "2. What was the last update date from the insurer?"
        )

    return (
        "I can help, but I need 2 details first:\n"
        "1. What specific issue are you asking about?\n"
        "2. What date did this issue occur?"
    )


def asked_clarifying_question(answer: str) -> bool:
    low = answer.lower()
    asks_for_details = any(
        marker in low
        for marker in ("i need", "please share", "before i can", "which")
    )
    has_two_slots = bool(re.search(r"\b1[.)]\s", answer)) and bool(
        re.search(r"\b2[.)]\s", answer)
    )
    has_question_mark = "?" in answer
    return asks_for_details and (has_two_slots or has_question_mark)
