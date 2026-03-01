from __future__ import annotations

import os
import re
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

from core.openai_client import create_openai_client
from rag.ambiguity import build_clarification_prompt, needs_clarification
from rag.answer import SYSTEM_PROMPT, build_context
from rag.citation import ensure_chunk_citation
from rag.retrieve import retrieve_chunks
from services.eligibility_service import check_eligibility
from services.quote_service import generate_quote
from services.rate_service import (
    BASE_BENEFIT_AMOUNT,
    DEFAULT_POLICY_ID,
    coerce_smoker,
    get_rate_quote,
)

try:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    _LANGCHAIN_AVAILABLE = True
except Exception:  # pragma: no cover - dependency optionality
    ChatPromptTemplate = None  # type: ignore[assignment]
    ChatOpenAI = None  # type: ignore[assignment]
    StrOutputParser = None  # type: ignore[assignment]
    _LANGCHAIN_AVAILABLE = False

try:
    from langgraph.graph import END, START, StateGraph

    _LANGGRAPH_AVAILABLE = True
except Exception:  # pragma: no cover - dependency optionality
    END = "__end__"  # type: ignore[assignment]
    START = "__start__"  # type: ignore[assignment]
    StateGraph = None  # type: ignore[assignment]
    _LANGGRAPH_AVAILABLE = False


_ROUTE_VALUES = ("rag", "eligibility", "quote", "rate")
_PROFILE_AGE_PATTERNS = (
    r"\bage\s*(\d{1,3})\b",
    r"\baged\s*(\d{1,3})\b",
    r"\bi\s*am\s*(\d{1,3})\b",
    r"\bi[' ]?m\s*(\d{1,3})\b",
)
_BENEFIT_PATTERNS = (
    r"\b(?:benefit|coverage|sum insured|insured amount)\D{0,20}(\d{4,9})\b",
    r"\$(\d{4,9})\b",
)
_ROUTER_SYSTEM_PROMPT = (
    "You are an insurance assistant router. "
    "Classify intent into exactly one route: rag, eligibility, quote, or rate. "
    "Route rules: "
    "eligibility for qualification/eligibility checks, "
    "quote for premium quote requests, "
    "rate for direct rate-table lookups, "
    "rag for policy questions answered from documents. "
    "Also extract structured profile fields when present in the user question."
)


class RouteDecision(BaseModel):
    route: Literal["rag", "eligibility", "quote", "rate"]
    age: int | None = Field(default=None, ge=0)
    smoker: bool | str | None = None
    riders: list[str] = Field(default_factory=list)
    benefit_amount: float | None = Field(default=None, gt=0)
    policy_id: str | None = None
    has_preexisting_condition: bool | None = None
    currently_hospitalized: bool | None = None


class AssistantState(TypedDict, total=False):
    question: str
    model: str
    session_id: str | None
    session_profile: dict[str, Any]
    pending_route: str | None
    retrieval_options: dict[str, Any]
    include_chunks: bool
    route: str
    route_decision: dict[str, Any]
    needs_clarification: bool
    answer: str
    chunks: list[dict[str, Any]]
    service_result: dict[str, Any] | None


_SESSION_PROFILE_FIELDS = (
    "age",
    "smoker",
    "riders",
    "benefit_amount",
    "policy_id",
    "has_preexisting_condition",
    "currently_hospitalized",
)
_ASSISTANT_SESSION_PROFILES: dict[str, dict[str, Any]] = {}
_ASSISTANT_PENDING_ROUTES: dict[str, str] = {}


def _normalize_session_id(session_id: str | None) -> str | None:
    token = str(session_id or "").strip()
    if not token:
        return None
    return token


def _merge_profile(
    base: dict[str, Any] | None,
    updates: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(base or {})
    if not updates:
        return merged

    for field in _SESSION_PROFILE_FIELDS:
        if field not in updates:
            continue
        value = updates[field]
        if field == "riders":
            if isinstance(value, list) and value:
                merged["riders"] = [str(item) for item in value]
            continue
        if value is None:
            continue
        merged[field] = value
    return merged


def remember_session_profile(session_id: str, profile: dict[str, Any]) -> None:
    normalized = _normalize_session_id(session_id)
    if not normalized:
        return
    current = _ASSISTANT_SESSION_PROFILES.get(normalized) or {}
    _ASSISTANT_SESSION_PROFILES[normalized] = _merge_profile(current, profile)


def clear_session_profile(session_id: str) -> None:
    normalized = _normalize_session_id(session_id)
    if not normalized:
        return
    _ASSISTANT_SESSION_PROFILES.pop(normalized, None)
    _ASSISTANT_PENDING_ROUTES.pop(normalized, None)


def _keyword_route(question: str) -> str:
    q = question.lower()
    if re.search(r"\b(eligible|eligibility|qualify|qualification)\b", q):
        return "eligibility"
    if re.search(r"\b(rate table|base rate|rate band|rider loading|rate)\b", q):
        return "rate"
    if re.search(r"\b(quote|premium|cost|pricing|monthly payment|price)\b", q):
        return "quote"
    return "rag"


def _extract_profile(question: str) -> dict[str, Any]:
    q = question.strip()
    lowered = q.lower()
    extracted: dict[str, Any] = {}

    age_value: int | None = None
    for pattern in _PROFILE_AGE_PATTERNS:
        match = re.search(pattern, lowered)
        if match:
            age_value = int(match.group(1))
            break
    if age_value is not None:
        extracted["age"] = age_value

    smoker_value: bool | None = None
    if re.search(r"\b(non[- ]?smoker|nonsmoker)\b", lowered):
        smoker_value = False
    elif re.search(r"\bsmoker\b", lowered):
        smoker_value = True
    if smoker_value is not None:
        extracted["smoker"] = smoker_value

    riders: list[str] = []
    if "early-stage cancer" in lowered or "early stage cancer" in lowered:
        riders.append("early_stage_cancer")
    if "return of premium" in lowered or "return-of-premium" in lowered:
        riders.append("return_of_premium")
    if riders:
        extracted["riders"] = riders

    for pattern in _BENEFIT_PATTERNS:
        match = re.search(pattern, lowered)
        if match:
            extracted["benefit_amount"] = float(match.group(1))
            break

    if re.search(r"\b(pre[- ]?existing condition|preexisting condition)\b", lowered):
        extracted["has_preexisting_condition"] = True
    if re.search(r"\b(currently hospitalized|hospitalized now)\b", lowered):
        extracted["currently_hospitalized"] = True

    return extracted


def _build_fallback_decision(question: str) -> RouteDecision:
    extracted = _extract_profile(question)
    return RouteDecision(route=_keyword_route(question), **extracted)


def route_question(question: str, *, model: str) -> RouteDecision:
    fallback = _build_fallback_decision(question)
    if not _LANGCHAIN_AVAILABLE or ChatPromptTemplate is None or ChatOpenAI is None:
        return fallback
    if not os.getenv("OPENAI_API_KEY"):
        return fallback

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _ROUTER_SYSTEM_PROMPT),
            ("human", "Question:\n{question}"),
        ]
    )
    llm = ChatOpenAI(model=model, temperature=0)
    chain = prompt | llm.with_structured_output(RouteDecision)
    try:
        decision = chain.invoke({"question": question})
    except Exception:
        return fallback

    if decision.route not in _ROUTE_VALUES:
        return fallback
    return decision


def _format_missing_fields(route: str, fields: list[str]) -> str:
    cleaned = [str(field).strip() for field in fields if str(field).strip()]
    if not cleaned:
        return "Could you share a bit more detail so I can continue?"

    phrase_by_field = {
        "age": "your age",
        "smoker": "whether you are a smoker",
        "riders": "which riders you want",
        "benefit_amount": "your desired benefit amount",
        "policy_id": "the policy id",
        "has_preexisting_condition": "whether you have a pre-existing condition",
        "currently_hospitalized": "whether you are currently hospitalized",
    }
    phrases = [phrase_by_field.get(item, item.replace("_", " ")) for item in cleaned]
    if len(phrases) == 1:
        joined = phrases[0]
    elif len(phrases) == 2:
        joined = f"{phrases[0]} and {phrases[1]}"
    else:
        joined = ", ".join(phrases[:-1]) + f", and {phrases[-1]}"

    if route == "eligibility":
        return f"To check eligibility, I still need to know {joined}."
    if route == "quote":
        return (
            f"I can generate that quote, but I still need to know {joined}. "
            "After that, I'll run eligibility and then calculate the premium."
        )
    if route == "rate":
        return f"I can look up the rate, but I still need to know {joined}."
    return f"I need a little more information. Please share {joined}."


def _format_eligibility_answer(result: dict[str, Any]) -> str:
    if result.get("eligible"):
        return (
            "Eligibility result: eligible. "
            "All deterministic eligibility rules passed."
        )
    reasons = result.get("reasons") or []
    reason_text = "; ".join(str(r) for r in reasons) if reasons else "rules not satisfied"
    return f"Eligibility result: ineligible. Reasons: {reason_text}."


def _format_rate_answer(result: dict[str, Any]) -> str:
    premium = result.get("monthly_premium")
    age_band = result.get("age_band") or {}
    riders = result.get("applied_riders") or []
    rider_names = ", ".join(str(r.get("rider")) for r in riders) if riders else "none"
    return (
        f"Rate result: monthly premium is ${premium} USD "
        f"(age band {age_band.get('min')}-{age_band.get('max')}, riders: {rider_names})."
    )


def _format_quote_answer(result: dict[str, Any]) -> str:
    if result.get("status") != "quoted":
        eligibility = result.get("eligibility") or {}
        reasons = eligibility.get("reasons") or []
        reason_text = "; ".join(str(r) for r in reasons) if reasons else "eligibility failed"
        return f"Quote rejected: {reason_text}."
    rate_quote = result.get("rate_quote") or {}
    return (
        f"Quote generated successfully. Estimated monthly premium: "
        f"${rate_quote.get('monthly_premium')} USD."
    )


def _generate_rag_answer(question: str, chunks: list[dict[str, Any]], *, model: str) -> str:
    context = build_context(chunks)
    if (
        _LANGCHAIN_AVAILABLE
        and ChatPromptTemplate is not None
        and ChatOpenAI is not None
        and StrOutputParser is not None
    ):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "Question:\n{question}\n\nContext:\n{context}"),
            ]
        )
        llm = ChatOpenAI(model=model, temperature=0)
        chain = prompt | llm | StrOutputParser()
        raw = chain.invoke({"question": question, "context": context})
        return ensure_chunk_citation(raw, chunks)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required")
    client = create_openai_client(api_key=openai_api_key)
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nContext:\n{context}",
            },
        ],
    )
    raw = (completion.choices[0].message.content or "").strip()
    return ensure_chunk_citation(raw, chunks)


def _classify_node(state: AssistantState) -> dict[str, Any]:
    question = str(state["question"])
    model = str(state["model"])
    if needs_clarification(question):
        return {
            "route": "rag",
            "needs_clarification": True,
            "answer": build_clarification_prompt(question),
            "chunks": [],
            "service_result": None,
        }

    decision = route_question(question, model=model)
    pending_route = str(state.get("pending_route") or "").strip().lower()
    decision_payload = decision.model_dump()
    has_profile_update = any(
        [
            decision_payload.get("age") is not None,
            decision_payload.get("smoker") is not None,
            bool(decision_payload.get("riders")),
            decision_payload.get("benefit_amount") is not None,
            decision_payload.get("policy_id") is not None,
            decision_payload.get("has_preexisting_condition") is not None,
            decision_payload.get("currently_hospitalized") is not None,
        ]
    )
    if (
        decision.route == "rag"
        and pending_route in {"eligibility", "quote", "rate"}
        and has_profile_update
    ):
        decision = RouteDecision(**{**decision_payload, "route": pending_route})

    return {
        "route": decision.route,
        "route_decision": decision.model_dump(),
        "needs_clarification": False,
    }


def _build_profile(state: AssistantState) -> dict[str, Any]:
    remembered = dict(state.get("session_profile") or {})
    decision = dict(state.get("route_decision") or {})
    merged = _merge_profile(remembered, decision)
    return {
        "age": merged.get("age"),
        "smoker": merged.get("smoker"),
        "riders": list(merged.get("riders") or []),
        "benefit_amount": float(
            merged.get("benefit_amount") or float(BASE_BENEFIT_AMOUNT)
        ),
        "policy_id": str(merged.get("policy_id") or DEFAULT_POLICY_ID),
        "has_preexisting_condition": bool(
            merged.get("has_preexisting_condition") or False
        ),
        "currently_hospitalized": bool(
            merged.get("currently_hospitalized") or False
        ),
    }


def _rag_node(state: AssistantState) -> dict[str, Any]:
    question = str(state["question"])
    model = str(state["model"])
    retrieval_options = dict(state.get("retrieval_options") or {})
    chunks = retrieve_chunks(question, **retrieval_options)
    if not chunks:
        return {
            "answer": "No relevant chunks found.",
            "chunks": [],
            "service_result": {"route": "rag", "retrieved_count": 0},
        }
    answer = _generate_rag_answer(question, chunks, model=model)
    return {
        "answer": answer,
        "chunks": chunks,
        "service_result": {"route": "rag", "retrieved_count": len(chunks)},
    }


def _eligibility_node(state: AssistantState) -> dict[str, Any]:
    profile = _build_profile(state)
    missing: list[str] = []
    if profile["age"] is None:
        missing.append("age")
    if missing:
        return {
            "answer": _format_missing_fields("eligibility", missing),
            "chunks": [],
            "service_result": None,
        }

    result = check_eligibility(
        age=int(profile["age"]),
        has_preexisting_condition=bool(profile["has_preexisting_condition"]),
        currently_hospitalized=bool(profile["currently_hospitalized"]),
        policy_id=str(profile["policy_id"]),
    )
    return {
        "answer": _format_eligibility_answer(result),
        "chunks": [],
        "service_result": result,
    }


def _rate_node(state: AssistantState) -> dict[str, Any]:
    profile = _build_profile(state)
    missing: list[str] = []
    if profile["age"] is None:
        missing.append("age")
    if profile["smoker"] is None:
        missing.append("smoker")
    if missing:
        return {
            "answer": _format_missing_fields("rate", missing),
            "chunks": [],
            "service_result": None,
        }

    smoker_value = coerce_smoker(profile["smoker"])
    result = get_rate_quote(
        age=int(profile["age"]),
        smoker=smoker_value,
        riders=list(profile["riders"]),
        benefit_amount=float(profile["benefit_amount"]),
        policy_id=str(profile["policy_id"]),
    )
    return {
        "answer": _format_rate_answer(result),
        "chunks": [],
        "service_result": result,
    }


def _quote_node(state: AssistantState) -> dict[str, Any]:
    profile = _build_profile(state)
    missing: list[str] = []
    if profile["age"] is None:
        missing.append("age")
    if profile["smoker"] is None:
        missing.append("smoker")
    if missing:
        return {
            "answer": _format_missing_fields("quote", missing),
            "chunks": [],
            "service_result": None,
        }

    smoker_value = coerce_smoker(profile["smoker"])
    result = generate_quote(
        age=int(profile["age"]),
        smoker=smoker_value,
        riders=list(profile["riders"]),
        benefit_amount=float(profile["benefit_amount"]),
        policy_id=str(profile["policy_id"]),
        has_preexisting_condition=bool(profile["has_preexisting_condition"]),
        currently_hospitalized=bool(profile["currently_hospitalized"]),
    )
    return {
        "answer": _format_quote_answer(result),
        "chunks": [],
        "service_result": result,
    }


def _select_route(state: AssistantState) -> str:
    if state.get("needs_clarification"):
        return "done"
    route = str(state.get("route") or "rag")
    if route in _ROUTE_VALUES:
        return route
    return "rag"


def _build_graph():
    graph = StateGraph(AssistantState)
    graph.add_node("classify", _classify_node)
    graph.add_node("rag", _rag_node)
    graph.add_node("eligibility", _eligibility_node)
    graph.add_node("rate", _rate_node)
    graph.add_node("quote", _quote_node)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        _select_route,
        {
            "rag": "rag",
            "eligibility": "eligibility",
            "rate": "rate",
            "quote": "quote",
            "done": END,
        },
    )
    graph.add_edge("rag", END)
    graph.add_edge("eligibility", END)
    graph.add_edge("rate", END)
    graph.add_edge("quote", END)
    return graph.compile()


_ASSISTANT_GRAPH: Any | None = None


def _run_without_langgraph(state: AssistantState) -> AssistantState:
    current = dict(state)
    current.update(_classify_node(current))
    route = _select_route(current)
    if route == "done":
        return current
    if route == "rag":
        current.update(_rag_node(current))
    elif route == "eligibility":
        current.update(_eligibility_node(current))
    elif route == "rate":
        current.update(_rate_node(current))
    elif route == "quote":
        current.update(_quote_node(current))
    return current


def _invoke_graph(state: AssistantState) -> AssistantState:
    global _ASSISTANT_GRAPH
    if not _LANGGRAPH_AVAILABLE or StateGraph is None:
        return _run_without_langgraph(state)

    if _ASSISTANT_GRAPH is None:
        _ASSISTANT_GRAPH = _build_graph()
    return _ASSISTANT_GRAPH.invoke(state)


def _profile_update_from_final_state(state: AssistantState) -> dict[str, Any]:
    updates: dict[str, Any] = {}

    decision = dict(state.get("route_decision") or {})
    updates = _merge_profile(updates, decision)

    route = str(state.get("route") or "")
    service_result = state.get("service_result")
    if not isinstance(service_result, dict):
        return updates

    if route == "eligibility":
        eligibility_profile = {
            "age": service_result.get("age"),
            "policy_id": service_result.get("policy_id"),
        }
        updates = _merge_profile(updates, eligibility_profile)
        return updates

    rate_payload: dict[str, Any] | None = None
    if route == "quote":
        maybe_rate = service_result.get("rate_quote")
        if isinstance(maybe_rate, dict):
            rate_payload = maybe_rate
    elif route == "rate":
        rate_payload = service_result

    if not isinstance(rate_payload, dict):
        return updates

    rider_entries = list(rate_payload.get("applied_riders") or [])
    rider_codes = [
        str(item.get("rider"))
        for item in rider_entries
        if isinstance(item, dict) and item.get("rider")
    ]
    rate_profile = {
        "age": rate_payload.get("age"),
        "smoker": rate_payload.get("smoker"),
        "benefit_amount": rate_payload.get("benefit_amount"),
        "policy_id": rate_payload.get("policy_id"),
        "riders": rider_codes,
    }
    return _merge_profile(updates, rate_profile)


def run_insurance_assistant(
    *,
    question: str,
    model: str,
    session_id: str | None = None,
    retrieval_options: dict[str, Any] | None = None,
    include_chunks: bool = False,
) -> dict[str, Any]:
    normalized_session_id = _normalize_session_id(session_id)
    remembered_profile = (
        dict(_ASSISTANT_SESSION_PROFILES.get(normalized_session_id) or {})
        if normalized_session_id
        else {}
    )
    initial_state: AssistantState = {
        "question": question,
        "model": model,
        "session_id": normalized_session_id,
        "session_profile": remembered_profile,
        "pending_route": (
            _ASSISTANT_PENDING_ROUTES.get(normalized_session_id)
            if normalized_session_id
            else None
        ),
        "retrieval_options": retrieval_options or {},
        "include_chunks": include_chunks,
    }
    final_state = _invoke_graph(initial_state)
    if normalized_session_id:
        updates = _profile_update_from_final_state(final_state)
        remember_session_profile(normalized_session_id, updates)
        route = str(final_state.get("route") or "").strip().lower()
        service_result = final_state.get("service_result")
        if route in {"eligibility", "quote", "rate"} and service_result is None:
            _ASSISTANT_PENDING_ROUTES[normalized_session_id] = route
        elif route in {"eligibility", "quote", "rate"} and service_result is not None:
            _ASSISTANT_PENDING_ROUTES.pop(normalized_session_id, None)
    chunks = list(final_state.get("chunks") or [])
    if not include_chunks:
        chunks = []
    return {
        "route": str(final_state.get("route") or "rag"),
        "answer": str(final_state.get("answer") or ""),
        "needs_clarification": bool(final_state.get("needs_clarification") or False),
        "chunks": chunks,
        "service_result": final_state.get("service_result"),
    }
