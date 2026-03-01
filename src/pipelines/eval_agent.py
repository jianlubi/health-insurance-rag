from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from assistant.orchestrator import clear_session_profile, run_insurance_assistant
from core.config import get_config


VALID_ROUTES = {"rag", "eligibility", "quote", "rate"}
VALID_TOOLS = {"eligibility", "quote", "rate"}
DEFAULT_TOOL_KEYS: dict[str, list[str]] = {
    "eligibility": ["eligible", "reasons", "evaluated_rules"],
    "rate": ["monthly_premium", "policy_id", "age"],
    "quote": ["status", "eligibility", "rate_quote", "application_url"],
}


def pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return (numerator / denominator) * 100.0


def write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_cases(path: Path, *, max_questions: int) -> list[dict[str, Any]]:
    if path.exists():
        raw = path.read_text(encoding="utf-8").strip()
        if raw:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                rows: list[dict[str, Any]] = []
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    question = item.get("question")
                    expected_route = str(item.get("expected_route") or "").strip().lower()
                    if not isinstance(question, str) or not question.strip():
                        continue
                    if expected_route not in VALID_ROUTES:
                        continue
                    rows.append(dict(item))
                if rows:
                    return rows[:max_questions]

    return [
        {
            "question": "What illnesses are covered by this policy?",
            "expected_route": "rag",
            "category": "routing",
        },
        {
            "question": "Can I get a quote?",
            "expected_route": "quote",
            "expected_tool": "quote",
            "expected_tool_called": False,
            "expected_answer_contains": [
                "i can generate that quote",
                "your age",
                "whether you are a smoker",
            ],
            "category": "tool_correctness",
        },
    ][:max_questions]


def _contains_all_markers(answer: str, markers: list[str]) -> bool:
    low = answer.lower()
    return all(str(marker).strip().lower() in low for marker in markers if str(marker).strip())


def _normalize_expected_tool(case: dict[str, Any]) -> str | None:
    expected_tool = case.get("expected_tool")
    if not isinstance(expected_tool, str):
        return None
    token = expected_tool.strip().lower()
    if token not in VALID_TOOLS:
        return None
    return token


def _tool_schema_correct(
    *,
    tool_name: str,
    service_result: dict[str, Any],
    expected_service_keys: list[str] | None,
    expected_quote_status: str | None,
    expected_rate_quote_is_none: bool | None,
) -> bool:
    keys = expected_service_keys or DEFAULT_TOOL_KEYS[tool_name]
    if not all(key in service_result for key in keys):
        return False

    if tool_name == "quote" and expected_quote_status:
        if str(service_result.get("status") or "").strip().lower() != expected_quote_status:
            return False

    if tool_name == "quote" and expected_rate_quote_is_none is not None:
        rate_quote_is_none = service_result.get("rate_quote") is None
        if rate_quote_is_none is not bool(expected_rate_quote_is_none):
            return False

    return True


def evaluate_case(
    *,
    index: int,
    case: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    question = str(case["question"]).strip()
    expected_route = str(case["expected_route"]).strip().lower()
    expected_tool = _normalize_expected_tool(case)
    expected_tool_called = (
        bool(case["expected_tool_called"])
        if "expected_tool_called" in case
        else (expected_tool is not None)
    )
    expected_needs_clarification = case.get("expected_needs_clarification")
    if expected_needs_clarification is not None:
        expected_needs_clarification = bool(expected_needs_clarification)
    expected_answer_contains = [
        str(item)
        for item in list(case.get("expected_answer_contains") or [])
        if str(item).strip()
    ]
    expected_quote_status = case.get("expected_quote_status")
    if isinstance(expected_quote_status, str):
        expected_quote_status = expected_quote_status.strip().lower() or None
    else:
        expected_quote_status = None
    expected_rate_quote_is_none = case.get("expected_rate_quote_is_none")
    if expected_rate_quote_is_none is not None:
        expected_rate_quote_is_none = bool(expected_rate_quote_is_none)

    expected_service_keys_raw = case.get("expected_service_keys")
    expected_service_keys: list[str] | None = None
    if isinstance(expected_service_keys_raw, list):
        expected_service_keys = [
            str(k).strip()
            for k in expected_service_keys_raw
            if str(k).strip()
        ]

    session_id = str(
        case.get("session_id")
        or f"agent-eval-{index}-{uuid.uuid4().hex[:8]}"
    ).strip()
    clear_before = bool(case.get("clear_session_before", True))
    if clear_before:
        clear_session_profile(session_id)

    started = time.perf_counter()
    failure = False
    error_message: str | None = None
    result: dict[str, Any] = {}
    try:
        result = run_insurance_assistant(
            question=question,
            model=model,
            session_id=session_id,
            include_chunks=False,
        )
    except Exception as exc:
        failure = True
        error_message = f"{type(exc).__name__}: {exc}"
        result = {
            "route": "",
            "answer": f"System failure during agent eval. {error_message}",
            "needs_clarification": False,
            "service_result": None,
        }
    latency_ms = (time.perf_counter() - started) * 1000.0

    actual_route = str(result.get("route") or "").strip().lower()
    answer = str(result.get("answer") or "")
    needs_clarification = bool(result.get("needs_clarification") or False)
    service_result_raw = result.get("service_result")
    service_result = service_result_raw if isinstance(service_result_raw, dict) else None

    routing_correct = actual_route == expected_route

    tool_called_correct: bool | None = None
    tool_schema_ok: bool | None = None
    tool_correct: bool | None = None
    if expected_tool is not None:
        actual_tool_called = service_result is not None
        tool_called_correct = actual_tool_called == expected_tool_called
        if expected_tool_called:
            if service_result is None:
                tool_schema_ok = False
            else:
                tool_schema_ok = _tool_schema_correct(
                    tool_name=expected_tool,
                    service_result=service_result,
                    expected_service_keys=expected_service_keys,
                    expected_quote_status=expected_quote_status,
                    expected_rate_quote_is_none=expected_rate_quote_is_none,
                )
        else:
            tool_schema_ok = service_result is None
        tool_correct = bool(tool_called_correct and tool_schema_ok)

    clarification_correct: bool | None = None
    if expected_needs_clarification is not None:
        clarification_correct = needs_clarification == expected_needs_clarification

    answer_contains_correct: bool | None = None
    if expected_answer_contains:
        answer_contains_correct = _contains_all_markers(answer, expected_answer_contains)

    checks: list[bool] = [routing_correct]
    for optional_check in (
        tool_correct,
        clarification_correct,
        answer_contains_correct,
    ):
        if optional_check is not None:
            checks.append(optional_check)
    passed = (not failure) and all(checks)

    return {
        "question": question,
        "category": str(case.get("category") or "uncategorized"),
        "session_id": session_id,
        "expected": {
            "route": expected_route,
            "tool": expected_tool,
            "tool_called": expected_tool_called if expected_tool is not None else None,
            "needs_clarification": expected_needs_clarification,
            "answer_contains": expected_answer_contains or None,
            "quote_status": expected_quote_status,
            "rate_quote_is_none": expected_rate_quote_is_none,
            "service_keys": expected_service_keys,
        },
        "actual": {
            "route": actual_route,
            "needs_clarification": needs_clarification,
            "answer": answer,
            "service_result": service_result_raw,
        },
        "metrics": {
            "routing_correct": routing_correct,
            "tool_called_correct": tool_called_correct,
            "tool_schema_correct": tool_schema_ok,
            "tool_correct": tool_correct,
            "clarification_correct": clarification_correct,
            "answer_contains_correct": answer_contains_correct,
            "passed": passed,
            "failure": failure,
            "failure_type": "system" if failure else None,
            "latency_ms": round(latency_ms, 2),
        },
        "error": error_message,
    }


def main() -> None:
    cfg = get_config()
    eval_agent_cfg = cfg["eval_agent"]
    models_cfg = cfg["models"]

    parser = argparse.ArgumentParser(
        description="Run agent routing and tool correctness evaluation."
    )
    parser.add_argument(
        "--questions-path",
        default=str(eval_agent_cfg["questions_path"]),
        help="Path to agent eval question JSON file.",
    )
    parser.add_argument(
        "--output-path",
        default=str(eval_agent_cfg["output_path"]),
        help="Path to agent eval output JSONL file.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=int(eval_agent_cfg["max_questions"]),
        help="Maximum number of cases to evaluate.",
    )
    parser.add_argument(
        "--model",
        default=str(models_cfg["answer_model"]),
        help="Router/assistant model name.",
    )
    parser.add_argument(
        "--routing-mode",
        choices=("auto", "fallback"),
        default="auto",
        help="Set to 'fallback' to force regex-based routing (no LLM router).",
    )
    args = parser.parse_args()

    load_dotenv()
    if args.routing_mode == "fallback":
        os.environ.pop("OPENAI_API_KEY", None)

    questions_path = Path(args.questions_path)
    output_path = Path(args.output_path)
    max_questions = max(1, int(args.max_questions))

    cases = load_cases(questions_path, max_questions=max_questions)
    results: list[dict[str, Any]] = []

    routing_checks = 0
    routing_correct = 0
    tool_checks = 0
    tool_correct = 0
    clarification_checks = 0
    clarification_correct = 0
    answer_contains_checks = 0
    answer_contains_correct = 0
    passes = 0
    failures = 0

    for i, case in enumerate(cases, start=1):
        row = evaluate_case(index=i, case=case, model=str(args.model))
        results.append(row)
        m = row["metrics"]

        routing_checks += 1
        if m["routing_correct"]:
            routing_correct += 1

        if m["tool_correct"] is not None:
            tool_checks += 1
            if m["tool_correct"]:
                tool_correct += 1

        if m["clarification_correct"] is not None:
            clarification_checks += 1
            if m["clarification_correct"]:
                clarification_correct += 1

        if m["answer_contains_correct"] is not None:
            answer_contains_checks += 1
            if m["answer_contains_correct"]:
                answer_contains_correct += 1

        if m["passed"]:
            passes += 1
        if m["failure"]:
            failures += 1

        status = "PASS" if m["passed"] else "FAIL"
        print(f"[{i}/{len(cases)}] {status} - route={row['actual']['route']}")

    write_jsonl(results, output_path)
    print(f"Saved agent eval results to: {output_path}")
    print(
        f"Routing accuracy: {routing_correct}/{routing_checks} "
        f"({pct(routing_correct, routing_checks):.1f}%)"
    )
    print(
        f"Tool correctness: {tool_correct}/{tool_checks} "
        f"({pct(tool_correct, tool_checks):.1f}%)"
    )
    print(
        f"Clarification correctness: {clarification_correct}/{clarification_checks} "
        f"({pct(clarification_correct, clarification_checks):.1f}%)"
    )
    print(
        f"Answer-content checks: {answer_contains_correct}/{answer_contains_checks} "
        f"({pct(answer_contains_correct, answer_contains_checks):.1f}%)"
    )
    print(f"Overall pass rate: {passes}/{len(cases)} ({pct(passes, len(cases)):.1f}%)")
    print(f"Failure rate: {failures}/{len(cases)} ({pct(failures, len(cases)):.1f}%)")


if __name__ == "__main__":
    main()
