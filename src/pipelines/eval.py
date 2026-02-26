from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from core.openai_client import create_openai_client, langfuse_enabled
from rag.answer import build_context
from rag.ambiguity import asked_clarifying_question, build_clarification_prompt
from rag.citation import ensure_chunk_citation, has_chunk_citation
from core.config import get_config
from rag.retrieve import retrieve_chunks

try:
    from langfuse import get_client
except Exception:  # pragma: no cover - optional dependency guard
    get_client = None  # type: ignore[assignment]


def load_questions(path: Path, *, max_questions: int) -> list[dict]:
    if path.exists():
        raw = path.read_text(encoding="utf-8").strip()
        if raw:
            data = json.loads(raw)
            if isinstance(data, list):
                questions: list[dict] = []
                for item in data:
                    if isinstance(item, str):
                        questions.append(
                            {
                                "question": item,
                                "expected_sections": [],
                                "category": "uncategorized",
                                "expected_behavior": "answerable",
                            }
                        )
                    elif isinstance(item, dict) and isinstance(item.get("question"), str):
                        expected_sections = item.get("expected_sections")
                        if not isinstance(expected_sections, list):
                            expected_sections = []
                        category = item.get("category")
                        if not isinstance(category, str) or not category.strip():
                            category = "uncategorized"
                        expected_behavior = item.get("expected_behavior")
                        if (
                            not isinstance(expected_behavior, str)
                            or not expected_behavior.strip()
                        ):
                            expected_behavior = "answerable"
                        questions.append(
                            {
                                "question": item["question"],
                                "expected_sections": [
                                    str(s) for s in expected_sections if isinstance(s, str)
                                ],
                                "category": category,
                                "expected_behavior": expected_behavior,
                            }
                        )
                if questions:
                    return questions[:max_questions]

    # Keep eval small for now; replace with richer policy questions later.
    return [
        {
            "question": "What illnesses are covered by this policy?",
            "expected_sections": ["SECTION 4"],
            "category": "common",
            "expected_behavior": "answerable",
        },
        {
            "question": "What are the main exclusions?",
            "expected_sections": ["SECTION 6"],
            "category": "common",
            "expected_behavior": "answerable",
        },
        {
            "question": "How do I submit a claim?",
            "expected_sections": ["SECTION 7"],
            "category": "common",
            "expected_behavior": "answerable",
        },
    ][:max_questions]


def ask_with_context(
    question: str,
    chunks: list[dict],
    client,
    *,
    openai_request_kwargs: dict[str, Any] | None = None,
) -> str:
    if not chunks:
        return "No relevant chunks found."

    models_cfg = get_config()["models"]
    context = build_context(chunks)
    completion = client.chat.completions.create(
        model=str(models_cfg["answer_model"]),
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer insurance-policy questions using ONLY the provided context. "
                    "If context is insufficient, say so clearly. "
                    "Cite chunk ids in square brackets like [demolife_critical_illness_policy.md:3]. "
                    "Citations must use exactly one colon and integer chunk index. "
                    "Do not cite section clause numbers like [demolife_critical_illness_policy.md:3.2]."
                ),
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nContext:\n{context}",
            },
        ],
        **(openai_request_kwargs or {}),
    )
    raw = completion.choices[0].message.content or ""
    return ensure_chunk_citation(raw, chunks)


def write_jsonl(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def has_citation(answer: str) -> bool:
    # Accept chunk-id citations (":3") and clause-style citations (":3.2").
    # This keeps groundedness robust to minor output formatting variance.
    return has_chunk_citation(answer)


def is_insufficient_context(answer: str) -> bool:
    low = answer.lower()
    insufficient_markers = [
        "insufficient context",
        "no relevant chunks found",
        "does not include",
        "doesn't include",
        "does not contain",
        "doesn't contain",
        "does not mention",
        "doesn't mention",
        "does not specify",
        "doesn't specify",
        "not enough information",
        "does not provide enough information",
        "doesn't provide enough information",
        "not enough context",
        "not provided in the context",
        "not in the context",
        "cannot answer",
        "can't answer",
        "cannot compare",
        "can't compare",
        "cannot project",
        "can't project",
        "unable to answer",
        "unable to compare",
        "i would need more specific information",
        "not included in the information provided",
        "not provided in the information provided",
        "please provide details",
    ]
    return any(marker in low for marker in insufficient_markers)


def retrieval_hit(chunks: list[dict], expected_sections: list[str]) -> bool | None:
    if not expected_sections:
        return None
    lowered_targets = [s.lower() for s in expected_sections]
    for chunk in chunks:
        section = str(chunk["metadata"].get("section") or "").lower()
        if any(target in section for target in lowered_targets):
            return True
    return False


def retrieval_relevance_score(chunks: list[dict], expected_sections: list[str]) -> float | None:
    if not expected_sections:
        return None
    lowered_targets = [s.lower() for s in expected_sections]
    best_rank: int | None = None
    for rank, chunk in enumerate(chunks, start=1):
        section = str(chunk["metadata"].get("section") or "").lower()
        if any(target in section for target in lowered_targets):
            best_rank = rank
            break
    if best_rank is None:
        return 0.0
    return round(max(0.0, 1.0 - (0.2 * (best_rank - 1))), 3)


_CITATION_PATTERN = re.compile(r"\[([^\[\]]+?\.md:\d+(?:\.\d+)?)\]")


def _normalize_citation_id(citation: str) -> str | None:
    token = str(citation or "").strip()
    if ":" not in token:
        return None
    source, idx_part = token.rsplit(":", 1)
    index = idx_part.split(".", 1)[0].strip()
    if not source.strip() or not index.isdigit():
        return None
    return f"{source.strip()}:{int(index)}"


def grounded_score(answer: str, chunks: list[dict]) -> float:
    if not chunks:
        return 0.0
    raw_ids = [m.group(1) for m in _CITATION_PATTERN.finditer(answer or "")]
    normalized = [_normalize_citation_id(c) for c in raw_ids]
    citation_ids = [c for c in normalized if c]
    if not citation_ids:
        return 0.0
    valid_ids = {str(chunk.get("id") or "").strip() for chunk in chunks}
    valid_count = sum(1 for cid in citation_ids if cid in valid_ids)
    return round(valid_count / max(1, len(citation_ids)), 3)


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}


def _tokenize_for_overlap(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {t for t in tokens if len(t) > 2 and t not in _STOPWORDS}


def answer_relevance_score(
    question: str,
    answer: str,
    answer_relevance: bool | None,
) -> float | None:
    if answer_relevance is None:
        return None
    if not answer_relevance:
        return 0.0
    q_tokens = _tokenize_for_overlap(question)
    if not q_tokens:
        return 1.0
    a_tokens = _tokenize_for_overlap(answer)
    overlap = len(q_tokens & a_tokens) / max(1, len(q_tokens))
    return round(min(1.0, 0.6 + (0.4 * overlap)), 3)


def pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return (numerator / denominator) * 100.0


def empty_category_stats() -> dict[str, int]:
    return {
        "total": 0,
        "retrieval_checks": 0,
        "retrieval_hits": 0,
        "grounded_checks": 0,
        "grounded": 0,
        "answer_relevance_checks": 0,
        "answer_relevance_hits": 0,
        "insufficient_context": 0,
        "failure": 0,
        "clarification_needed": 0,
        "asked_clarifying_question": 0,
        "expected_unanswerable": 0,
        "handled_unanswerable": 0,
        "expected_answerable": 0,
        "answerable_but_insufficient": 0,
        "answerable_with_failure": 0,
    }


def _get_langfuse_client() -> Any | None:
    if not langfuse_enabled():
        return None
    if get_client is None:
        return None
    try:
        return get_client()
    except Exception:
        return None


def _score_langfuse_numeric(
    *,
    client: Any | None,
    trace_id: str | None,
    name: str,
    value: float | None,
    comment: str = "",
) -> None:
    if client is None or not trace_id or value is None:
        return
    try:
        client.create_score(
            name=name,
            value=float(value),
            data_type="NUMERIC",
            trace_id=trace_id,
            comment=comment,
        )
    except Exception:
        return


def _build_langfuse_openai_kwargs(
    *,
    trace_id: str | None,
    category: str,
    question_index: int,
) -> dict[str, Any] | None:
    if not trace_id:
        return None
    return {
        "trace_id": trace_id,
        "metadata": {
            "pipeline": "eval",
            "question_index": question_index,
            "category": category,
        },
    }


def main() -> None:
    cfg = get_config()
    eval_cfg = cfg["eval"]

    parser = argparse.ArgumentParser(description="Run batch RAG evaluation.")
    parser.add_argument(
        "--questions-path",
        default=str(eval_cfg["questions_path"]),
        help="Path to eval question JSON file.",
    )
    parser.add_argument(
        "--output-path",
        default=str(eval_cfg["output_path"]),
        help="Path to output JSONL results.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(eval_cfg["top_k"]),
        help="Number of retrieved chunks per answerable question.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=int(eval_cfg["max_questions"]),
        help="Maximum number of questions to evaluate.",
    )
    args = parser.parse_args()

    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required")

    questions_path = Path(args.questions_path)
    output_path = Path(args.output_path)
    top_k = max(1, args.top_k)
    max_questions = max(1, args.max_questions)

    questions = load_questions(questions_path, max_questions=max_questions)
    client = create_openai_client(api_key=openai_api_key)
    langfuse_client = _get_langfuse_client()

    results: list[dict] = []
    retrieval_checks = 0
    retrieval_hits = 0
    grounded_checks = 0
    grounded_answers = 0
    answer_relevance_checks = 0
    answer_relevance_hits = 0
    insufficient_context_count = 0
    failures = 0
    clarification_needed_count = 0
    asked_clarification_count = 0
    expected_unanswerable = 0
    handled_unanswerable = 0
    expected_answerable = 0
    answerable_but_insufficient = 0
    answerable_with_failure = 0
    category_stats: dict[str, dict[str, int]] = {}

    for i, item in enumerate(questions, start=1):
        question = item["question"]
        expected_sections = item["expected_sections"]
        category = item.get("category", "uncategorized")
        expected_behavior = item.get("expected_behavior", "answerable")
        clarification_needed = category == "ambiguous"
        if clarification_needed:
            expected_behavior = "needs_clarification"

        chunks: list[dict] = []
        answer = ""
        failure = False
        asked_for_clarification = False
        error_message: str | None = None
        retrieval_latency_ms: float | None = None
        total_latency_ms: float | None = None
        retrieval_cache_hit: bool | None = None
        langfuse_trace_id: str | None = None
        openai_request_kwargs: dict[str, Any] | None = None
        if langfuse_client is not None:
            try:
                langfuse_trace_id = str(langfuse_client.create_trace_id())
                openai_request_kwargs = _build_langfuse_openai_kwargs(
                    trace_id=langfuse_trace_id,
                    category=category,
                    question_index=i,
                )
            except Exception:
                langfuse_trace_id = None
                openai_request_kwargs = None

        question_start = time.perf_counter()
        try:
            if clarification_needed:
                answer = build_clarification_prompt(question)
                asked_for_clarification = asked_clarifying_question(answer)
            else:
                retrieval_start = time.perf_counter()
                chunks, retrieval_meta = retrieve_chunks(
                    question,
                    top_k=top_k,
                    openai_request_kwargs=openai_request_kwargs,
                    return_meta=True,
                )
                retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0
                retrieval_cache_hit = retrieval_meta.get("retrieval_cache_hit")
                answer = ask_with_context(
                    question,
                    chunks,
                    client,
                    openai_request_kwargs=openai_request_kwargs,
                )
        except Exception as exc:
            failure = True
            error_message = f"{type(exc).__name__}: {exc}"
            answer = f"System failure during evaluation. {error_message}"
        finally:
            total_latency_ms = (time.perf_counter() - question_start) * 1000.0

        hit = (
            None
            if failure or clarification_needed
            else retrieval_hit(chunks, expected_sections)
        )
        is_unanswerable_expected = expected_behavior in {
            "not_available",
            "needs_clarification",
        }
        grounded = (
            has_citation(answer)
            if (not failure) and (not is_unanswerable_expected)
            else None
        )
        insufficient_context = (
            False
            if clarification_needed
            else (not failure) and is_insufficient_context(answer)
        )
        answer_relevance = (
            (not insufficient_context)
            if (not failure) and (not is_unanswerable_expected)
            else None
        )
        hit_score = (
            None
            if failure or clarification_needed
            else retrieval_relevance_score(chunks, expected_sections)
        )
        grounded_numeric = (
            grounded_score(answer, chunks)
            if (not failure) and (not is_unanswerable_expected)
            else None
        )
        answer_relevance_numeric = answer_relevance_score(
            question,
            answer,
            answer_relevance,
        )
        _score_langfuse_numeric(
            client=langfuse_client,
            trace_id=langfuse_trace_id,
            name="retrieval_score",
            value=hit_score,
            comment="Eval retrieval relevance score (0..1), rank-aware.",
        )
        _score_langfuse_numeric(
            client=langfuse_client,
            trace_id=langfuse_trace_id,
            name="grounded",
            value=grounded_numeric,
            comment="Eval groundedness score (0..1) based on citation validity.",
        )
        _score_langfuse_numeric(
            client=langfuse_client,
            trace_id=langfuse_trace_id,
            name="answer_relevance",
            value=answer_relevance_numeric,
            comment="Eval answer relevance score (0..1), overlap-weighted.",
        )

        if hit is not None:
            retrieval_checks += 1
            if hit:
                retrieval_hits += 1
        if grounded is not None:
            grounded_checks += 1
            if grounded:
                grounded_answers += 1
        if answer_relevance is not None:
            answer_relevance_checks += 1
            if answer_relevance:
                answer_relevance_hits += 1
        if insufficient_context:
            insufficient_context_count += 1
        if failure:
            failures += 1
        if clarification_needed:
            clarification_needed_count += 1
            if asked_for_clarification:
                asked_clarification_count += 1
        if is_unanswerable_expected:
            expected_unanswerable += 1
            if expected_behavior == "needs_clarification":
                if asked_for_clarification:
                    handled_unanswerable += 1
            elif insufficient_context:
                handled_unanswerable += 1
        else:
            expected_answerable += 1
            if insufficient_context:
                answerable_but_insufficient += 1
            if failure:
                answerable_with_failure += 1

        stats = category_stats.setdefault(category, empty_category_stats())
        stats["total"] += 1
        if hit is not None:
            stats["retrieval_checks"] += 1
            if hit:
                stats["retrieval_hits"] += 1
        if grounded is not None:
            stats["grounded_checks"] += 1
            if grounded:
                stats["grounded"] += 1
        if answer_relevance is not None:
            stats["answer_relevance_checks"] += 1
            if answer_relevance:
                stats["answer_relevance_hits"] += 1
        if insufficient_context:
            stats["insufficient_context"] += 1
        if failure:
            stats["failure"] += 1
        if clarification_needed:
            stats["clarification_needed"] += 1
            if asked_for_clarification:
                stats["asked_clarifying_question"] += 1
        if is_unanswerable_expected:
            stats["expected_unanswerable"] += 1
            if expected_behavior == "needs_clarification":
                if asked_for_clarification:
                    stats["handled_unanswerable"] += 1
            elif insufficient_context:
                stats["handled_unanswerable"] += 1
        else:
            stats["expected_answerable"] += 1
            if insufficient_context:
                stats["answerable_but_insufficient"] += 1
            if failure:
                stats["answerable_with_failure"] += 1

        results.append(
            {
                "question": question,
                "category": category,
                "expected_behavior": expected_behavior,
                "top_k": top_k,
                "expected_sections": expected_sections,
                "retrieved": [
                    {
                        "id": c["id"],
                        "distance": c["distance"],
                        "source": c["metadata"]["source"],
                        "section": c["metadata"]["section"],
                    }
                    for c in chunks
                ],
                "answer": answer,
                "error": error_message,
                "metrics": {
                    "retrieval_hit": hit,
                    "grounded": grounded,
                    "answer_relevance": answer_relevance,
                    "retrieval_score": hit_score,
                    "grounded_score": grounded_numeric,
                    "answer_relevance_score": answer_relevance_numeric,
                    "retrieval_latency_ms": (
                        round(retrieval_latency_ms, 2)
                        if retrieval_latency_ms is not None
                        else None
                    ),
                    "total_latency_ms": (
                        round(total_latency_ms, 2)
                        if total_latency_ms is not None
                        else None
                    ),
                    "retrieval_cache_hit": retrieval_cache_hit,
                    "langfuse_trace_id": langfuse_trace_id,
                    "insufficient_context": insufficient_context,
                    "clarification_needed": clarification_needed,
                    "asked_clarifying_question": asked_for_clarification,
                    "failure": failure,
                    "failure_type": "system" if failure else None,
                },
            }
        )
        print(f"[{i}/{len(questions)}] Retrieved {len(chunks)} chunks")
    if langfuse_client is not None:
        try:
            langfuse_client.flush()
        except Exception:
            pass

    write_jsonl(results, output_path)
    print(f"Saved eval results to: {output_path}")
    print(
        f"Retrieval hit rate: {retrieval_hits}/{retrieval_checks} "
        f"({pct(retrieval_hits, retrieval_checks):.1f}%)"
    )
    print(
        f"Grounding rate: {grounded_answers}/{grounded_checks} "
        f"({pct(grounded_answers, grounded_checks):.1f}%)"
    )
    print(
        f"Answer relevance rate: {answer_relevance_hits}/{answer_relevance_checks} "
        f"({pct(answer_relevance_hits, answer_relevance_checks):.1f}%)"
    )
    print(
        f"Insufficient-context rate: {insufficient_context_count}/{len(questions)} "
        f"({pct(insufficient_context_count, len(questions)):.1f}%)"
    )
    print(
        f"Failure rate: {failures}/{len(questions)} "
        f"({pct(failures, len(questions)):.1f}%)"
    )
    print(
        f"Clarifying-question asked: {asked_clarification_count}/"
        f"{clarification_needed_count} "
        f"({pct(asked_clarification_count, clarification_needed_count):.1f}%)"
    )
    print(
        f"Expected-unanswerable handled: {handled_unanswerable}/{expected_unanswerable} "
        f"({pct(handled_unanswerable, expected_unanswerable):.1f}%)"
    )
    print(
        f"Answerable-but-insufficient: {answerable_but_insufficient}/{expected_answerable} "
        f"({pct(answerable_but_insufficient, expected_answerable):.1f}%)"
    )
    print(
        f"Answerable-with-failure: {answerable_with_failure}/"
        f"{expected_answerable} "
        f"({pct(answerable_with_failure, expected_answerable):.1f}%)"
    )

    print("\nCategory breakdown:")
    for category in sorted(category_stats.keys()):
        stats = category_stats[category]
        print(
            f"- {category}: "
            f"retrieval {stats['retrieval_hits']}/{stats['retrieval_checks']} "
            f"({pct(stats['retrieval_hits'], stats['retrieval_checks']):.1f}%), "
            f"grounded {stats['grounded']}/{stats['grounded_checks']} "
            f"({pct(stats['grounded'], stats['grounded_checks']):.1f}%), "
            f"answer relevance {stats['answer_relevance_hits']}/"
            f"{stats['answer_relevance_checks']} "
            f"({pct(stats['answer_relevance_hits'], stats['answer_relevance_checks']):.1f}%), "
            f"insufficient {stats['insufficient_context']}/{stats['total']} "
            f"({pct(stats['insufficient_context'], stats['total']):.1f}%), "
            f"failure {stats['failure']}/{stats['total']} "
            f"({pct(stats['failure'], stats['total']):.1f}%), "
            f"clarifying q {stats['asked_clarifying_question']}/"
            f"{stats['clarification_needed']} "
            f"({pct(stats['asked_clarifying_question'], stats['clarification_needed']):.1f}%), "
            f"unanswerable handled {stats['handled_unanswerable']}/"
            f"{stats['expected_unanswerable']} "
            f"({pct(stats['handled_unanswerable'], stats['expected_unanswerable']):.1f}%), "
            f"answerable but insufficient {stats['answerable_but_insufficient']}/"
            f"{stats['expected_answerable']} "
            f"({pct(stats['answerable_but_insufficient'], stats['expected_answerable']):.1f}%), "
            f"answerable with failure {stats['answerable_with_failure']}/"
            f"{stats['expected_answerable']} "
            f"({pct(stats['answerable_with_failure'], stats['expected_answerable']):.1f}%)"
        )


if __name__ == "__main__":
    main()

