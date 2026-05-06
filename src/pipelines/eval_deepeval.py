from __future__ import annotations

import argparse
import os
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv

from core.config import get_config
from core.openai_client import create_openai_client
from pipelines.eval import ask_with_context, load_questions, write_jsonl
from rag.citation import ensure_chunk_citation
from rag.retrieve import retrieve_chunks

_UNANSWERABLE_BEHAVIORS = {"not_available", "needs_clarification"}
_REFERENCE_METRIC_NAMES = ("contextual_precision", "contextual_recall")
_METRIC_PRINT_ORDER = (
    "answer_relevancy",
    "faithfulness",
    "contextual_relevancy",
    "contextual_precision",
    "contextual_recall",
)
_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
_LOCAL_PROXY_HOSTS = {"127.0.0.1", "localhost", "::1"}
_CITATION_PATTERN = re.compile(r"\[([^\[\]]+?\.md:\d+(?:\.\d+)?)\]")
_OVERLAP_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
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


def pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return (numerator / denominator) * 100.0


def _expected_behavior(item: dict[str, Any]) -> str:
    raw = item.get("expected_behavior")
    if not isinstance(raw, str):
        return "answerable"
    token = raw.strip().lower()
    return token or "answerable"


def _extract_expected_output(item: dict[str, Any]) -> str | None:
    raw = item.get("expected_output")
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    return value or None


def _build_retrieval_context(chunks: list[dict[str, Any]]) -> list[str]:
    if not chunks:
        return ["No retrieval context found."]

    context: list[str] = []
    for chunk in chunks:
        content = str(chunk.get("content") or "").strip()
        if content:
            context.append(content)
    if not context:
        return ["No retrieval context found."]
    return context


def _normalize_citation_id(citation: str) -> str | None:
    token = str(citation or "").strip()
    if ":" not in token:
        return None
    source, idx_part = token.rsplit(":", 1)
    index = idx_part.split(".", 1)[0].strip()
    if not source.strip() or not index.isdigit():
        return None
    return f"{source.strip()}:{int(index)}"


def _extract_cited_chunk_ids(answer: str) -> set[str]:
    raw_ids = [m.group(1) for m in _CITATION_PATTERN.finditer(answer or "")]
    normalized = [_normalize_citation_id(citation) for citation in raw_ids]
    return {token for token in normalized if token}


def _select_metric_chunks(*, answer: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cited_ids = _extract_cited_chunk_ids(answer)
    if not cited_ids:
        return chunks
    selected = [chunk for chunk in chunks if str(chunk.get("id") or "").strip() in cited_ids]
    return selected or chunks


def _citation_validity_score(*, answer: str, chunks: list[dict[str, Any]]) -> float:
    cited_ids = _extract_cited_chunk_ids(answer)
    if not cited_ids:
        return 0.0
    valid_ids = {str(chunk.get("id") or "").strip() for chunk in chunks}
    matched = sum(1 for cited in cited_ids if cited in valid_ids)
    return matched / max(1, len(cited_ids))


def _tokenize_for_overlap(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {t for t in tokens if len(t) > 2 and t not in _OVERLAP_STOPWORDS}


def _split_into_sentences(text: str) -> list[str]:
    raw_parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    return [part.strip() for part in raw_parts if part and part.strip()]


def _compact_answer_for_metrics(
    *,
    answer: str,
    max_sentences: int,
    max_chars: int,
) -> str:
    original = answer or ""
    normalized = (answer or "").replace("\n- ", ". ").replace("\n", " ").strip()
    # Strip low-signal preambles that tend to hurt answer relevancy scoring.
    normalized = re.sub(
        r"^\s*(based on the provided context|according to the provided context|from the context)[,:]?\s*",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    sentences = _split_into_sentences(normalized)
    if not sentences:
        compact = _clip_text(normalized, max_chars)
    else:
        compact = " ".join(sentences[: max(1, max_sentences)]).strip()
        compact = _clip_text(compact, max_chars)

    if not _CITATION_PATTERN.search(compact):
        first = _CITATION_PATTERN.search(original)
        if first is not None:
            candidate = f"{compact} [{first.group(1)}]".strip()
            compact = _clip_text(candidate, max_chars)
    return compact


def _build_context_for_rewrite(chunks: list[dict[str, Any]], *, max_chars: int) -> str:
    if not chunks:
        return "No retrieval context found."
    parts: list[str] = []
    for chunk in chunks:
        chunk_id = str(chunk.get("id") or "").strip() or "chunk"
        content = _clip_text(str(chunk.get("content") or ""), max_chars)
        parts.append(f"[{chunk_id}] {content}")
    return "\n\n".join(parts)


def _rewrite_metric_answer(
    *,
    question: str,
    answer: str,
    chunks: list[dict[str, Any]],
    client: Any,
    model: str,
    max_chars: int,
) -> str:
    if not answer.strip():
        return answer
    context = _build_context_for_rewrite(chunks, max_chars=max_chars)
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Rewrite the answer to be direct, concise, and strictly faithful to the provided context. "
                    "Keep 1-2 sentences. Do not add assumptions. "
                    "Preserve or add chunk-id citations like [file.md:3]."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Original answer:\n{answer}\n\n"
                    f"Context:\n{context}"
                ),
            },
        ],
    )
    rewritten = str(completion.choices[0].message.content or "").strip()
    if not rewritten:
        return answer
    rewritten = _clip_text(rewritten, max_chars)
    return ensure_chunk_citation(rewritten, chunks)


def _compress_chunk_content(
    *,
    question_tokens: set[str],
    content: str,
    max_sentences: int,
) -> tuple[str, float]:
    sentences = _split_into_sentences(content)
    if not sentences:
        return content.strip(), 0.0

    scored: list[tuple[float, int, str]] = []
    for idx, sentence in enumerate(sentences):
        sentence_tokens = _tokenize_for_overlap(sentence)
        if question_tokens:
            overlap = len(question_tokens & sentence_tokens) / max(1, len(question_tokens))
        else:
            overlap = 0.0
        scored.append((overlap, idx, sentence))

    positive = [row for row in scored if row[0] > 0.0]
    candidates = positive if positive else scored
    top = sorted(candidates, key=lambda row: (row[0], -len(row[2])), reverse=True)[
        : max(1, max_sentences)
    ]
    keep_indices = {row[1] for row in top}
    selected = [sent for i, sent in enumerate(sentences) if i in keep_indices]
    compressed = " ".join(selected).strip()
    if not compressed:
        compressed = sentences[0]
    best_score = max((row[0] for row in scored), default=0.0)
    return compressed, best_score


def _compress_chunks_for_question(
    *,
    question: str,
    chunks: list[dict[str, Any]],
    max_sentences: int,
    drop_irrelevant: bool,
) -> list[dict[str, Any]]:
    if not chunks:
        return chunks

    question_tokens = _tokenize_for_overlap(question)
    compressed_rows: list[tuple[dict[str, Any], float]] = []
    for chunk in chunks:
        chunk_copy = deepcopy(chunk)
        content = str(chunk_copy.get("content") or "")
        compressed_content, best_overlap = _compress_chunk_content(
            question_tokens=question_tokens,
            content=content,
            max_sentences=max(1, max_sentences),
        )
        chunk_copy["content"] = compressed_content
        compressed_rows.append((chunk_copy, best_overlap))

    if drop_irrelevant:
        positive_rows = [row for row in compressed_rows if row[1] > 0.0]
        if positive_rows:
            return [row[0] for row in positive_rows]
    return [row[0] for row in compressed_rows]


def _clear_proxy_env() -> None:
    for key in _PROXY_ENV_KEYS:
        os.environ[key] = ""


def _is_local_blackhole_proxy(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    host = str(parsed.hostname or "").strip().lower()
    port = parsed.port
    return host in _LOCAL_PROXY_HOSTS and int(port or 0) == 9


def _detect_blackhole_proxy_vars() -> list[str]:
    bad: list[str] = []
    for key in _PROXY_ENV_KEYS:
        value = str(os.getenv(key) or "").strip()
        if value and _is_local_blackhole_proxy(value):
            bad.append(key)
    return bad


def _import_deepeval() -> tuple[type[Any], dict[str, type[Any]]]:
    try:
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            ContextualRelevancyMetric,
            FaithfulnessMetric,
        )
        from deepeval.test_case import LLMTestCase
    except Exception as exc:  # pragma: no cover - dependency availability
        raise RuntimeError(
            "deepeval is required for this pipeline. "
            "Install dependencies with: venv\\Scripts\\pip install -r requirements.txt"
        ) from exc

    return LLMTestCase, {
        "answer_relevancy": AnswerRelevancyMetric,
        "faithfulness": FaithfulnessMetric,
        "contextual_relevancy": ContextualRelevancyMetric,
        "contextual_precision": ContextualPrecisionMetric,
        "contextual_recall": ContextualRecallMetric,
    }


def _metric_success(metric: Any, *, threshold: float, score: float | None) -> bool:
    success = getattr(metric, "success", None)
    if isinstance(success, bool):
        return success

    checker = getattr(metric, "is_successful", None)
    if callable(checker):
        try:
            maybe_success = checker()
            if isinstance(maybe_success, bool):
                return maybe_success
        except Exception:
            pass

    if score is None:
        return False
    return score >= threshold


def _measure_metric(
    *,
    metric_name: str,
    metric: Any,
    test_case: Any,
    threshold: float,
    timeout_fallback_enabled: bool,
    timeout_fallback_max_context_items: int,
    timeout_fallback_max_context_chars: int,
    timeout_fallback_max_answer_chars: int,
    heuristic_faithfulness_fallback: bool,
    answer: str,
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    started = time.perf_counter()
    error_message: str | None = None
    score: float | None = None
    reason: str | None = None
    passed = False
    fallback_mode: str | None = None

    try:
        metric.measure(test_case)
        raw_score = getattr(metric, "score", None)
        if isinstance(raw_score, (int, float)):
            score = float(raw_score)
        raw_reason = getattr(metric, "reason", None)
        if raw_reason is not None:
            reason = str(raw_reason)
        passed = _metric_success(metric, threshold=threshold, score=score)
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        low = error_message.lower()
        is_timeout = ("timeout" in low) or ("timed out" in low)
        if timeout_fallback_enabled and is_timeout:
            try:
                retry_case = _build_timeout_fallback_case(
                    test_case=test_case,
                    max_context_items=max(1, timeout_fallback_max_context_items),
                    max_context_chars=max(80, timeout_fallback_max_context_chars),
                    max_answer_chars=max(80, timeout_fallback_max_answer_chars),
                )
                metric.measure(retry_case)
                raw_score = getattr(metric, "score", None)
                if isinstance(raw_score, (int, float)):
                    score = float(raw_score)
                raw_reason = getattr(metric, "reason", None)
                if raw_reason is not None:
                    reason = str(raw_reason)
                passed = _metric_success(metric, threshold=threshold, score=score)
                error_message = None
                fallback_mode = "compact_retry"
            except Exception as retry_exc:
                error_message = (
                    f"{error_message} | fallback_failed: "
                    f"{type(retry_exc).__name__}: {retry_exc}"
                )
        if (
            metric_name == "faithfulness"
            and heuristic_faithfulness_fallback
            and error_message is not None
            and is_timeout
        ):
            score = _citation_validity_score(answer=answer, chunks=chunks)
            passed = score >= threshold
            reason = (
                "Heuristic faithfulness fallback used after metric timeout: "
                "score is citation-validity ratio against metric context."
            )
            error_message = None
            fallback_mode = "heuristic_citation_validity"

    return {
        "metric": metric_name,
        "score": score,
        "threshold": threshold,
        "passed": passed,
        "reason": reason,
        "error": error_message,
        "fallback": fallback_mode,
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
    }


def _new_metric(
    *,
    metric_name: str,
    metric_cls: type[Any],
    threshold: float,
    model: str,
    verbose_mode: bool,
    async_mode: bool,
    faithfulness_truths_extraction_limit: int | None,
) -> Any:
    kwargs: dict[str, Any] = {
        "threshold": threshold,
        "model": model,
        "include_reason": True,
        "verbose_mode": verbose_mode,
        "async_mode": async_mode,
    }
    if (
        metric_name == "faithfulness"
        and faithfulness_truths_extraction_limit is not None
        and faithfulness_truths_extraction_limit > 0
    ):
        kwargs["truths_extraction_limit"] = faithfulness_truths_extraction_limit
    return metric_cls(
        **kwargs,
    )


def _clip_text(text: str, max_chars: int) -> str:
    value = (text or "").strip()
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[: max(0, max_chars)].rstrip()


def _build_timeout_fallback_case(
    *,
    test_case: Any,
    max_context_items: int,
    max_context_chars: int,
    max_answer_chars: int,
) -> Any:
    payload = test_case.model_dump()
    retrieval_context = payload.get("retrieval_context") or []
    if not isinstance(retrieval_context, list):
        retrieval_context = [str(retrieval_context)]
    clipped_context = [
        _clip_text(str(item), max_context_chars)
        for item in retrieval_context[:max(1, max_context_items)]
        if str(item).strip()
    ]
    payload["retrieval_context"] = clipped_context or ["No retrieval context found."]
    payload["actual_output"] = _clip_text(str(payload.get("actual_output") or ""), max_answer_chars)
    return test_case.model_copy(update=payload)


def _apply_deepeval_runtime_env(
    *,
    retry_max_attempts: int,
    per_attempt_timeout_seconds: int,
    per_task_timeout_seconds: int,
) -> None:
    os.environ["DEEPEVAL_RETRY_MAX_ATTEMPTS"] = str(max(1, retry_max_attempts))
    os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = str(
        max(1, per_attempt_timeout_seconds)
    )
    os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"] = str(
        max(1, per_task_timeout_seconds)
    )


def _preflight_openai(client: Any) -> None:
    try:
        client.models.list()
    except Exception as exc:
        raise RuntimeError(f"OpenAI preflight failed: {type(exc).__name__}: {exc}") from exc


def _preflight_database(database_url: str) -> None:
    try:
        import psycopg2
    except Exception as exc:
        raise RuntimeError(
            "Database preflight failed: psycopg2 is unavailable in this environment."
        ) from exc

    try:
        with psycopg2.connect(database_url, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
    except Exception as exc:
        raise RuntimeError(
            f"Database preflight failed: {type(exc).__name__}: {exc}"
        ) from exc


def main() -> None:
    cfg = get_config()
    eval_cfg = cfg["eval_deepeval"]

    parser = argparse.ArgumentParser(
        description="Run RAG evaluation using DeepEval metrics."
    )
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
        help="Number of retrieved chunks per question.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=int(eval_cfg["max_questions"]),
        help="Maximum number of questions to evaluate.",
    )
    parser.add_argument(
        "--judge-model",
        default=str(eval_cfg["judge_model"]),
        help="Model used by DeepEval metrics as judge.",
    )
    parser.add_argument(
        "--include-unanswerable",
        action=argparse.BooleanOptionalAction,
        default=bool(eval_cfg["include_unanswerable"]),
        help="Include not-available / clarification-needed cases in DeepEval.",
    )
    parser.add_argument(
        "--use-reference-metrics",
        action=argparse.BooleanOptionalAction,
        default=bool(eval_cfg["use_reference_metrics"]),
        help=(
            "Enable contextual precision/recall metrics (requires expected_output "
            "in each question)."
        ),
    )
    parser.add_argument(
        "--answer-relevancy-threshold",
        type=float,
        default=float(eval_cfg["answer_relevancy_threshold"]),
        help="Passing threshold for DeepEval AnswerRelevancyMetric.",
    )
    parser.add_argument(
        "--faithfulness-threshold",
        type=float,
        default=float(eval_cfg["faithfulness_threshold"]),
        help="Passing threshold for DeepEval FaithfulnessMetric.",
    )
    parser.add_argument(
        "--contextual-relevancy-threshold",
        type=float,
        default=float(eval_cfg["contextual_relevancy_threshold"]),
        help="Passing threshold for DeepEval ContextualRelevancyMetric.",
    )
    parser.add_argument(
        "--contextual-precision-threshold",
        type=float,
        default=float(eval_cfg["contextual_precision_threshold"]),
        help="Passing threshold for DeepEval ContextualPrecisionMetric.",
    )
    parser.add_argument(
        "--contextual-recall-threshold",
        type=float,
        default=float(eval_cfg["contextual_recall_threshold"]),
        help="Passing threshold for DeepEval ContextualRecallMetric.",
    )
    parser.add_argument(
        "--verbose-metrics",
        action=argparse.BooleanOptionalAction,
        default=bool(eval_cfg["verbose_metrics"]),
        help="Enable DeepEval metric verbose mode.",
    )
    parser.add_argument(
        "--metric-async-mode",
        action=argparse.BooleanOptionalAction,
        default=bool(eval_cfg["metric_async_mode"]),
        help="Enable async execution inside DeepEval metric.measure().",
    )
    parser.add_argument(
        "--faithfulness-truths-extraction-limit",
        type=int,
        default=int(eval_cfg["faithfulness_truths_extraction_limit"]),
        help=(
            "Limit truth extraction for faithfulness metric to improve stability "
            "on long contexts."
        ),
    )
    parser.add_argument(
        "--deepeval-retry-max-attempts",
        type=int,
        default=int(eval_cfg["deepeval_retry_max_attempts"]),
        help="DeepEval provider retry attempts per call (including first attempt).",
    )
    parser.add_argument(
        "--deepeval-per-attempt-timeout-seconds",
        type=int,
        default=int(eval_cfg["deepeval_per_attempt_timeout_seconds"]),
        help="DeepEval per-attempt model timeout (seconds).",
    )
    parser.add_argument(
        "--deepeval-per-task-timeout-seconds",
        type=int,
        default=int(eval_cfg["deepeval_per_task_timeout_seconds"]),
        help="DeepEval outer per-metric timeout budget (seconds).",
    )
    parser.add_argument(
        "--timeout-fallback-enabled",
        action=argparse.BooleanOptionalAction,
        default=bool(eval_cfg["timeout_fallback_enabled"]),
        help=(
            "When metric timeout occurs, retry once with compacted answer/context "
            "to reduce transient timeout failures."
        ),
    )
    parser.add_argument(
        "--timeout-fallback-max-context-items",
        type=int,
        default=int(eval_cfg["timeout_fallback_max_context_items"]),
        help="Fallback retry: maximum retrieval_context items to keep.",
    )
    parser.add_argument(
        "--timeout-fallback-max-context-chars",
        type=int,
        default=int(eval_cfg["timeout_fallback_max_context_chars"]),
        help="Fallback retry: maximum characters per context item.",
    )
    parser.add_argument(
        "--timeout-fallback-max-answer-chars",
        type=int,
        default=int(eval_cfg["timeout_fallback_max_answer_chars"]),
        help="Fallback retry: maximum answer characters.",
    )
    parser.add_argument(
        "--compact-metric-answer",
        action=argparse.BooleanOptionalAction,
        default=bool(eval_cfg["compact_metric_answer"]),
        help=(
            "Compact answer text before metric scoring to emphasize direct answer "
            "content over verbose phrasing."
        ),
    )
    parser.add_argument(
        "--compact-metric-answer-max-sentences",
        type=int,
        default=int(eval_cfg["compact_metric_answer_max_sentences"]),
        help="Compacted metric answer: maximum kept sentences.",
    )
    parser.add_argument(
        "--compact-metric-answer-max-chars",
        type=int,
        default=int(eval_cfg["compact_metric_answer_max_chars"]),
        help="Compacted metric answer: maximum characters.",
    )
    parser.add_argument(
        "--rewrite-metric-answer",
        action=argparse.BooleanOptionalAction,
        default=bool(eval_cfg["rewrite_metric_answer"]),
        help=(
            "Rewrite metric answer into a strict 1-2 sentence faithful form before "
            "DeepEval scoring."
        ),
    )
    parser.add_argument(
        "--rewrite-metric-answer-model",
        default=str(eval_cfg["rewrite_metric_answer_model"]),
        help="Model used to rewrite metric answer text.",
    )
    parser.add_argument(
        "--rewrite-metric-answer-max-chars",
        type=int,
        default=int(eval_cfg["rewrite_metric_answer_max_chars"]),
        help="Maximum characters for rewritten metric answer.",
    )
    parser.add_argument(
        "--heuristic-faithfulness-fallback",
        action=argparse.BooleanOptionalAction,
        default=bool(eval_cfg["heuristic_faithfulness_fallback"]),
        help=(
            "If faithfulness metric times out even after compact retry, compute a "
            "citation-validity fallback score."
        ),
    )
    parser.add_argument(
        "--use-rerank",
        action=argparse.BooleanOptionalAction,
        default=bool(eval_cfg["use_rerank"]),
        help="Enable embedding rerank during retrieval for eval.",
    )
    parser.add_argument(
        "--use-sentence-window",
        action=argparse.BooleanOptionalAction,
        default=bool(eval_cfg["use_sentence_window"]),
        help="Enable sentence-window post processing during retrieval for eval.",
    )
    parser.add_argument(
        "--sentence-window-size",
        type=int,
        default=int(eval_cfg["sentence_window_size"]),
        help="Sentence window radius for retrieval sentence-window mode.",
    )
    parser.add_argument(
        "--use-context-compression",
        action=argparse.BooleanOptionalAction,
        default=bool(eval_cfg["use_context_compression"]),
        help=(
            "Compress retrieved chunk text to question-relevant sentences before "
            "answering and metric scoring."
        ),
    )
    parser.add_argument(
        "--compression-max-sentences",
        type=int,
        default=int(eval_cfg["compression_max_sentences"]),
        help="Max kept sentences per retrieved chunk after context compression.",
    )
    parser.add_argument(
        "--compression-drop-irrelevant",
        action=argparse.BooleanOptionalAction,
        default=bool(eval_cfg["compression_drop_irrelevant"]),
        help=(
            "Drop retrieved chunks with zero question-token overlap if at least "
            "one relevant chunk remains."
        ),
    )
    parser.add_argument(
        "--clear-proxy-env",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Clear HTTP(S)_PROXY/ALL_PROXY env vars in-process before calling "
            "OpenAI. Useful when local blackhole proxies are configured."
        ),
    )
    parser.add_argument(
        "--preflight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run quick OpenAI + database connectivity checks before evaluation.",
    )
    parser.add_argument(
        "--skip-metrics-on-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Skip DeepEval metric calls when retrieval/answer generation fails for "
            "a question."
        ),
    )
    args = parser.parse_args()

    load_dotenv()
    if bool(args.clear_proxy_env):
        _clear_proxy_env()

    bad_proxy_vars = _detect_blackhole_proxy_vars()
    if bad_proxy_vars:
        joined = ", ".join(sorted(set(bad_proxy_vars)))
        raise SystemExit(
            "Detected local blackhole proxy env vars "
            f"({joined}) pointing to 127.0.0.1:9. "
            "Unset them or rerun with --clear-proxy-env."
        )

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is required")

    questions_path = Path(args.questions_path)
    output_path = Path(args.output_path)
    top_k = max(1, int(args.top_k))
    max_questions = max(1, int(args.max_questions))
    judge_model = str(args.judge_model).strip()
    if not judge_model:
        raise ValueError("judge model must be a non-empty string")
    metric_async_mode = bool(args.metric_async_mode)
    faithfulness_truths_extraction_limit = int(args.faithfulness_truths_extraction_limit)
    deepeval_retry_max_attempts = max(1, int(args.deepeval_retry_max_attempts))
    deepeval_per_attempt_timeout_seconds = max(
        1, int(args.deepeval_per_attempt_timeout_seconds)
    )
    deepeval_per_task_timeout_seconds = max(1, int(args.deepeval_per_task_timeout_seconds))
    timeout_fallback_enabled = bool(args.timeout_fallback_enabled)
    timeout_fallback_max_context_items = max(
        1, int(args.timeout_fallback_max_context_items)
    )
    timeout_fallback_max_context_chars = max(
        80, int(args.timeout_fallback_max_context_chars)
    )
    timeout_fallback_max_answer_chars = max(80, int(args.timeout_fallback_max_answer_chars))
    compact_metric_answer = bool(args.compact_metric_answer)
    compact_metric_answer_max_sentences = max(
        1, int(args.compact_metric_answer_max_sentences)
    )
    compact_metric_answer_max_chars = max(80, int(args.compact_metric_answer_max_chars))
    rewrite_metric_answer = bool(args.rewrite_metric_answer)
    rewrite_metric_answer_model = str(args.rewrite_metric_answer_model).strip()
    if not rewrite_metric_answer_model:
        rewrite_metric_answer_model = judge_model
    rewrite_metric_answer_max_chars = max(80, int(args.rewrite_metric_answer_max_chars))
    heuristic_faithfulness_fallback = bool(args.heuristic_faithfulness_fallback)
    use_rerank = bool(args.use_rerank)
    use_sentence_window = bool(args.use_sentence_window)
    sentence_window_size = max(0, int(args.sentence_window_size))
    use_context_compression = bool(args.use_context_compression)
    compression_max_sentences = max(1, int(args.compression_max_sentences))
    compression_drop_irrelevant = bool(args.compression_drop_irrelevant)

    _apply_deepeval_runtime_env(
        retry_max_attempts=deepeval_retry_max_attempts,
        per_attempt_timeout_seconds=deepeval_per_attempt_timeout_seconds,
        per_task_timeout_seconds=deepeval_per_task_timeout_seconds,
    )

    try:
        LLMTestCase, metric_classes = _import_deepeval()
    except RuntimeError as exc:
        raise SystemExit(str(exc))

    questions = load_questions(questions_path, max_questions=max_questions)
    client = create_openai_client(api_key=openai_api_key)
    if bool(args.preflight):
        try:
            _preflight_openai(client)
            _preflight_database(database_url)
        except RuntimeError as exc:
            raise SystemExit(str(exc))

    thresholds: dict[str, float] = {
        "answer_relevancy": float(args.answer_relevancy_threshold),
        "faithfulness": float(args.faithfulness_threshold),
        "contextual_relevancy": float(args.contextual_relevancy_threshold),
        "contextual_precision": float(args.contextual_precision_threshold),
        "contextual_recall": float(args.contextual_recall_threshold),
    }
    metric_summary: dict[str, dict[str, float]] = {
        key: {"count": 0.0, "passed": 0.0, "score_total": 0.0}
        for key in _METRIC_PRINT_ORDER
    }

    results: list[dict[str, Any]] = []
    skipped_unanswerable = 0
    skipped_reference_cases = 0
    failures = 0

    for i, item in enumerate(questions, start=1):
        question = str(item["question"]).strip()
        category = str(item.get("category") or "uncategorized")
        expected_behavior = _expected_behavior(item)

        if (
            not bool(args.include_unanswerable)
            and expected_behavior in _UNANSWERABLE_BEHAVIORS
        ):
            skipped_unanswerable += 1
            print(f"[{i}/{len(questions)}] SKIP - expected_behavior={expected_behavior}")
            continue

        started = time.perf_counter()
        error_message: str | None = None
        chunks: list[dict[str, Any]] = []
        eval_chunks: list[dict[str, Any]] = []
        answer = ""

        try:
            chunks = retrieve_chunks(
                question,
                top_k=top_k,
                use_rerank=use_rerank,
                use_sentence_window=use_sentence_window,
                sentence_window_size=sentence_window_size,
            )
            if use_context_compression:
                eval_chunks = _compress_chunks_for_question(
                    question=question,
                    chunks=chunks,
                    max_sentences=compression_max_sentences,
                    drop_irrelevant=compression_drop_irrelevant,
                )
            else:
                eval_chunks = chunks
            answer = ask_with_context(question, eval_chunks, client)
        except Exception as exc:
            failures += 1
            error_message = f"{type(exc).__name__}: {exc}"
            answer = f"System failure during deepeval run. {error_message}"
            eval_chunks = chunks

        metric_chunks = _select_metric_chunks(answer=answer, chunks=eval_chunks)
        metric_answer = (
            _compact_answer_for_metrics(
                answer=answer,
                max_sentences=compact_metric_answer_max_sentences,
                max_chars=compact_metric_answer_max_chars,
            )
            if compact_metric_answer
            else answer
        )
        if rewrite_metric_answer and (not error_message):
            try:
                metric_answer = _rewrite_metric_answer(
                    question=question,
                    answer=metric_answer,
                    chunks=metric_chunks,
                    client=client,
                    model=rewrite_metric_answer_model,
                    max_chars=rewrite_metric_answer_max_chars,
                )
            except Exception:
                # Keep metric scoring resilient even if rewrite call fails.
                pass
        expected_output = _extract_expected_output(item)
        retrieval_context = _build_retrieval_context(metric_chunks)
        test_case_kwargs: dict[str, Any] = {
            "input": question,
            "actual_output": metric_answer,
            "retrieval_context": retrieval_context,
        }
        if expected_output:
            test_case_kwargs["expected_output"] = expected_output
        test_case = LLMTestCase(**test_case_kwargs)

        metric_results: dict[str, Any] = {}
        if error_message and bool(args.skip_metrics_on_failure):
            for metric_name in _METRIC_PRINT_ORDER:
                metric_results[metric_name] = {
                    "metric": metric_name,
                    "score": None,
                    "threshold": thresholds[metric_name],
                    "passed": None,
                    "reason": None,
                    "error": None,
                    "latency_ms": 0.0,
                    "skipped": True,
                    "skip_reason": "question_execution_failed",
                }
        else:
            for metric_name in _METRIC_PRINT_ORDER:
                if (
                    metric_name in _REFERENCE_METRIC_NAMES
                    and bool(args.use_reference_metrics)
                    and not expected_output
                ):
                    skipped_reference_cases += 1
                    metric_results[metric_name] = {
                        "metric": metric_name,
                        "score": None,
                        "threshold": thresholds[metric_name],
                        "passed": None,
                        "reason": None,
                        "error": None,
                        "latency_ms": 0.0,
                        "skipped": True,
                        "skip_reason": "expected_output missing",
                    }
                    continue

                if (
                    metric_name in _REFERENCE_METRIC_NAMES
                    and not bool(args.use_reference_metrics)
                ):
                    metric_results[metric_name] = {
                        "metric": metric_name,
                        "score": None,
                        "threshold": thresholds[metric_name],
                        "passed": None,
                        "reason": None,
                        "error": None,
                        "latency_ms": 0.0,
                        "skipped": True,
                        "skip_reason": "reference metrics disabled",
                    }
                    continue

                metric = _new_metric(
                    metric_name=metric_name,
                    metric_cls=metric_classes[metric_name],
                    threshold=thresholds[metric_name],
                    model=judge_model,
                    verbose_mode=bool(args.verbose_metrics),
                    async_mode=metric_async_mode,
                    faithfulness_truths_extraction_limit=faithfulness_truths_extraction_limit,
                )
                outcome = _measure_metric(
                    metric_name=metric_name,
                    metric=metric,
                    test_case=test_case,
                    threshold=thresholds[metric_name],
                    timeout_fallback_enabled=timeout_fallback_enabled,
                    timeout_fallback_max_context_items=timeout_fallback_max_context_items,
                    timeout_fallback_max_context_chars=timeout_fallback_max_context_chars,
                    timeout_fallback_max_answer_chars=timeout_fallback_max_answer_chars,
                    heuristic_faithfulness_fallback=heuristic_faithfulness_fallback,
                    answer=metric_answer,
                    chunks=metric_chunks,
                )
                metric_results[metric_name] = outcome

                metric_summary[metric_name]["count"] += 1
                if outcome["passed"] is True:
                    metric_summary[metric_name]["passed"] += 1
                if isinstance(outcome["score"], float):
                    metric_summary[metric_name]["score_total"] += outcome["score"]

        total_latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        results.append(
            {
                "question": question,
                "category": category,
                "expected_behavior": expected_behavior,
                "top_k": top_k,
                "expected_sections": item.get("expected_sections") or [],
                "expected_output": expected_output,
                "retrieval_mode": {
                    "use_rerank": use_rerank,
                    "use_sentence_window": use_sentence_window,
                    "sentence_window_size": sentence_window_size,
                    "use_context_compression": use_context_compression,
                    "compression_max_sentences": compression_max_sentences,
                    "compression_drop_irrelevant": compression_drop_irrelevant,
                    "deepeval_retry_max_attempts": deepeval_retry_max_attempts,
                    "deepeval_per_attempt_timeout_seconds": deepeval_per_attempt_timeout_seconds,
                    "deepeval_per_task_timeout_seconds": deepeval_per_task_timeout_seconds,
                    "timeout_fallback_enabled": timeout_fallback_enabled,
                    "compact_metric_answer": compact_metric_answer,
                    "compact_metric_answer_max_sentences": compact_metric_answer_max_sentences,
                    "compact_metric_answer_max_chars": compact_metric_answer_max_chars,
                    "rewrite_metric_answer": rewrite_metric_answer,
                    "rewrite_metric_answer_model": rewrite_metric_answer_model,
                    "rewrite_metric_answer_max_chars": rewrite_metric_answer_max_chars,
                    "heuristic_faithfulness_fallback": heuristic_faithfulness_fallback,
                },
                "retrieved_raw": [
                    {
                        "id": c.get("id"),
                        "distance": c.get("distance"),
                        "source": c.get("metadata", {}).get("source"),
                        "section": c.get("metadata", {}).get("section"),
                    }
                    for c in chunks
                ],
                "retrieved": [
                    {
                        "id": c.get("id"),
                        "distance": c.get("distance"),
                        "source": c.get("metadata", {}).get("source"),
                        "section": c.get("metadata", {}).get("section"),
                    }
                    for c in metric_chunks
                ],
                "retrieval_context": retrieval_context,
                "metric_actual_output": metric_answer,
                "answer": answer,
                "error": error_message,
                "metrics": metric_results,
                "latency_ms": total_latency_ms,
            }
        )

        case_failures = sum(
            1
            for details in metric_results.values()
            if isinstance(details, dict) and details.get("error")
        )
        if error_message:
            state = "FAIL"
        else:
            state = "OK" if case_failures == 0 else "METRIC_ERROR"
        print(
            f"[{i}/{len(questions)}] {state} - retrieved={len(eval_chunks)}/{len(chunks)} "
            f"category={category}"
        )

    write_jsonl(results, output_path)
    print(f"Saved deepeval results to: {output_path}")
    print(
        f"Questions evaluated: {len(results)}/{len(questions)} "
        f"(skipped-unanswerable={skipped_unanswerable})"
    )
    if bool(args.use_reference_metrics):
        print(f"Reference-metric skips (missing expected_output): {skipped_reference_cases}")
    print(f"System failures: {failures}/{len(results)} ({pct(failures, len(results)):.1f}%)")

    print("\nDeepEval metric summary:")
    for metric_name in _METRIC_PRINT_ORDER:
        summary = metric_summary[metric_name]
        checks = int(summary["count"])
        passed = int(summary["passed"])
        avg_score = (summary["score_total"] / checks) if checks else 0.0
        print(
            f"- {metric_name}: pass {passed}/{checks} "
            f"({pct(passed, checks):.1f}%), avg_score={avg_score:.3f}"
        )


if __name__ == "__main__":
    main()
