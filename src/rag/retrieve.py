from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Any

from dotenv import load_dotenv

from core.config import get_config
from core.openai_client import create_openai_client
from core.telemetry import get_tracer, setup_telemetry
from rag.retrieval.auto_merging_retriever import auto_merge_chunks
from rag.retrieval.chunk_retriever import (
    fetch_candidate_chunks,
    fetch_hybrid_candidate_chunks,
)
from rag.retrieval.llm_rerank_retriever import llm_rerank_chunks
from rag.retrieval.redis_cache import get_json, set_json
from rag.retrieval.rerank_retriever import rerank_chunks
from rag.retrieval.sentence_window_retriever import sentence_window_chunks

tracer = get_tracer(__name__)


def _normalize_question(question: str) -> str:
    return " ".join(question.strip().split())


def _retrieval_cache_key(
    *,
    namespace: str,
    version: str,
    question: str,
    top_k: int,
    candidate_k: int,
    use_hybrid_search: bool,
    keyword_candidate_k: int,
    hybrid_alpha: float,
    hybrid_rrf_k: int,
    use_rerank: bool,
    use_llm_rerank: bool,
    llm_rerank_candidate_k: int,
    llm_rerank_keep_k: int,
    use_auto_merging: bool,
    auto_merge_max_gap: int,
    auto_merge_max_chunks: int,
    use_sentence_window: bool,
    sentence_window_size: int,
    embedding_model: str,
    rerank_model: str,
    llm_rerank_model: str,
    table_name: str,
) -> str:
    payload = {
        "version": version,
        "question": _normalize_question(question),
        "top_k": top_k,
        "candidate_k": candidate_k,
        "use_hybrid_search": use_hybrid_search,
        "keyword_candidate_k": keyword_candidate_k,
        "hybrid_alpha": round(float(hybrid_alpha), 6),
        "hybrid_rrf_k": hybrid_rrf_k,
        "use_rerank": use_rerank,
        "use_llm_rerank": use_llm_rerank,
        "llm_rerank_candidate_k": llm_rerank_candidate_k,
        "llm_rerank_keep_k": llm_rerank_keep_k,
        "use_auto_merging": use_auto_merging,
        "auto_merge_max_gap": auto_merge_max_gap,
        "auto_merge_max_chunks": auto_merge_max_chunks,
        "use_sentence_window": use_sentence_window,
        "sentence_window_size": sentence_window_size,
        "embedding_model": embedding_model,
        "rerank_model": rerank_model,
        "llm_rerank_model": llm_rerank_model,
        "table_name": table_name,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"{namespace}:retrieval:{digest}"


def retrieve_chunks(
    question: str,
    *,
    top_k: int | None = None,
    candidate_k: int | None = None,
    use_hybrid_search: bool | None = None,
    keyword_candidate_k: int | None = None,
    hybrid_alpha: float | None = None,
    hybrid_rrf_k: int | None = None,
    use_rerank: bool | None = None,
    use_llm_rerank: bool | None = None,
    llm_rerank_candidate_k: int | None = None,
    llm_rerank_keep_k: int | None = None,
    use_auto_merging: bool | None = None,
    auto_merge_max_gap: int | None = None,
    auto_merge_max_chunks: int | None = None,
    use_sentence_window: bool | None = None,
    sentence_window_size: int | None = None,
    openai_request_kwargs: dict[str, Any] | None = None,
    return_meta: bool = False,
) -> list[dict] | tuple[list[dict], dict[str, Any]]:
    load_dotenv()
    setup_telemetry()

    database_url = os.getenv("DATABASE_URL")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not database_url:
        raise ValueError("DATABASE_URL is required")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required")

    cfg = get_config()
    retrieval_cfg = cfg["retrieval"]
    models_cfg = cfg["models"]
    cache_cfg = cfg["cache"]

    top_k = int(top_k if top_k is not None else retrieval_cfg["top_k"])
    candidate_k = int(
        candidate_k if candidate_k is not None else retrieval_cfg["candidate_k"]
    )
    use_hybrid_search = (
        bool(use_hybrid_search)
        if use_hybrid_search is not None
        else bool(retrieval_cfg["use_hybrid_search"])
    )
    keyword_candidate_k = int(
        keyword_candidate_k
        if keyword_candidate_k is not None
        else retrieval_cfg["keyword_candidate_k"]
    )
    hybrid_alpha = float(
        hybrid_alpha if hybrid_alpha is not None else retrieval_cfg["hybrid_alpha"]
    )
    hybrid_rrf_k = int(
        hybrid_rrf_k if hybrid_rrf_k is not None else retrieval_cfg["hybrid_rrf_k"]
    )
    use_rerank = (
        bool(use_rerank) if use_rerank is not None else bool(retrieval_cfg["use_rerank"])
    )
    use_llm_rerank = (
        bool(use_llm_rerank)
        if use_llm_rerank is not None
        else bool(retrieval_cfg["use_llm_rerank"])
    )
    llm_rerank_candidate_k = int(
        llm_rerank_candidate_k
        if llm_rerank_candidate_k is not None
        else retrieval_cfg["llm_rerank_candidate_k"]
    )
    llm_rerank_keep_k = int(
        llm_rerank_keep_k
        if llm_rerank_keep_k is not None
        else retrieval_cfg["llm_rerank_keep_k"]
    )
    use_auto_merging = (
        bool(use_auto_merging)
        if use_auto_merging is not None
        else bool(retrieval_cfg["use_auto_merging"])
    )
    auto_merge_max_gap = int(
        auto_merge_max_gap
        if auto_merge_max_gap is not None
        else retrieval_cfg["auto_merge_max_gap"]
    )
    auto_merge_max_chunks = int(
        auto_merge_max_chunks
        if auto_merge_max_chunks is not None
        else retrieval_cfg["auto_merge_max_chunks"]
    )
    use_sentence_window = (
        bool(use_sentence_window)
        if use_sentence_window is not None
        else bool(retrieval_cfg["use_sentence_window"])
    )
    sentence_window_size = int(
        sentence_window_size
        if sentence_window_size is not None
        else retrieval_cfg["sentence_window_size"]
    )

    embedding_model = str(models_cfg["embedding_model"])
    rerank_model = str(models_cfg["rerank_model"])
    llm_rerank_model = str(models_cfg["llm_rerank_model"])
    table_name = str(retrieval_cfg["table_name"])
    cache_enabled = bool(cache_cfg["enabled"]) and str(cache_cfg["backend"]).lower() == "redis"
    cache_redis_url = os.getenv("REDIS_URL", str(cache_cfg["redis_url"]))
    cache_embedding_ttl_seconds = int(cache_cfg["embedding_ttl_seconds"])
    retrieval_cache_enabled = cache_enabled and bool(cache_cfg["retrieval_enabled"])
    retrieval_cache_ttl_seconds = int(cache_cfg["retrieval_ttl_seconds"])
    retrieval_cache_version = str(cache_cfg["retrieval_version"])
    cache_namespace = str(cache_cfg["key_prefix"])
    fetch_k = max(
        top_k,
        candidate_k,
        keyword_candidate_k if use_hybrid_search else 0,
        llm_rerank_candidate_k if use_llm_rerank else 0,
    )
    retrieval_cache_key = _retrieval_cache_key(
        namespace=cache_namespace,
        version=retrieval_cache_version,
        question=question,
        top_k=top_k,
        candidate_k=candidate_k,
        use_hybrid_search=use_hybrid_search,
        keyword_candidate_k=keyword_candidate_k,
        hybrid_alpha=hybrid_alpha,
        hybrid_rrf_k=hybrid_rrf_k,
        use_rerank=use_rerank,
        use_llm_rerank=use_llm_rerank,
        llm_rerank_candidate_k=llm_rerank_candidate_k,
        llm_rerank_keep_k=llm_rerank_keep_k,
        use_auto_merging=use_auto_merging,
        auto_merge_max_gap=auto_merge_max_gap,
        auto_merge_max_chunks=auto_merge_max_chunks,
        use_sentence_window=use_sentence_window,
        sentence_window_size=sentence_window_size,
        embedding_model=embedding_model,
        rerank_model=rerank_model,
        llm_rerank_model=llm_rerank_model,
        table_name=table_name,
    )
    retrieval_meta: dict[str, Any] = {
        "retrieval_cache_enabled": retrieval_cache_enabled,
        "retrieval_cache_hit": None if not retrieval_cache_enabled else False,
    }
    if retrieval_cache_enabled:
        with tracer.start_as_current_span("rag.retrieval_cache_lookup"):
            cached_results = get_json(cache_redis_url, retrieval_cache_key)
        if isinstance(cached_results, list):
            retrieval_meta["retrieval_cache_hit"] = True
            if return_meta:
                return cached_results, retrieval_meta
            return cached_results

    client = create_openai_client(api_key=openai_api_key)
    if use_hybrid_search:
        with tracer.start_as_current_span("rag.fetch_hybrid_candidates"):
            results = fetch_hybrid_candidate_chunks(
                question,
                client=client,
                database_url=database_url,
                table_name=table_name,
                embedding_model=embedding_model,
                vector_fetch_k=max(1, fetch_k),
                keyword_fetch_k=max(1, max(fetch_k, keyword_candidate_k)),
                alpha=max(0.0, min(1.0, hybrid_alpha)),
                rrf_k=max(1, hybrid_rrf_k),
                limit=max(1, fetch_k),
                embedding_cache_enabled=cache_enabled,
                embedding_cache_redis_url=cache_redis_url,
                embedding_cache_ttl_seconds=max(0, cache_embedding_ttl_seconds),
                embedding_cache_namespace=cache_namespace,
                openai_request_kwargs=openai_request_kwargs,
            )
    else:
        with tracer.start_as_current_span("rag.fetch_vector_candidates"):
            results = fetch_candidate_chunks(
                question,
                client=client,
                database_url=database_url,
                table_name=table_name,
                embedding_model=embedding_model,
                fetch_k=fetch_k,
                embedding_cache_enabled=cache_enabled,
                embedding_cache_redis_url=cache_redis_url,
                embedding_cache_ttl_seconds=max(0, cache_embedding_ttl_seconds),
                embedding_cache_namespace=cache_namespace,
                openai_request_kwargs=openai_request_kwargs,
            )

    if use_rerank and len(results) > top_k:
        try:
            with tracer.start_as_current_span("rag.embedding_rerank"):
                results = rerank_chunks(
                    question,
                    results,
                    client=client,
                    model=rerank_model,
                    openai_request_kwargs=openai_request_kwargs,
                )
        except Exception:
            # Retrieval should still work even if reranking fails.
            pass

    if use_llm_rerank and len(results) > 1:
        try:
            with tracer.start_as_current_span("rag.llm_rerank"):
                candidate_limit = min(
                    max(1, llm_rerank_candidate_k),
                    len(results),
                )
                keep_limit = min(
                    max(1, llm_rerank_keep_k),
                    candidate_limit,
                )
                llm_candidates = results[:candidate_limit]
                reranked_candidates = llm_rerank_chunks(
                    question,
                    llm_candidates,
                    client=client,
                    model=llm_rerank_model,
                    openai_request_kwargs=openai_request_kwargs,
                )
                results = reranked_candidates[:keep_limit] + results[candidate_limit:]
        except Exception:
            # Retrieval should still work even if LLM reranking fails.
            pass

    if use_auto_merging and results:
        try:
            with tracer.start_as_current_span("rag.auto_merge"):
                results = auto_merge_chunks(
                    results,
                    max_gap=max(0, auto_merge_max_gap),
                    max_merged_chunks=max(1, auto_merge_max_chunks),
                )
        except Exception:
            # Retrieval should still work even if auto-merging fails.
            pass

    if use_sentence_window and results:
        try:
            with tracer.start_as_current_span("rag.sentence_window"):
                results = sentence_window_chunks(
                    question,
                    results,
                    client=client,
                    model=rerank_model,
                    window_size=max(0, sentence_window_size),
                    openai_request_kwargs=openai_request_kwargs,
                )
        except Exception:
            # Retrieval should still work even if sentence-window selection fails.
            pass

    final_results = results[:top_k]
    if retrieval_cache_enabled:
        with tracer.start_as_current_span("rag.retrieval_cache_store"):
            set_json(
                cache_redis_url,
                retrieval_cache_key,
                final_results,
                ttl_seconds=max(0, retrieval_cache_ttl_seconds),
            )
    if return_meta:
        return final_results, retrieval_meta
    return final_results


def main() -> None:
    cfg = get_config()
    retrieval_cfg = cfg["retrieval"]
    answer_cfg = cfg["answer"]

    parser = argparse.ArgumentParser(description="Retrieve policy chunks for a question.")
    parser.add_argument("question", nargs="*", help="Question text.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(retrieval_cfg["top_k"]),
        help="Final number of chunks.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=int(retrieval_cfg["candidate_k"]),
        help="Number of initial candidates before optional post-processing.",
    )
    parser.add_argument(
        "--use-hybrid-search",
        action=argparse.BooleanOptionalAction,
        default=bool(retrieval_cfg["use_hybrid_search"]),
        help="Enable or disable hybrid search (vector + keyword RRF).",
    )
    parser.add_argument(
        "--keyword-candidate-k",
        type=int,
        default=int(retrieval_cfg["keyword_candidate_k"]),
        help="Keyword candidate count used by hybrid search.",
    )
    parser.add_argument(
        "--hybrid-alpha",
        type=float,
        default=float(retrieval_cfg["hybrid_alpha"]),
        help="Hybrid fusion weight for vector branch (0..1).",
    )
    parser.add_argument(
        "--hybrid-rrf-k",
        type=int,
        default=int(retrieval_cfg["hybrid_rrf_k"]),
        help="RRF damping constant for hybrid fusion.",
    )
    parser.add_argument(
        "--use-rerank",
        action=argparse.BooleanOptionalAction,
        default=bool(retrieval_cfg["use_rerank"]),
        help="Enable or disable reranking.",
    )
    parser.add_argument(
        "--use-llm-rerank",
        action=argparse.BooleanOptionalAction,
        default=bool(retrieval_cfg["use_llm_rerank"]),
        help="Enable or disable LLM reranking on top candidates.",
    )
    parser.add_argument(
        "--llm-rerank-candidate-k",
        type=int,
        default=int(retrieval_cfg["llm_rerank_candidate_k"]),
        help="How many top candidates to send to LLM reranker.",
    )
    parser.add_argument(
        "--llm-rerank-keep-k",
        type=int,
        default=int(retrieval_cfg["llm_rerank_keep_k"]),
        help="How many LLM-reranked chunks to keep.",
    )
    parser.add_argument(
        "--use-auto-merging",
        action=argparse.BooleanOptionalAction,
        default=bool(retrieval_cfg["use_auto_merging"]),
        help="Enable or disable auto-merging.",
    )
    parser.add_argument(
        "--auto-merge-max-gap",
        type=int,
        default=int(retrieval_cfg["auto_merge_max_gap"]),
        help="Maximum index gap allowed when merging adjacent chunks.",
    )
    parser.add_argument(
        "--auto-merge-max-chunks",
        type=int,
        default=int(retrieval_cfg["auto_merge_max_chunks"]),
        help="Maximum number of chunks to merge into one group.",
    )
    parser.add_argument(
        "--use-sentence-window",
        action=argparse.BooleanOptionalAction,
        default=bool(retrieval_cfg["use_sentence_window"]),
        help="Enable or disable sentence-window post-processing.",
    )
    parser.add_argument(
        "--sentence-window-size",
        type=int,
        default=int(retrieval_cfg["sentence_window_size"]),
        help="Sentence window radius around the best sentence.",
    )
    args = parser.parse_args()

    default_question = str(answer_cfg["default_question"])
    question = " ".join(args.question).strip() if args.question else default_question
    results = retrieve_chunks(
        question,
        top_k=max(1, args.top_k),
        candidate_k=max(1, args.candidate_k),
        use_hybrid_search=args.use_hybrid_search,
        keyword_candidate_k=max(1, args.keyword_candidate_k),
        hybrid_alpha=max(0.0, min(1.0, args.hybrid_alpha)),
        hybrid_rrf_k=max(1, args.hybrid_rrf_k),
        use_rerank=args.use_rerank,
        use_llm_rerank=args.use_llm_rerank,
        llm_rerank_candidate_k=max(1, args.llm_rerank_candidate_k),
        llm_rerank_keep_k=max(1, args.llm_rerank_keep_k),
        use_auto_merging=args.use_auto_merging,
        auto_merge_max_gap=max(0, args.auto_merge_max_gap),
        auto_merge_max_chunks=max(1, args.auto_merge_max_chunks),
        use_sentence_window=args.use_sentence_window,
        sentence_window_size=max(0, args.sentence_window_size),
    )

    print(f"Question: {question}")
    print(f"Retrieved: {len(results)} chunks")
    for i, item in enumerate(results, start=1):
        meta = item["metadata"]
        rerank_text = (
            f" rerank={item['rerank_score']:.4f}"
            if "rerank_score" in item
            else ""
        )
        llm_rerank_text = (
            f" llm_rank={item['llm_rerank_rank']}"
            if "llm_rerank_rank" in item
            else ""
        )
        hybrid_text = (
            f" hybrid={item['hybrid_score']:.6f}"
            if "hybrid_score" in item
            else ""
        )
        sentence_window_text = (
            f" sw={item['sentence_window_score']:.4f}"
            if "sentence_window_score" in item
            else ""
        )
        auto_merge_text = (
            f" merged={item['merged_from_count']}"
            if item.get("auto_merged")
            else ""
        )
        print(
            f"[{i}] id={item['id']} distance={item['distance']:.4f} "
            f"source={meta['source']} section={meta['section']}"
            f"{rerank_text}{llm_rerank_text}{hybrid_text}"
            f"{sentence_window_text}{auto_merge_text}"
        )


if __name__ == "__main__":
    main()

