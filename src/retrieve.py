from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv
from openai import OpenAI

from config import get_config
from retrieval.auto_merging_retriever import auto_merge_chunks
from retrieval.chunk_retriever import fetch_candidate_chunks
from retrieval.llm_rerank_retriever import llm_rerank_chunks
from retrieval.rerank_retriever import rerank_chunks
from retrieval.sentence_window_retriever import sentence_window_chunks


def retrieve_chunks(
    question: str,
    *,
    top_k: int | None = None,
    candidate_k: int | None = None,
    use_rerank: bool | None = None,
    use_llm_rerank: bool | None = None,
    llm_rerank_candidate_k: int | None = None,
    llm_rerank_keep_k: int | None = None,
    use_auto_merging: bool | None = None,
    auto_merge_max_gap: int | None = None,
    auto_merge_max_chunks: int | None = None,
    use_sentence_window: bool | None = None,
    sentence_window_size: int | None = None,
) -> list[dict]:
    load_dotenv()

    database_url = os.getenv("DATABASE_URL")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not database_url:
        raise ValueError("DATABASE_URL is required")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required")

    cfg = get_config()
    retrieval_cfg = cfg["retrieval"]
    models_cfg = cfg["models"]

    top_k = int(top_k if top_k is not None else retrieval_cfg["top_k"])
    candidate_k = int(
        candidate_k if candidate_k is not None else retrieval_cfg["candidate_k"]
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
    fetch_k = max(
        top_k,
        candidate_k,
        llm_rerank_candidate_k if use_llm_rerank else 0,
    )

    client = OpenAI(api_key=openai_api_key)
    results = fetch_candidate_chunks(
        question,
        client=client,
        database_url=database_url,
        table_name=table_name,
        embedding_model=embedding_model,
        fetch_k=fetch_k,
    )

    if use_rerank and len(results) > top_k:
        try:
            results = rerank_chunks(
                question,
                results,
                client=client,
                model=rerank_model,
            )
        except Exception:
            # Retrieval should still work even if reranking fails.
            pass

    if use_llm_rerank and len(results) > 1:
        try:
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
            )
            results = reranked_candidates[:keep_limit] + results[candidate_limit:]
        except Exception:
            # Retrieval should still work even if LLM reranking fails.
            pass

    if use_auto_merging and results:
        try:
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
            results = sentence_window_chunks(
                question,
                results,
                client=client,
                model=rerank_model,
                window_size=max(0, sentence_window_size),
            )
        except Exception:
            # Retrieval should still work even if sentence-window selection fails.
            pass

    return results[:top_k]


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
            f"{rerank_text}{llm_rerank_text}{sentence_window_text}{auto_merge_text}"
        )


if __name__ == "__main__":
    main()
