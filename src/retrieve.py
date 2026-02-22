from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from retrieval.auto_merging_retriever import auto_merge_chunks
from retrieval.chunk_retriever import fetch_candidate_chunks
from retrieval.rerank_retriever import rerank_chunks
from retrieval.sentence_window_retriever import sentence_window_chunks


def retrieve_chunks(
    question: str,
    *,
    top_k: int = 4,
    candidate_k: int = 12,
    use_rerank: bool = False,
    use_auto_merging: bool = False,
    auto_merge_max_gap: int = 1,
    auto_merge_max_chunks: int = 3,
    use_sentence_window: bool = False,
    sentence_window_size: int = 1,
) -> list[dict]:
    load_dotenv()

    database_url = os.getenv("DATABASE_URL")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not database_url:
        raise ValueError("DATABASE_URL is required")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required")

    embedding_model = "text-embedding-3-small"
    rerank_model = "text-embedding-3-large"
    table_name = "policy_chunks"
    fetch_k = max(top_k, candidate_k)

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
    default_question = "What illnesses are covered by this policy?"
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else default_question
    results = retrieve_chunks(question, top_k=4)

    print(f"Question: {question}")
    print(f"Retrieved: {len(results)} chunks")
    for i, item in enumerate(results, start=1):
        meta = item["metadata"]
        rerank_text = (
            f" rerank={item['rerank_score']:.4f}"
            if "rerank_score" in item
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
            f"{rerank_text}{sentence_window_text}{auto_merge_text}"
        )


if __name__ == "__main__":
    main()
