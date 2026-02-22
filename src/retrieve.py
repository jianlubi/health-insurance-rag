from __future__ import annotations

import math
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
import psycopg2


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in values) + "]"


def embed_query(question: str, client: OpenAI, model: str) -> list[float]:
    resp = client.embeddings.create(model=model, input=question)
    return resp.data[0].embedding


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def rerank_chunks(
    question: str,
    chunks: list[dict],
    *,
    client: OpenAI,
    model: str,
) -> list[dict]:
    if not chunks:
        return []

    resp = client.embeddings.create(
        model=model,
        input=[question, *[chunk["content"] for chunk in chunks]],
    )
    vectors = [item.embedding for item in resp.data]
    query_vector = vectors[0]
    chunk_vectors = vectors[1:]

    reranked: list[dict] = []
    for chunk, vector in zip(chunks, chunk_vectors, strict=False):
        item = dict(chunk)
        item["rerank_score"] = _cosine_similarity(query_vector, vector)
        reranked.append(item)

    reranked.sort(key=lambda c: c["rerank_score"], reverse=True)
    return reranked


def retrieve_chunks(
    question: str,
    *,
    top_k: int = 4,
    candidate_k: int = 12,
    use_rerank: bool = True,
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

    client = OpenAI(api_key=openai_api_key)
    query_vector = embed_query(question, client, embedding_model)
    vector_text = vector_literal(query_vector)
    fetch_k = max(top_k, candidate_k)

    with psycopg2.connect(database_url) as conn:
        with conn.cursor() as cur:
            # With very small datasets, low ivfflat probes can miss nearest rows.
            cur.execute("SET ivfflat.probes = 100;")
            cur.execute(
                f"""
                SELECT
                    id,
                    content,
                    source,
                    section,
                    chunk_index,
                    token_count,
                    embedding <=> %s::vector AS distance
                FROM {table_name}
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (vector_text, vector_text, fetch_k),
            )
            rows = cur.fetchall()

    results: list[dict] = []
    for rank, row in enumerate(rows, start=1):
        results.append(
            {
                "id": row[0],
                "content": row[1],
                "metadata": {
                    "source": row[2],
                    "section": row[3],
                    "index": row[4],
                    "token_count": row[5],
                },
                "distance": float(row[6]),
                "initial_rank": rank,
            }
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
        print(
            f"[{i}] id={item['id']} distance={item['distance']:.4f} "
            f"source={meta['source']} section={meta['section']}{rerank_text}"
        )


if __name__ == "__main__":
    main()
