from __future__ import annotations

import math
from typing import Any


def cosine_similarity(a: list[float], b: list[float]) -> float:
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
    client: Any,
    model: str,
    openai_request_kwargs: dict[str, Any] | None = None,
) -> list[dict]:
    if not chunks:
        return []

    resp = client.embeddings.create(
        model=model,
        input=[question, *[chunk["content"] for chunk in chunks]],
        **(openai_request_kwargs or {}),
    )
    vectors = [item.embedding for item in resp.data]
    query_vector = vectors[0]
    chunk_vectors = vectors[1:]

    reranked: list[dict] = []
    for chunk, vector in zip(chunks, chunk_vectors, strict=False):
        item = dict(chunk)
        item["rerank_score"] = cosine_similarity(query_vector, vector)
        reranked.append(item)

    reranked.sort(key=lambda c: c["rerank_score"], reverse=True)
    return reranked
