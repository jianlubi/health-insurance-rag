from __future__ import annotations

import hashlib
import psycopg2
from openai import OpenAI

from retrieval.redis_cache import get_json, set_json


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in values) + "]"


def _normalize_query(question: str) -> str:
    return " ".join(question.strip().split())


def _embedding_cache_key(
    *,
    namespace: str,
    model: str,
    question: str,
) -> str:
    normalized = _normalize_query(question)
    digest = hashlib.sha256(f"{model}\n{normalized}".encode("utf-8")).hexdigest()
    return f"{namespace}:query_embedding:{digest}"


def _get_cached_embedding(
    *,
    enabled: bool,
    redis_url: str,
    cache_key: str,
) -> list[float] | None:
    if not enabled:
        return None
    payload = get_json(redis_url, cache_key)
    if not isinstance(payload, list):
        return None
    try:
        return [float(x) for x in payload]
    except Exception:
        return None


def _set_cached_embedding(
    *,
    enabled: bool,
    redis_url: str,
    cache_key: str,
    embedding: list[float],
    ttl_seconds: int,
) -> None:
    if not enabled:
        return
    set_json(
        redis_url,
        cache_key,
        embedding,
        ttl_seconds=max(0, int(ttl_seconds)),
    )


def embed_query(
    question: str,
    client: OpenAI,
    model: str,
    *,
    cache_enabled: bool,
    cache_redis_url: str,
    cache_ttl_seconds: int,
    cache_namespace: str,
) -> list[float]:
    cache_key = _embedding_cache_key(
        namespace=cache_namespace,
        model=model,
        question=question,
    )
    cached = _get_cached_embedding(
        enabled=cache_enabled,
        redis_url=cache_redis_url,
        cache_key=cache_key,
    )
    if cached is not None:
        return cached

    resp = client.embeddings.create(model=model, input=question)
    embedding = resp.data[0].embedding
    _set_cached_embedding(
        enabled=cache_enabled,
        redis_url=cache_redis_url,
        cache_key=cache_key,
        embedding=embedding,
        ttl_seconds=max(0, int(cache_ttl_seconds)),
    )
    return embedding


def fetch_vector_candidate_chunks(
    question: str,
    *,
    client: OpenAI,
    database_url: str,
    table_name: str,
    embedding_model: str,
    fetch_k: int,
    embedding_cache_enabled: bool = False,
    embedding_cache_redis_url: str = "redis://127.0.0.1:6379/0",
    embedding_cache_ttl_seconds: int = 86400,
    embedding_cache_namespace: str = "health_rag",
) -> list[dict]:
    query_vector = embed_query(
        question,
        client,
        embedding_model,
        cache_enabled=embedding_cache_enabled,
        cache_redis_url=embedding_cache_redis_url,
        cache_ttl_seconds=embedding_cache_ttl_seconds,
        cache_namespace=embedding_cache_namespace,
    )
    vector_text = vector_literal(query_vector)

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

    return results


def fetch_keyword_candidate_chunks(
    question: str,
    *,
    database_url: str,
    table_name: str,
    fetch_k: int,
) -> list[dict]:
    with psycopg2.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                WITH query AS (
                    SELECT plainto_tsquery('english', %s) AS q
                )
                SELECT
                    id,
                    content,
                    source,
                    section,
                    chunk_index,
                    token_count,
                    ts_rank_cd(
                        to_tsvector('english', content),
                        (SELECT q FROM query)
                    ) AS keyword_score
                FROM {table_name}
                WHERE to_tsvector('english', content) @@ (SELECT q FROM query)
                ORDER BY keyword_score DESC
                LIMIT %s;
                """,
                (question, fetch_k),
            )
            rows = cur.fetchall()

    results: list[dict] = []
    for rank, row in enumerate(rows, start=1):
        keyword_score = float(row[6])
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
                # Keep the response shape stable for downstream consumers.
                "distance": 1.0 / (1.0 + max(0.0, keyword_score)),
                "initial_rank": rank,
                "keyword_rank": rank,
            }
        )
    return results


def _clamp(value: float, *, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def fuse_by_reciprocal_rank(
    vector_results: list[dict],
    keyword_results: list[dict],
    *,
    alpha: float,
    rrf_k: int,
    limit: int,
) -> list[dict]:
    alpha = _clamp(alpha, lo=0.0, hi=1.0)
    rrf_k = max(1, int(rrf_k))
    limit = max(1, int(limit))

    merged: dict[str, dict] = {}

    for rank, item in enumerate(vector_results, start=1):
        chunk = dict(item)
        chunk["vector_rank"] = rank
        chunk["_vector_rrf"] = 1.0 / (rrf_k + rank)
        chunk.setdefault("_keyword_rrf", 0.0)
        merged[str(chunk["id"])] = chunk

    for rank, item in enumerate(keyword_results, start=1):
        key = str(item["id"])
        if key in merged:
            chunk = merged[key]
        else:
            chunk = dict(item)
            chunk.setdefault("distance", 1.0)
            chunk["_vector_rrf"] = 0.0
            merged[key] = chunk

        chunk["keyword_rank"] = rank
        chunk["_keyword_rrf"] = 1.0 / (rrf_k + rank)
        if "initial_rank" in chunk:
            chunk["initial_rank"] = min(int(chunk["initial_rank"]), rank)
        else:
            chunk["initial_rank"] = rank

    fused = list(merged.values())
    for chunk in fused:
        vector_rrf = float(chunk.pop("_vector_rrf", 0.0))
        keyword_rrf = float(chunk.pop("_keyword_rrf", 0.0))
        chunk["hybrid_score"] = (alpha * vector_rrf) + ((1.0 - alpha) * keyword_rrf)

    fused.sort(
        key=lambda c: (
            -float(c.get("hybrid_score", 0.0)),
            int(c.get("initial_rank", 10**9)),
            float(c.get("distance", 1.0)),
        )
    )
    return fused[:limit]


def fetch_hybrid_candidate_chunks(
    question: str,
    *,
    client: OpenAI,
    database_url: str,
    table_name: str,
    embedding_model: str,
    vector_fetch_k: int,
    keyword_fetch_k: int,
    alpha: float,
    rrf_k: int,
    limit: int,
    embedding_cache_enabled: bool = False,
    embedding_cache_redis_url: str = "redis://127.0.0.1:6379/0",
    embedding_cache_ttl_seconds: int = 86400,
    embedding_cache_namespace: str = "health_rag",
) -> list[dict]:
    vector_results = fetch_vector_candidate_chunks(
        question,
        client=client,
        database_url=database_url,
        table_name=table_name,
        embedding_model=embedding_model,
        fetch_k=max(1, vector_fetch_k),
        embedding_cache_enabled=embedding_cache_enabled,
        embedding_cache_redis_url=embedding_cache_redis_url,
        embedding_cache_ttl_seconds=embedding_cache_ttl_seconds,
        embedding_cache_namespace=embedding_cache_namespace,
    )
    try:
        keyword_results = fetch_keyword_candidate_chunks(
            question,
            database_url=database_url,
            table_name=table_name,
            fetch_k=max(1, keyword_fetch_k),
        )
    except Exception:
        keyword_results = []
    if not keyword_results:
        return vector_results[: max(1, limit)]
    if not vector_results:
        return keyword_results[: max(1, limit)]
    return fuse_by_reciprocal_rank(
        vector_results,
        keyword_results,
        alpha=alpha,
        rrf_k=rrf_k,
        limit=limit,
    )


def fetch_candidate_chunks(
    question: str,
    *,
    client: OpenAI,
    database_url: str,
    table_name: str,
    embedding_model: str,
    fetch_k: int,
    embedding_cache_enabled: bool = False,
    embedding_cache_redis_url: str = "redis://127.0.0.1:6379/0",
    embedding_cache_ttl_seconds: int = 86400,
    embedding_cache_namespace: str = "health_rag",
) -> list[dict]:
    return fetch_vector_candidate_chunks(
        question,
        client=client,
        database_url=database_url,
        table_name=table_name,
        embedding_model=embedding_model,
        fetch_k=fetch_k,
        embedding_cache_enabled=embedding_cache_enabled,
        embedding_cache_redis_url=embedding_cache_redis_url,
        embedding_cache_ttl_seconds=embedding_cache_ttl_seconds,
        embedding_cache_namespace=embedding_cache_namespace,
    )
