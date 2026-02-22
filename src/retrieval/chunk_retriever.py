from __future__ import annotations

from openai import OpenAI
import psycopg2


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in values) + "]"


def embed_query(question: str, client: OpenAI, model: str) -> list[float]:
    resp = client.embeddings.create(model=model, input=question)
    return resp.data[0].embedding


def fetch_candidate_chunks(
    question: str,
    *,
    client: OpenAI,
    database_url: str,
    table_name: str,
    embedding_model: str,
    fetch_k: int,
) -> list[dict]:
    query_vector = embed_query(question, client, embedding_model)
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

