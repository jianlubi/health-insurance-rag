from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_batch

from core.config import get_config
from core.openai_client import create_openai_client


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def batched(items: list[dict], batch_size: int) -> Iterable[list[dict]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def embed_records(records: list[dict], client, model: str) -> list[list[float]]:
    texts = [r["text"] for r in records]
    resp = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in resp.data]

# pgvector SQL inserts expect values like "[0.1,0.2,0.3]" (other vector DB SDKs
# usually accept raw list[float] directly, so this conversion is pgvector-specific).
def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in values) + "]"


def ensure_schema(cur, *, table_name: str, embedding_dim: int) -> None:
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            source TEXT NOT NULL,
            section TEXT,
            chunk_index INTEGER NOT NULL,
            token_count INTEGER NOT NULL,
            embedding vector({embedding_dim}) NOT NULL
        );
        """
    )
    cur.execute(
        f"""
        CREATE INDEX IF NOT EXISTS {table_name}_embedding_ivfflat_idx
        ON {table_name}
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
        """
    )


def clear_table(cur, *, table_name: str) -> None:
    # Indexing is a full-corpus refresh; clear old rows so removed/renamed chunks
    # cannot remain in retrieval results.
    cur.execute(f"TRUNCATE TABLE {table_name};")


def upsert_rows(cur, *, table_name: str, rows: list[tuple]) -> None:
    query = f"""
        INSERT INTO {table_name} (
            id, content, source, section, chunk_index, token_count, embedding
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
        ON CONFLICT (id) DO UPDATE SET
            content = EXCLUDED.content,
            source = EXCLUDED.source,
            section = EXCLUDED.section,
            chunk_index = EXCLUDED.chunk_index,
            token_count = EXCLUDED.token_count,
            embedding = EXCLUDED.embedding;
    """
    execute_batch(cur, query, rows, page_size=100)


def main() -> None:
    cfg = get_config()
    index_cfg = cfg["index"]

    parser = argparse.ArgumentParser(description="Embed and index chunks into pgvector.")
    parser.add_argument(
        "--input-path",
        default=str(index_cfg["input_path"]),
        help="Input chunk JSONL file path.",
    )
    parser.add_argument(
        "--table-name",
        default=str(index_cfg["table_name"]),
        help="Target pgvector table name.",
    )
    parser.add_argument(
        "--embedding-model",
        default=str(index_cfg["embedding_model"]),
        help="Embedding model for indexing.",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=int(index_cfg["embed_batch_size"]),
        help="Batch size for embedding and upsert.",
    )
    args = parser.parse_args()

    load_dotenv()

    database_url = os.getenv("DATABASE_URL")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not database_url:
        raise ValueError("DATABASE_URL is required")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required")

    input_path = Path(args.input_path)
    table_name = str(args.table_name)
    embedding_model = str(args.embedding_model)
    embed_batch_size = max(1, int(args.embed_batch_size))

    records = read_jsonl(input_path)
    if not records:
        print("No records found in JSONL. Run ingest.py first.")
        return

    client = create_openai_client(api_key=openai_api_key)

    # pgvector columns must declare a fixed dimension (vector(N)), so we embed
    # one batch first and infer N from the first embedding.
    first_batch = records[:embed_batch_size]
    first_embeddings = embed_records(first_batch, client, embedding_model)
    embedding_dim = len(first_embeddings[0])

    with psycopg2.connect(database_url) as conn:
        with conn.cursor() as cur:
            ensure_schema(cur, table_name=table_name, embedding_dim=embedding_dim)
            clear_table(cur, table_name=table_name)
            print(f"Cleared existing rows in table '{table_name}'")

            # Upsert first embedded batch.
            first_rows: list[tuple] = []
            for record, embedding in zip(first_batch, first_embeddings, strict=False):
                meta = record["metadata"]
                first_rows.append(
                    (
                        record["id"],
                        record["text"],
                        meta["source"],
                        meta.get("section"),
                        meta["index"],
                        meta["token_count"],
                        vector_literal(embedding),
                    )
                )
            upsert_rows(cur, table_name=table_name, rows=first_rows)
            print(f"Indexed batch 1 with {len(first_rows)} chunks")

            # Upsert remaining batches.
            batch_num = 1
            for batch in batched(records[embed_batch_size:], embed_batch_size):
                embeddings = embed_records(batch, client, embedding_model)
                rows: list[tuple] = []
                for record, embedding in zip(batch, embeddings, strict=False):
                    meta = record["metadata"]
                    rows.append(
                        (
                            record["id"],
                            record["text"],
                            meta["source"],
                            meta.get("section"),
                            meta["index"],
                            meta["token_count"],
                            vector_literal(embedding),
                        )
                    )
                upsert_rows(cur, table_name=table_name, rows=rows)
                batch_num += 1
                print(f"Indexed batch {batch_num} with {len(rows)} chunks")

        conn.commit()

    print(f"Done. Indexed {len(records)} chunks into table '{table_name}'.")


if __name__ == "__main__":
    main()

