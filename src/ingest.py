from __future__ import annotations

import json
from pathlib import Path

from chunking import chunk_policy_file


def build_records(
    policies_dir: Path,
    *,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    min_chunk_tokens: int,
    model_name: str,
) -> list[dict]:
    records: list[dict] = []
    policy_files = sorted(policies_dir.glob("*.md"))

    for file_path in policy_files:
        chunks = chunk_policy_file(
            file_path,
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
            min_chunk_tokens=min_chunk_tokens,
            model_name=model_name,
        )
        for chunk in chunks:
            records.append(
                {
                    "id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": {
                        "source": chunk.source,
                        "section": chunk.section,
                        "index": chunk.index,
                        "token_count": chunk.token_count,
                    },
                }
            )
    return records


def write_jsonl(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    policies_dir = Path("data/policies")
    output_path = Path("data/chunks/policy_chunks.jsonl")
    chunk_size_tokens = 400
    chunk_overlap_tokens = 80
    min_chunk_tokens = 40
    model_name = "text-embedding-3-small"

    records = build_records(
        policies_dir,
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
        min_chunk_tokens=min_chunk_tokens,
        model_name=model_name,
    )
    write_jsonl(records, output_path)

    print(f"Files processed: {len(list(policies_dir.glob('*.md')))}")
    print(f"Chunks written: {len(records)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
