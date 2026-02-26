from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.chunking import chunk_policy_file
from core.config import get_config


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
    cfg = get_config()
    ingest_cfg = cfg["ingest"]

    parser = argparse.ArgumentParser(description="Build chunk JSONL from policy markdown.")
    parser.add_argument(
        "--policies-dir",
        default=str(ingest_cfg["policies_dir"]),
        help="Directory containing policy markdown files.",
    )
    parser.add_argument(
        "--output-path",
        default=str(ingest_cfg["output_path"]),
        help="Output JSONL path for chunks.",
    )
    parser.add_argument(
        "--chunk-size-tokens",
        type=int,
        default=int(ingest_cfg["chunk_size_tokens"]),
        help="Maximum chunk size in tokens.",
    )
    parser.add_argument(
        "--chunk-overlap-tokens",
        type=int,
        default=int(ingest_cfg["chunk_overlap_tokens"]),
        help="Chunk overlap in tokens.",
    )
    parser.add_argument(
        "--min-chunk-tokens",
        type=int,
        default=int(ingest_cfg["min_chunk_tokens"]),
        help="Minimum chunk size in tokens.",
    )
    parser.add_argument(
        "--model-name",
        default=str(ingest_cfg["model_name"]),
        help="Tokenizer model name used for token counting.",
    )
    args = parser.parse_args()

    policies_dir = Path(args.policies_dir)
    output_path = Path(args.output_path)
    chunk_size_tokens = max(1, int(args.chunk_size_tokens))
    chunk_overlap_tokens = max(0, int(args.chunk_overlap_tokens))
    min_chunk_tokens = max(1, int(args.min_chunk_tokens))
    model_name = str(args.model_name)

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

