from __future__ import annotations

import re


# Convert malformed citation styles like [file.md:6:5.5] to [file.md:6].
_EXTRA_COLON_CITATION = re.compile(r"\[([^\[\]]+?\.md:\d+):[^\[\]]+\]")
_CHUNK_CITATION = re.compile(r"\[[^\]]+\.md:\d+(?:\.\d+)?\]")


def normalize_chunk_citations(text: str) -> str:
    if not text:
        return text
    return _EXTRA_COLON_CITATION.sub(r"[\1]", text)


def has_chunk_citation(text: str) -> bool:
    return bool(_CHUNK_CITATION.search(text or ""))


def ensure_chunk_citation(text: str, chunks: list[dict]) -> str:
    normalized = normalize_chunk_citations(text)
    if has_chunk_citation(normalized):
        return normalized
    if not chunks:
        return normalized
    first_id = str(chunks[0].get("id") or "").strip()
    if not first_id:
        return normalized
    return normalized.rstrip() + f" [{first_id}]"
