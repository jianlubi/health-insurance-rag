from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import tiktoken


@dataclass(frozen=True)
class Chunk:
    """A retrievable text unit with source metadata."""

    chunk_id: str
    text: str
    source: str
    section: str | None
    index: int
    token_count: int

# Normalize whitespace so chunk boundaries are stable across files/platforms.
def _normalize_text(text: str) -> str:
    # Collapse repeated spaces/tabs into a single space.
    text = re.sub(r"[ \t]+", " ", text)
    # Limit long blank-line runs while preserving paragraph breaks.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_markdown_sections(markdown_text: str) -> list[tuple[str | None, str]]:
    """
    Split markdown into (section_title, section_text) blocks.
    Uses headings as primary boundaries to preserve policy structure.
    """
    sections: list[tuple[str | None, str]] = []
    current_title: str | None = None
    current_lines: list[str] = []

    for line in markdown_text.splitlines():
        # Match markdown headings: "# Title", "## Title", ... up to "###### Title".
        heading = re.match(r"^\s{0,3}(#{1,6})\s+(.*)\s*$", line)
        if heading:
            # Keep heading text as metadata so retrieval can return section context.
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
                current_lines = []
            current_title = heading.group(2).strip() or None
            continue
        current_lines.append(line)

    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))
    return [(title, text) for title, text in sections if text]

def _split_with_overlap(
    text: str,
    *,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    encoding,
) -> list[str]:
    text = _normalize_text(text)
    if not text:
        return []

    tokens = encoding.encode(text)
    if not tokens:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size_tokens, len(tokens))
        candidate = encoding.decode(tokens[start:end]).strip()
        if candidate:
            chunks.append(candidate)
        if end >= len(tokens):
            break
        # Overlap preserves context across neighboring chunks (important for RAG recall).
        start = max(end - chunk_overlap_tokens, start + 1)

    return chunks

# Tiny trailing chunks are often low-signal; merge into previous chunk.
def _merge_small_chunks(
    chunks: list[str], *, min_tokens: int, encoding
) -> list[str]:
    if not chunks:
        return []
    merged: list[str] = []
    for chunk in chunks:      
        if merged and len(encoding.encode(chunk)) < min_tokens:
            merged[-1] = f"{merged[-1]}\n{chunk}".strip()
        else:
            merged.append(chunk)
    return merged


def chunk_markdown(
    markdown_text: str,
    *,
    source: str = "unknown",
    chunk_size_tokens: int = 400,
    chunk_overlap_tokens: int = 80,
    min_chunk_tokens: int = 40,
    model_name: str = "text-embedding-3-small",
) -> list[Chunk]:
    """
    Create retrieval chunks from markdown policy text.
    Typical RAG knobs:
    - chunk_size_tokens: max tokens per chunk
    - chunk_overlap_tokens: tokens repeated into next chunk
    - min_chunk_tokens: drop very short chunks that hurt retrieval quality
    """
    normalized = _normalize_text(markdown_text)
    if not normalized:
        return []

    encoding = tiktoken.encoding_for_model(model_name)
    output: list[Chunk] = []
    running_index = 0

    for section_title, section_text in _split_markdown_sections(normalized):
        # First split each policy section, then assign stable IDs/metadata.
        section_chunks = _split_with_overlap(
            section_text,
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
            encoding=encoding,
        )
        section_chunks = _merge_small_chunks(
            section_chunks, min_tokens=min_chunk_tokens, encoding=encoding
        )
        for piece in section_chunks:
            token_count = len(encoding.encode(piece))
            if token_count < min_chunk_tokens:
                continue
            chunk = Chunk(
                chunk_id=f"{source}:{running_index}",
                text=piece,
                source=source,
                section=section_title,
                index=running_index,
                token_count=token_count,
            )
            output.append(chunk)
            running_index += 1

    return output


def chunk_policy_file(
    file_path: str | Path,
    *,
    chunk_size_tokens: int = 400,
    chunk_overlap_tokens: int = 80,
    min_chunk_tokens: int = 40,
    model_name: str = "text-embedding-3-small",
) -> list[Chunk]:
    path = Path(file_path)
    text = path.read_text(encoding="utf-8")
    return chunk_markdown(
        text,
        source=path.name,
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
        min_chunk_tokens=min_chunk_tokens,
        model_name=model_name,
    )
