from __future__ import annotations

import re
from typing import Any

from rag.retrieval.rerank_retriever import cosine_similarity


def _split_sentences(text: str) -> list[str]:
    # Keep sentence splitting lightweight and deterministic for retrieval windows.
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [p.strip() for p in parts if p.strip()]


def _build_sentence_window(sentences: list[str], center: int, window_size: int) -> str:
    start = max(0, center - window_size)
    end = min(len(sentences), center + window_size + 1)
    return " ".join(sentences[start:end]).strip()


def sentence_window_chunks(
    question: str,
    chunks: list[dict],
    *,
    client: Any,
    model: str,
    window_size: int,
    openai_request_kwargs: dict[str, Any] | None = None,
) -> list[dict]:
    if not chunks:
        return []

    sentence_records: list[tuple[int, int, str]] = []
    chunk_sentences: list[list[str]] = []
    for chunk_index, chunk in enumerate(chunks):
        sentences = _split_sentences(str(chunk.get("content") or ""))
        if not sentences:
            sentences = [str(chunk.get("content") or "").strip()]
        chunk_sentences.append(sentences)
        for sentence_index, sentence in enumerate(sentences):
            if sentence:
                sentence_records.append((chunk_index, sentence_index, sentence))

    if not sentence_records:
        return chunks

    resp = client.embeddings.create(
        model=model,
        input=[question, *[item[2] for item in sentence_records]],
        **(openai_request_kwargs or {}),
    )
    vectors = [item.embedding for item in resp.data]
    query_vector = vectors[0]
    sentence_vectors = vectors[1:]

    best_for_chunk: dict[int, tuple[float, int]] = {}
    for (chunk_index, sentence_index, _), vector in zip(
        sentence_records, sentence_vectors, strict=False
    ):
        score = cosine_similarity(query_vector, vector)
        current = best_for_chunk.get(chunk_index)
        if current is None or score > current[0]:
            best_for_chunk[chunk_index] = (score, sentence_index)

    windowed: list[dict] = []
    for chunk_index, chunk in enumerate(chunks):
        best = best_for_chunk.get(chunk_index)
        if best is None:
            continue
        best_score, best_sentence_index = best
        sentences = chunk_sentences[chunk_index]
        window_text = _build_sentence_window(sentences, best_sentence_index, window_size)
        item = dict(chunk)
        if window_text:
            item["content"] = window_text
        item["sentence_window_score"] = best_score
        item["metadata"] = {
            **chunk["metadata"],
            "sentence_window_center": best_sentence_index,
            "sentence_window_size": window_size,
        }
        windowed.append(item)

    windowed.sort(key=lambda c: c.get("sentence_window_score", 0.0), reverse=True)
    return windowed


