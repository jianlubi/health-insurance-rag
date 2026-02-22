from __future__ import annotations

import json
import re

from openai import OpenAI


def _clip_text(text: str, *, max_chars: int = 900) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _parse_ranked_indices(raw: str, *, candidate_count: int) -> list[int]:
    ranked: list[int] = []
    seen: set[int] = set()

    def add_index(value: int) -> None:
        if 1 <= value <= candidate_count and value not in seen:
            ranked.append(value)
            seen.add(value)

    parsed_values: list[object] = []
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            maybe = payload.get("ranked_indices")
            if isinstance(maybe, list):
                parsed_values = maybe
        elif isinstance(payload, list):
            parsed_values = payload
    except json.JSONDecodeError:
        parsed_values = []

    for item in parsed_values:
        if isinstance(item, int):
            add_index(item)
        elif isinstance(item, str) and item.strip().isdigit():
            add_index(int(item.strip()))

    if not ranked:
        for match in re.findall(r"\d+", raw):
            add_index(int(match))

    for idx in range(1, candidate_count + 1):
        add_index(idx)

    return ranked


def llm_rerank_chunks(
    question: str,
    chunks: list[dict],
    *,
    client: OpenAI,
    model: str,
) -> list[dict]:
    if not chunks:
        return []

    numbered_chunks: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        numbered_chunks.append(
            (
                f"Chunk {idx}\n"
                f"id: {chunk.get('id')}\n"
                f"source: {meta.get('source')}\n"
                f"section: {meta.get('section')}\n"
                f"text: {_clip_text(str(chunk.get('content') or ''))}"
            )
        )

    system_prompt = (
        "You rerank retrieved chunks for answering a policy question. "
        "Return JSON only with key 'ranked_indices', a list of chunk numbers "
        "from most relevant to least relevant."
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        "Chunks:\n"
        f"{chr(10).join(numbered_chunks)}\n\n"
        "Output format:\n"
        '{"ranked_indices":[1,2,3]}'
    )

    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = completion.choices[0].message.content or ""
    ranked_indices = _parse_ranked_indices(raw, candidate_count=len(chunks))

    reranked: list[dict] = []
    for rank, chunk_num in enumerate(ranked_indices, start=1):
        item = dict(chunks[chunk_num - 1])
        item["llm_rerank_rank"] = rank
        reranked.append(item)
    return reranked

