from __future__ import annotations


def _score_for_order(chunk: dict) -> float:
    if "sentence_window_score" in chunk:
        return float(chunk["sentence_window_score"])
    if "rerank_score" in chunk:
        return float(chunk["rerank_score"])
    return -float(chunk.get("distance", 0.0))


def _merge_group(group: list[dict]) -> dict:
    if len(group) == 1:
        item = dict(group[0])
        item["auto_merged"] = False
        item["merged_from_count"] = 1
        return item

    first = group[0]
    source = str(first["metadata"].get("source") or "unknown")
    idxs = [int(c["metadata"]["index"]) for c in group]
    min_idx, max_idx = min(idxs), max(idxs)
    ids = [str(c["id"]) for c in group]

    sections: list[str] = []
    for chunk in group:
        section = str(chunk["metadata"].get("section") or "")
        if section and section not in sections:
            sections.append(section)

    merged = dict(first)
    merged["id"] = f"{source}:{min_idx}-{max_idx}"
    merged["content"] = "\n\n".join(str(c.get("content") or "") for c in group).strip()
    merged["distance"] = min(float(c.get("distance", 0.0)) for c in group)
    merged["initial_rank"] = min(
        int(c.get("initial_rank", 10**9)) for c in group if c.get("initial_rank") is not None
    )
    merged["auto_merged"] = True
    merged["merged_from_ids"] = ids
    merged["merged_from_count"] = len(ids)

    if any("rerank_score" in c for c in group):
        merged["rerank_score"] = max(float(c.get("rerank_score", -1e9)) for c in group)
    if any("sentence_window_score" in c for c in group):
        merged["sentence_window_score"] = max(
            float(c.get("sentence_window_score", -1e9)) for c in group
        )

    merged["metadata"] = {
        **first["metadata"],
        "section": " | ".join(sections) if sections else first["metadata"].get("section"),
        "index": min_idx,
        "token_count": sum(int(c["metadata"].get("token_count", 0)) for c in group),
        "merged_range_start": min_idx,
        "merged_range_end": max_idx,
        "merged_from_ids": ids,
    }
    return merged


def auto_merge_chunks(
    chunks: list[dict],
    *,
    max_gap: int = 1,
    max_merged_chunks: int = 3,
) -> list[dict]:
    if not chunks:
        return []

    max_gap = max(0, int(max_gap))
    max_merged_chunks = max(1, int(max_merged_chunks))

    ordered = sorted(
        chunks,
        key=lambda c: (
            str(c["metadata"].get("source") or ""),
            str(c["metadata"].get("section") or ""),
            int(c["metadata"].get("index", 0)),
        ),
    )

    merged: list[dict] = []
    current_group: list[dict] = [ordered[0]]

    def flush() -> None:
        nonlocal current_group
        if current_group:
            merged.append(_merge_group(current_group))
            current_group = []

    for chunk in ordered[1:]:
        prev = current_group[-1]
        same_source = str(prev["metadata"].get("source")) == str(
            chunk["metadata"].get("source")
        )
        same_section = str(prev["metadata"].get("section")) == str(
            chunk["metadata"].get("section")
        )
        prev_idx = int(prev["metadata"].get("index", 0))
        idx = int(chunk["metadata"].get("index", 0))
        close_enough = 0 <= (idx - prev_idx) <= max_gap
        group_not_full = len(current_group) < max_merged_chunks

        if same_source and same_section and close_enough and group_not_full:
            current_group.append(chunk)
        else:
            flush()
            current_group = [chunk]

    flush()
    merged.sort(key=_score_for_order, reverse=True)
    return merged

