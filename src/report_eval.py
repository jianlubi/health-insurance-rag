from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from config import get_config


def pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return (numerator / denominator) * 100.0


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def bool_count(rows: list[dict], key: str) -> tuple[int, int]:
    checks = 0
    hits = 0
    for row in rows:
        val = row.get("metrics", {}).get(key)
        if val is None:
            continue
        checks += 1
        if bool(val):
            hits += 1
    return hits, checks


def failure_type_counts(rows: list[dict]) -> tuple[dict[str, int], int]:
    counts: dict[str, int] = defaultdict(int)
    total_failures = 0
    for row in rows:
        metrics = row.get("metrics", {})
        if metrics.get("failure") is not True:
            continue
        total_failures += 1
        failure_type = str(metrics.get("failure_type") or "unspecified")
        counts[failure_type] += 1
    return dict(counts), total_failures


def summarize(rows: list[dict]) -> None:
    total = len(rows)
    retrieval_hits, retrieval_checks = bool_count(rows, "retrieval_hit")
    grounded_hits, grounded_checks = bool_count(rows, "grounded")
    insufficient_hits, _ = bool_count(rows, "insufficient_context")
    failure_hits, _ = bool_count(rows, "failure")
    failure_types, failure_type_total = failure_type_counts(rows)
    clarification_needed_hits, _ = bool_count(rows, "clarification_needed")
    clarification_rows = [
        r for r in rows if r.get("metrics", {}).get("clarification_needed") is True
    ]
    asked_clarifying_hits, asked_clarifying_checks = bool_count(
        clarification_rows, "asked_clarifying_question"
    )

    print("Overall")
    print(f"- rows: {total}")
    print(
        f"- retrieval_hit: {retrieval_hits}/{retrieval_checks} "
        f"({pct(retrieval_hits, retrieval_checks):.1f}%)"
    )
    print(
        f"- grounded: {grounded_hits}/{grounded_checks} "
        f"({pct(grounded_hits, grounded_checks):.1f}%)"
    )
    print(
        f"- insufficient_context: {insufficient_hits}/{total} "
        f"({pct(insufficient_hits, total):.1f}%)"
    )
    print(
        f"- failure: {failure_hits}/{total} "
        f"({pct(failure_hits, total):.1f}%)"
    )
    if failure_type_total > 0:
        breakdown = ", ".join(
            f"{name}={count}" for name, count in sorted(failure_types.items())
        )
        print(f"- failure_type (on failed rows): {breakdown}")
    else:
        print("- failure_type (on failed rows): none")
    print(
        f"- clarification_needed: {clarification_needed_hits}/{total} "
        f"({pct(clarification_needed_hits, total):.1f}%)"
    )
    print(
        f"- asked_clarifying_question: "
        f"{asked_clarifying_hits}/{asked_clarifying_checks} "
        f"({pct(asked_clarifying_hits, asked_clarifying_checks):.1f}%)"
    )

    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_category[str(row.get("category") or "uncategorized")].append(row)

    print("\nBy category")
    for category in sorted(by_category.keys()):
        cat_rows = by_category[category]
        cat_total = len(cat_rows)
        cat_retrieval_hits, cat_retrieval_checks = bool_count(cat_rows, "retrieval_hit")
        cat_grounded_hits, cat_grounded_checks = bool_count(cat_rows, "grounded")
        cat_insufficient_hits, _ = bool_count(cat_rows, "insufficient_context")
        cat_failure_hits, _ = bool_count(cat_rows, "failure")
        cat_failure_types, cat_failure_type_total = failure_type_counts(cat_rows)
        cat_clarification_needed_hits, _ = bool_count(cat_rows, "clarification_needed")
        cat_clarification_rows = [
            r for r in cat_rows if r.get("metrics", {}).get("clarification_needed") is True
        ]
        cat_asked_clarifying_hits, cat_asked_clarifying_checks = bool_count(
            cat_clarification_rows, "asked_clarifying_question"
        )
        cat_failure_type_str = (
            ", ".join(
                f"{name}={count}" for name, count in sorted(cat_failure_types.items())
            )
            if cat_failure_type_total > 0
            else "none"
        )
        print(
            f"- {category}: "
            f"retrieval {cat_retrieval_hits}/{cat_retrieval_checks} "
            f"({pct(cat_retrieval_hits, cat_retrieval_checks):.1f}%), "
            f"grounded {cat_grounded_hits}/{cat_grounded_checks} "
            f"({pct(cat_grounded_hits, cat_grounded_checks):.1f}%), "
            f"insufficient {cat_insufficient_hits}/{cat_total} "
            f"({pct(cat_insufficient_hits, cat_total):.1f}%), "
            f"failure {cat_failure_hits}/{cat_total} "
            f"({pct(cat_failure_hits, cat_total):.1f}%), "
            f"failure_type {cat_failure_type_str}, "
            f"clarification_needed {cat_clarification_needed_hits}/{cat_total} "
            f"({pct(cat_clarification_needed_hits, cat_total):.1f}%), "
            f"asked_clarifying_question {cat_asked_clarifying_hits}/"
            f"{cat_asked_clarifying_checks} "
            f"({pct(cat_asked_clarifying_hits, cat_asked_clarifying_checks):.1f}%)"
        )

    misses = [r for r in rows if r.get("metrics", {}).get("retrieval_hit") is False]
    print(f"\nRetrieval misses ({len(misses)})")
    for i, row in enumerate(misses, start=1):
        category = str(row.get("category") or "uncategorized")
        question = str(row.get("question") or "")
        expected_sections = row.get("expected_sections") or []
        print(f"{i}. [{category}] {question}")
        print(f"   expected_sections={expected_sections}")


def main() -> None:
    cfg = get_config()
    eval_cfg = cfg["eval"]

    parser = argparse.ArgumentParser(description="Summarize RAG eval results JSONL.")
    parser.add_argument(
        "--input",
        default=str(eval_cfg["output_path"]),
        help="Path to eval results JSONL file.",
    )
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    summarize(rows)


if __name__ == "__main__":
    main()
