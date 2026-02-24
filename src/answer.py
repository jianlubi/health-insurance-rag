from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv
from openai import OpenAI

from ambiguity import build_clarification_prompt, needs_clarification
from citation import ensure_chunk_citation
from config import get_config
from retrieve import retrieve_chunks

SYSTEM_PROMPT = (
    "You answer insurance-policy questions using ONLY the provided context. "
    "If context is insufficient, say so clearly. "
    "Cite chunk ids in square brackets like [demolife_critical_illness_policy.md:3]. "
    "Citations must use exactly one colon and integer chunk index. "
    "Do not cite section clause numbers like [demolife_critical_illness_policy.md:3.2]."
)


def build_context(chunks: list[dict]) -> str:
    parts: list[str] = []
    for chunk in chunks:
        meta = chunk["metadata"]
        parts.append(
            (
                f"[{chunk['id']}]\n"
                f"source: {meta['source']}\n"
                f"section: {meta['section']}\n"
                f"text: {chunk['content']}"
            )
        )
    return "\n\n".join(parts)


def answer_question(question: str, *, top_k: int | None = None) -> str:
    load_dotenv()
    cfg = get_config()
    retrieval_cfg = cfg["retrieval"]
    models_cfg = cfg["models"]

    if needs_clarification(question):
        return build_clarification_prompt(question)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required")

    top_k = int(top_k if top_k is not None else retrieval_cfg["top_k"])
    chunks = retrieve_chunks(question, top_k=top_k)
    if not chunks:
        return "No relevant chunks found."

    context = build_context(chunks)
    client = OpenAI(api_key=openai_api_key)

    completion = client.chat.completions.create(
        model=str(models_cfg["answer_model"]),
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nContext:\n{context}",
            },
        ],
    )
    raw = completion.choices[0].message.content or ""
    return ensure_chunk_citation(raw, chunks)


def main() -> None:
    cfg = get_config()
    retrieval_cfg = cfg["retrieval"]
    answer_cfg = cfg["answer"]

    parser = argparse.ArgumentParser(description="Ask one policy question.")
    parser.add_argument("question", nargs="*", help="Question text.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(retrieval_cfg["top_k"]),
        help="Number of chunks to retrieve.",
    )
    args = parser.parse_args()

    default_question = str(answer_cfg["default_question"])
    question = " ".join(args.question).strip() if args.question else default_question

    answer = answer_question(question, top_k=max(1, args.top_k))
    print(f"Question: {question}\n")
    print("Answer:")
    print(answer)


if __name__ == "__main__":
    main()
