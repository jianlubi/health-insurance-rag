from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from ambiguity import build_clarification_prompt, needs_clarification
from retrieve import retrieve_chunks

SYSTEM_PROMPT = (
    "You answer insurance-policy questions using ONLY the provided context. "
    "If context is insufficient, say so clearly. "
    "Cite chunk ids in square brackets like [demolife_critical_illness_policy.md:3]. "
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


def answer_question(question: str, *, top_k: int = 4) -> str:
    load_dotenv()

    if needs_clarification(question):
        return build_clarification_prompt(question)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required")

    chunks = retrieve_chunks(question, top_k=top_k)
    if not chunks:
        return "No relevant chunks found."

    context = build_context(chunks)
    client = OpenAI(api_key=openai_api_key)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
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
    return completion.choices[0].message.content or ""


def main() -> None:
    default_question = "What illnesses are covered by this policy?"
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else default_question

    answer = answer_question(question, top_k=4)
    print(f"Question: {question}\n")
    print("Answer:")
    print(answer)


if __name__ == "__main__":
    main()
