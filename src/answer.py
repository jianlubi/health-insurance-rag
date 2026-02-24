from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from ambiguity import build_clarification_prompt, needs_clarification
from citation import ensure_chunk_citation
from config import get_config
from rate_service import (
    BASE_BENEFIT_AMOUNT,
    DEFAULT_POLICY_ID,
    coerce_smoker,
    get_rate_quote,
)
from retrieve import retrieve_chunks

SYSTEM_PROMPT = (
    "You answer insurance-policy questions using ONLY the provided context. "
    "If context is insufficient, say so clearly. "
    "Cite chunk ids in square brackets like [demolife_critical_illness_policy.md:3]. "
    "Citations must use exactly one colon and integer chunk index. "
    "Do not cite section clause numbers like [demolife_critical_illness_policy.md:3.2]."
)
_RATE_QUESTION_PATTERN = re.compile(
    r"\b(premium|quote|cost|pricing|monthly payment|smoker|non-smoker|rider)\b",
    re.IGNORECASE,
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


def _should_use_rate_tool(question: str) -> bool:
    return bool(_RATE_QUESTION_PATTERN.search(question))


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
    is_rate_question = _should_use_rate_tool(question)
    if is_rate_question:
        chunks: list[dict] = []
    else:
        chunks = retrieve_chunks(question, top_k=top_k)
    if not chunks and not is_rate_question:
        return "No relevant chunks found."

    context = (
        build_context(chunks)
        if chunks
        else "No policy context retrieved for this request."
    )
    client = OpenAI(api_key=openai_api_key)

    if not is_rate_question:
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

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_rate_quote",
                "description": (
                    "Get a monthly premium quote from database rate tables "
                    "using customer profile fields."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "age": {"type": "integer", "minimum": 0},
                        "smoker": {
                            "type": "string",
                            "description": (
                                "Smoking status. Use one of: smoker, non-smoker, "
                                "true, false."
                            ),
                        },
                        "riders": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "benefit_amount": {
                            "type": "number",
                            "minimum": 0.01,
                        },
                        "policy_id": {
                            "type": "string",
                        },
                    },
                    "required": ["age", "smoker"],
                },
            },
        }
    ]
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                SYSTEM_PROMPT
                + " For premium/rate quote questions, you must call the "
                "get_rate_quote function and use its result as the source of truth. "
                "Do not compute premiums from the policy document text. "
                f"If benefit amount is missing, assume {int(BASE_BENEFIT_AMOUNT)}. "
                "If required rating inputs are missing, ask a concise follow-up question."
            ),
        },
        {
            "role": "user",
            "content": f"Question:\n{question}\n\nContext:\n{context}",
        },
    ]
    tool_called = False
    for _ in range(3):
        completion = client.chat.completions.create(
            model=str(models_cfg["answer_model"]),
            temperature=0,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        message = completion.choices[0].message
        if not message.tool_calls:
            if tool_called:
                return (message.content or "").strip()
            raw = (message.content or "").strip()
            if raw:
                return raw
            return (
                "I can provide a premium quote from the rating service, "
                "but I need at least age and smoker status."
            )

        tool_calls_payload: list[dict[str, Any]] = []
        for tool_call in message.tool_calls:
            tool_calls_payload.append(
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments or "{}",
                    },
                }
            )
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": tool_calls_payload,
            }
        )

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            args_raw = tool_call.function.arguments or "{}"
            try:
                args = json.loads(args_raw)
            except json.JSONDecodeError:
                args = {}

            if tool_name != "get_rate_quote":
                tool_result: dict[str, Any] = {
                    "error": f"unsupported tool '{tool_name}'"
                }
            else:
                try:
                    age = int(args["age"])
                    smoker = coerce_smoker(args["smoker"])
                    riders_raw = args.get("riders", [])
                    if isinstance(riders_raw, str):
                        riders = [riders_raw]
                    elif isinstance(riders_raw, list):
                        riders = [str(item) for item in riders_raw]
                    else:
                        raise ValueError("riders must be a list of strings")
                    benefit_amount = float(
                        args.get("benefit_amount", BASE_BENEFIT_AMOUNT)
                    )
                    policy_id = str(args.get("policy_id", DEFAULT_POLICY_ID))
                    tool_result = get_rate_quote(
                        age=age,
                        smoker=smoker,
                        riders=riders,
                        benefit_amount=benefit_amount,
                        policy_id=policy_id,
                    )
                except KeyError:
                    tool_result = {
                        "error": "missing required inputs: age and smoker are required"
                    }
                except Exception as exc:
                    tool_result = {"error": str(exc)}
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result),
                }
            )
            tool_called = True

    return "I could not complete the rate lookup. Please try again."


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
