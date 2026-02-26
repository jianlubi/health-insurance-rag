from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.openai_client import create_openai_client
from core.telemetry import get_tracer, instrument_fastapi_app, setup_telemetry
from rag.ambiguity import build_clarification_prompt, needs_clarification
from rag.answer import SYSTEM_PROMPT, build_context
from rag.citation import ensure_chunk_citation
from core.config import get_config
from services.rate_service import (
    BASE_BENEFIT_AMOUNT,
    DEFAULT_POLICY_ID,
    coerce_smoker,
    get_rate_quote,
)
from rag.retrieve import retrieve_chunks


load_dotenv()
CFG = get_config()
RETRIEVAL_CFG = CFG["retrieval"]
MODELS_CFG = CFG["models"]
APP_VERSION = "0.1.0"
_RATE_QUESTION_PATTERN = re.compile(
    r"\b(premium|quote|cost|pricing|monthly payment|smoker|non-smoker|rider)\b",
    re.IGNORECASE,
)

setup_telemetry(
    service_version=APP_VERSION,
)

app = FastAPI(
    title="Health Insurance RAG API",
    version=APP_VERSION,
    description=(
        "HTTP API for retrieval-based policy Q&A and DB-backed premium rate quotes."
    ),
)
instrument_fastapi_app(app)
tracer = get_tracer(__name__, APP_VERSION)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question.")
    top_k: int = Field(default=int(RETRIEVAL_CFG["top_k"]), ge=1, le=20)
    candidate_k: int = Field(default=int(RETRIEVAL_CFG["candidate_k"]), ge=1, le=100)
    use_hybrid_search: bool = bool(RETRIEVAL_CFG["use_hybrid_search"])
    keyword_candidate_k: int = Field(
        default=int(RETRIEVAL_CFG["keyword_candidate_k"]), ge=1, le=100
    )
    hybrid_alpha: float = Field(default=float(RETRIEVAL_CFG["hybrid_alpha"]), ge=0, le=1)
    hybrid_rrf_k: int = Field(default=int(RETRIEVAL_CFG["hybrid_rrf_k"]), ge=1, le=200)
    use_rerank: bool = bool(RETRIEVAL_CFG["use_rerank"])
    use_llm_rerank: bool = bool(RETRIEVAL_CFG["use_llm_rerank"])
    llm_rerank_candidate_k: int = Field(
        default=int(RETRIEVAL_CFG["llm_rerank_candidate_k"]), ge=1, le=50
    )
    llm_rerank_keep_k: int = Field(
        default=int(RETRIEVAL_CFG["llm_rerank_keep_k"]), ge=1, le=20
    )
    use_auto_merging: bool = bool(RETRIEVAL_CFG["use_auto_merging"])
    auto_merge_max_gap: int = Field(
        default=int(RETRIEVAL_CFG["auto_merge_max_gap"]), ge=0, le=5
    )
    auto_merge_max_chunks: int = Field(
        default=int(RETRIEVAL_CFG["auto_merge_max_chunks"]), ge=1, le=10
    )
    use_sentence_window: bool = bool(RETRIEVAL_CFG["use_sentence_window"])
    sentence_window_size: int = Field(
        default=int(RETRIEVAL_CFG["sentence_window_size"]), ge=0, le=5
    )
    model: str = str(MODELS_CFG["answer_model"])
    include_chunks: bool = False


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question.")
    top_k: int = Field(default=int(RETRIEVAL_CFG["top_k"]), ge=1, le=20)
    candidate_k: int = Field(default=int(RETRIEVAL_CFG["candidate_k"]), ge=1, le=100)
    use_hybrid_search: bool = bool(RETRIEVAL_CFG["use_hybrid_search"])
    keyword_candidate_k: int = Field(
        default=int(RETRIEVAL_CFG["keyword_candidate_k"]), ge=1, le=100
    )
    hybrid_alpha: float = Field(default=float(RETRIEVAL_CFG["hybrid_alpha"]), ge=0, le=1)
    hybrid_rrf_k: int = Field(default=int(RETRIEVAL_CFG["hybrid_rrf_k"]), ge=1, le=200)
    use_rerank: bool = bool(RETRIEVAL_CFG["use_rerank"])
    use_llm_rerank: bool = bool(RETRIEVAL_CFG["use_llm_rerank"])
    llm_rerank_candidate_k: int = Field(
        default=int(RETRIEVAL_CFG["llm_rerank_candidate_k"]), ge=1, le=50
    )
    llm_rerank_keep_k: int = Field(
        default=int(RETRIEVAL_CFG["llm_rerank_keep_k"]), ge=1, le=20
    )
    use_auto_merging: bool = bool(RETRIEVAL_CFG["use_auto_merging"])
    auto_merge_max_gap: int = Field(
        default=int(RETRIEVAL_CFG["auto_merge_max_gap"]), ge=0, le=5
    )
    auto_merge_max_chunks: int = Field(
        default=int(RETRIEVAL_CFG["auto_merge_max_chunks"]), ge=1, le=10
    )
    use_sentence_window: bool = bool(RETRIEVAL_CFG["use_sentence_window"])
    sentence_window_size: int = Field(
        default=int(RETRIEVAL_CFG["sentence_window_size"]), ge=0, le=5
    )


class ChunkResult(BaseModel):
    id: str
    distance: float
    source: str
    section: str | None = None
    index: int
    token_count: int
    initial_rank: int | None = None
    vector_rank: int | None = None
    keyword_rank: int | None = None
    hybrid_score: float | None = None
    rerank_score: float | None = None
    llm_rerank_rank: int | None = None
    sentence_window_score: float | None = None
    auto_merged: bool | None = None
    merged_from_count: int | None = None


class RetrieveResponse(BaseModel):
    question: str
    top_k: int
    candidate_k: int
    use_hybrid_search: bool
    keyword_candidate_k: int
    hybrid_alpha: float
    hybrid_rrf_k: int
    use_rerank: bool
    use_llm_rerank: bool
    llm_rerank_candidate_k: int
    llm_rerank_keep_k: int
    use_auto_merging: bool
    auto_merge_max_gap: int
    auto_merge_max_chunks: int
    use_sentence_window: bool
    sentence_window_size: int
    retrieved_count: int
    chunks: list[ChunkResult]


class AskResponse(BaseModel):
    question: str
    answer: str
    needs_clarification: bool
    top_k: int
    candidate_k: int
    use_hybrid_search: bool
    keyword_candidate_k: int
    hybrid_alpha: float
    hybrid_rrf_k: int
    use_rerank: bool
    use_llm_rerank: bool
    llm_rerank_candidate_k: int
    llm_rerank_keep_k: int
    use_auto_merging: bool
    auto_merge_max_gap: int
    auto_merge_max_chunks: int
    use_sentence_window: bool
    sentence_window_size: int
    retrieved_count: int
    chunks: list[ChunkResult] | None = None


class RateQuoteRequest(BaseModel):
    age: int = Field(..., ge=0)
    smoker: bool
    riders: list[str] = Field(default_factory=list)
    benefit_amount: float = Field(default=float(BASE_BENEFIT_AMOUNT), gt=0)
    policy_id: str = Field(default=DEFAULT_POLICY_ID, min_length=1)


class RateQuoteResponse(BaseModel):
    policy_id: str
    age: int
    smoker: bool
    age_band: dict[str, int]
    benefit_amount: float
    base_monthly_rate: float
    scaled_base_monthly_rate: float
    applied_riders: list[dict[str, Any]]
    unknown_riders: list[str]
    total_loading_pct: float
    monthly_premium: float
    currency: str
    assumptions: dict[str, Any]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _validate_question(question: str) -> str:
    q = question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="question must not be empty")
    return q


def _map_chunk(raw: dict) -> ChunkResult:
    meta = raw["metadata"]
    return ChunkResult(
        id=raw["id"],
        distance=float(raw["distance"]),
        source=str(meta["source"]),
        section=meta.get("section"),
        index=int(meta["index"]),
        token_count=int(meta["token_count"]),
        initial_rank=raw.get("initial_rank"),
        vector_rank=raw.get("vector_rank"),
        keyword_rank=raw.get("keyword_rank"),
        hybrid_score=raw.get("hybrid_score"),
        rerank_score=raw.get("rerank_score"),
        llm_rerank_rank=raw.get("llm_rerank_rank"),
        sentence_window_score=raw.get("sentence_window_score"),
        auto_merged=raw.get("auto_merged"),
        merged_from_count=raw.get("merged_from_count"),
    )


def _should_use_rate_tool(question: str) -> bool:
    return bool(_RATE_QUESTION_PATTERN.search(question))


def _answer_with_chunks(question: str, chunks: list[dict], model: str) -> str:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required")

    is_rate_question = _should_use_rate_tool(question)
    if not chunks and not is_rate_question:
        return "No relevant chunks found."

    context = (
        build_context(chunks)
        if chunks
        else "No policy context retrieved for this request."
    )
    client = create_openai_client(api_key=openai_api_key)
    if not is_rate_question:
        with tracer.start_as_current_span("openai.answer_completion"):
            completion = client.chat.completions.create(
                model=str(model),
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
        raw = (completion.choices[0].message.content or "").strip()
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
                            "description": (
                                "Selected riders. Example values: "
                                "early_stage_cancer, return_of_premium."
                            ),
                        },
                        "benefit_amount": {
                            "type": "number",
                            "minimum": 0.01,
                            "description": (
                                "Requested benefit amount. "
                                f"Default is {int(BASE_BENEFIT_AMOUNT)}."
                            ),
                        },
                        "policy_id": {
                            "type": "string",
                            "description": (
                                "Policy id for rating tables. "
                                f"Default is {DEFAULT_POLICY_ID}."
                            ),
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
        with tracer.start_as_current_span("openai.rate_tool_step"):
            completion = client.chat.completions.create(
                model=str(model),
                temperature=0,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
        message = completion.choices[0].message

        if not message.tool_calls:
            raw = (message.content or "").strip()
            if tool_called:
                return raw
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
                    with tracer.start_as_current_span("rate_service.get_rate_quote"):
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


@app.post("/rates/quote", response_model=RateQuoteResponse)
def quote_rate(payload: RateQuoteRequest) -> RateQuoteResponse:
    try:
        with tracer.start_as_current_span("rate_service.quote_rate_api"):
            quote = get_rate_quote(
                age=payload.age,
                smoker=payload.smoker,
                riders=payload.riders,
                benefit_amount=payload.benefit_amount,
                policy_id=payload.policy_id,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"rate quote failed: {exc}") from exc
    return RateQuoteResponse(**quote)


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(payload: RetrieveRequest) -> RetrieveResponse:
    question = _validate_question(payload.question)
    try:
        with tracer.start_as_current_span("rag.retrieve_endpoint") as span:
            span.set_attribute("rag.top_k", payload.top_k)
            span.set_attribute("rag.candidate_k", payload.candidate_k)
            span.set_attribute("rag.use_hybrid_search", payload.use_hybrid_search)
            span.set_attribute("rag.use_rerank", payload.use_rerank)
            span.set_attribute("rag.use_llm_rerank", payload.use_llm_rerank)
            chunks = retrieve_chunks(
                question,
                top_k=payload.top_k,
                candidate_k=payload.candidate_k,
                use_hybrid_search=payload.use_hybrid_search,
                keyword_candidate_k=payload.keyword_candidate_k,
                hybrid_alpha=payload.hybrid_alpha,
                hybrid_rrf_k=payload.hybrid_rrf_k,
                use_rerank=payload.use_rerank,
                use_llm_rerank=payload.use_llm_rerank,
                llm_rerank_candidate_k=payload.llm_rerank_candidate_k,
                llm_rerank_keep_k=payload.llm_rerank_keep_k,
                use_auto_merging=payload.use_auto_merging,
                auto_merge_max_gap=payload.auto_merge_max_gap,
                auto_merge_max_chunks=payload.auto_merge_max_chunks,
                use_sentence_window=payload.use_sentence_window,
                sentence_window_size=payload.sentence_window_size,
            )
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"retrieval failed: {exc}") from exc

    mapped = [_map_chunk(c) for c in chunks]
    return RetrieveResponse(
        question=question,
        top_k=payload.top_k,
        candidate_k=payload.candidate_k,
        use_hybrid_search=payload.use_hybrid_search,
        keyword_candidate_k=payload.keyword_candidate_k,
        hybrid_alpha=payload.hybrid_alpha,
        hybrid_rrf_k=payload.hybrid_rrf_k,
        use_rerank=payload.use_rerank,
        use_llm_rerank=payload.use_llm_rerank,
        llm_rerank_candidate_k=payload.llm_rerank_candidate_k,
        llm_rerank_keep_k=payload.llm_rerank_keep_k,
        use_auto_merging=payload.use_auto_merging,
        auto_merge_max_gap=payload.auto_merge_max_gap,
        auto_merge_max_chunks=payload.auto_merge_max_chunks,
        use_sentence_window=payload.use_sentence_window,
        sentence_window_size=payload.sentence_window_size,
        retrieved_count=len(mapped),
        chunks=mapped,
    )


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    question = _validate_question(payload.question)
    is_rate_question = _should_use_rate_tool(question)

    if needs_clarification(question):
        answer = build_clarification_prompt(question)
        return AskResponse(
            question=question,
            answer=answer,
            needs_clarification=True,
            top_k=payload.top_k,
            candidate_k=payload.candidate_k,
            use_hybrid_search=payload.use_hybrid_search,
            keyword_candidate_k=payload.keyword_candidate_k,
            hybrid_alpha=payload.hybrid_alpha,
            hybrid_rrf_k=payload.hybrid_rrf_k,
            use_rerank=payload.use_rerank,
            use_llm_rerank=payload.use_llm_rerank,
            llm_rerank_candidate_k=payload.llm_rerank_candidate_k,
            llm_rerank_keep_k=payload.llm_rerank_keep_k,
            use_auto_merging=payload.use_auto_merging,
            auto_merge_max_gap=payload.auto_merge_max_gap,
            auto_merge_max_chunks=payload.auto_merge_max_chunks,
            use_sentence_window=payload.use_sentence_window,
            sentence_window_size=payload.sentence_window_size,
            retrieved_count=0,
            chunks=[] if payload.include_chunks else None,
        )

    try:
        with tracer.start_as_current_span("rag.ask_endpoint") as span:
            span.set_attribute("rag.top_k", payload.top_k)
            span.set_attribute("rag.candidate_k", payload.candidate_k)
            span.set_attribute("rag.use_hybrid_search", payload.use_hybrid_search)
            span.set_attribute("rag.use_rerank", payload.use_rerank)
            span.set_attribute("rag.use_llm_rerank", payload.use_llm_rerank)
            span.set_attribute("rag.is_rate_question", is_rate_question)
            if is_rate_question:
                chunks: list[dict] = []
            else:
                chunks = retrieve_chunks(
                    question,
                    top_k=payload.top_k,
                    candidate_k=payload.candidate_k,
                    use_hybrid_search=payload.use_hybrid_search,
                    keyword_candidate_k=payload.keyword_candidate_k,
                    hybrid_alpha=payload.hybrid_alpha,
                    hybrid_rrf_k=payload.hybrid_rrf_k,
                    use_rerank=payload.use_rerank,
                    use_llm_rerank=payload.use_llm_rerank,
                    llm_rerank_candidate_k=payload.llm_rerank_candidate_k,
                    llm_rerank_keep_k=payload.llm_rerank_keep_k,
                    use_auto_merging=payload.use_auto_merging,
                    auto_merge_max_gap=payload.auto_merge_max_gap,
                    auto_merge_max_chunks=payload.auto_merge_max_chunks,
                    use_sentence_window=payload.use_sentence_window,
                    sentence_window_size=payload.sentence_window_size,
                )
            answer = _answer_with_chunks(question, chunks, payload.model)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"answering failed: {exc}") from exc

    mapped = [_map_chunk(c) for c in chunks]
    return AskResponse(
        question=question,
        answer=answer,
        needs_clarification=False,
        top_k=payload.top_k,
        candidate_k=payload.candidate_k,
        use_hybrid_search=payload.use_hybrid_search,
        keyword_candidate_k=payload.keyword_candidate_k,
        hybrid_alpha=payload.hybrid_alpha,
        hybrid_rrf_k=payload.hybrid_rrf_k,
        use_rerank=payload.use_rerank,
        use_llm_rerank=payload.use_llm_rerank,
        llm_rerank_candidate_k=payload.llm_rerank_candidate_k,
        llm_rerank_keep_k=payload.llm_rerank_keep_k,
        use_auto_merging=payload.use_auto_merging,
        auto_merge_max_gap=payload.auto_merge_max_gap,
        auto_merge_max_chunks=payload.auto_merge_max_chunks,
        use_sentence_window=payload.use_sentence_window,
        sentence_window_size=payload.sentence_window_size,
        retrieved_count=len(mapped),
        chunks=mapped if payload.include_chunks else None,
    )

