from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

from ambiguity import build_clarification_prompt, needs_clarification
from answer import SYSTEM_PROMPT, build_context
from citation import ensure_chunk_citation
from config import get_config
from retrieve import retrieve_chunks


load_dotenv()
CFG = get_config()
RETRIEVAL_CFG = CFG["retrieval"]
MODELS_CFG = CFG["models"]

app = FastAPI(
    title="Health Insurance RAG API",
    version="0.1.0",
    description="HTTP API for retrieval and question answering over policy documents.",
)


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


def _answer_with_chunks(question: str, chunks: list[dict], model: str) -> str:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required")

    if not chunks:
        return "No relevant chunks found."

    context = build_context(chunks)
    client = OpenAI(api_key=openai_api_key)
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
    raw = completion.choices[0].message.content or ""
    return ensure_chunk_citation(raw, chunks)


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(payload: RetrieveRequest) -> RetrieveResponse:
    question = _validate_question(payload.question)
    try:
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
