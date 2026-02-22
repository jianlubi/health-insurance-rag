from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

from ambiguity import build_clarification_prompt, needs_clarification
from answer import SYSTEM_PROMPT, build_context
from retrieve import retrieve_chunks


load_dotenv()

app = FastAPI(
    title="Health Insurance RAG API",
    version="0.1.0",
    description="HTTP API for retrieval and question answering over policy documents.",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question.")
    top_k: int = Field(default=4, ge=1, le=20)
    candidate_k: int = Field(default=12, ge=1, le=100)
    use_rerank: bool = True
    model: str = "gpt-4o-mini"
    include_chunks: bool = False


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question.")
    top_k: int = Field(default=4, ge=1, le=20)
    candidate_k: int = Field(default=12, ge=1, le=100)
    use_rerank: bool = True


class ChunkResult(BaseModel):
    id: str
    distance: float
    source: str
    section: str | None = None
    index: int
    token_count: int
    initial_rank: int | None = None
    rerank_score: float | None = None


class RetrieveResponse(BaseModel):
    question: str
    top_k: int
    candidate_k: int
    use_rerank: bool
    retrieved_count: int
    chunks: list[ChunkResult]


class AskResponse(BaseModel):
    question: str
    answer: str
    needs_clarification: bool
    top_k: int
    candidate_k: int
    use_rerank: bool
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
        rerank_score=raw.get("rerank_score"),
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
        model=model,
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


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(payload: RetrieveRequest) -> RetrieveResponse:
    question = _validate_question(payload.question)
    try:
        chunks = retrieve_chunks(
            question,
            top_k=payload.top_k,
            candidate_k=payload.candidate_k,
            use_rerank=payload.use_rerank,
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
        use_rerank=payload.use_rerank,
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
            use_rerank=payload.use_rerank,
            retrieved_count=0,
            chunks=[] if payload.include_chunks else None,
        )

    try:
        chunks = retrieve_chunks(
            question,
            top_k=payload.top_k,
            candidate_k=payload.candidate_k,
            use_rerank=payload.use_rerank,
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
        use_rerank=payload.use_rerank,
        retrieved_count=len(mapped),
        chunks=mapped if payload.include_chunks else None,
    )
