from __future__ import annotations

from html import escape
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from assistant.orchestrator import remember_session_profile, run_insurance_assistant
from core.config import get_config
from core.telemetry import get_tracer, instrument_fastapi_app, setup_telemetry
from rag.retrieve import retrieve_chunks
from services.eligibility_service import check_eligibility
from services.quote_service import generate_quote
from services.rate_service import (
    BASE_BENEFIT_AMOUNT,
    DEFAULT_POLICY_ID,
    get_rate_quote,
)


load_dotenv()
CFG = get_config()
RETRIEVAL_CFG = CFG["retrieval"]
MODELS_CFG = CFG["models"]
APP_VERSION = "0.2.0"

setup_telemetry(service_version=APP_VERSION)

app = FastAPI(
    title="Health Insurance AI Assistant API",
    version=APP_VERSION,
    description=(
        "Insurance AI assistant API with LangGraph routing across "
        "RAG, eligibility checks, quotes, and rate lookups."
    ),
)
instrument_fastapi_app(app)
tracer = get_tracer(__name__, APP_VERSION)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question.")
    session_id: str | None = Field(
        default=None,
        description="Optional session id to retain profile fields across turns.",
    )
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
    include_service_result: bool = False


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
    route: Literal["rag", "eligibility", "quote", "rate"]
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
    service_result: dict[str, Any] | None = None
    chunks: list[ChunkResult] | None = None


class AssistantAskResponse(BaseModel):
    question: str
    route: Literal["rag", "eligibility", "quote", "rate"]
    answer: str
    needs_clarification: bool
    retrieved_count: int
    service_result: dict[str, Any] | None = None
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


class EligibilityCheckRequest(BaseModel):
    age: int = Field(..., ge=0)
    has_preexisting_condition: bool = False
    currently_hospitalized: bool = False
    policy_id: str = Field(default=DEFAULT_POLICY_ID, min_length=1)
    session_id: str | None = None


class EligibilityCheckResponse(BaseModel):
    policy_id: str
    age: int
    eligible: bool
    reasons: list[str]
    evaluated_rules: list[dict[str, Any]]


class QuoteRequest(BaseModel):
    age: int = Field(..., ge=0)
    smoker: bool
    riders: list[str] = Field(default_factory=list)
    benefit_amount: float = Field(default=float(BASE_BENEFIT_AMOUNT), gt=0)
    has_preexisting_condition: bool = False
    currently_hospitalized: bool = False
    policy_id: str = Field(default=DEFAULT_POLICY_ID, min_length=1)
    session_id: str | None = None


class QuoteResponse(BaseModel):
    status: Literal["quoted", "rejected"]
    message: str
    eligibility: EligibilityCheckResponse
    rate_quote: RateQuoteResponse | None = None
    application_url: str | None = None


def _validate_question(question: str) -> str:
    q = question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="question must not be empty")
    return q


def _map_chunk(raw: dict[str, Any]) -> ChunkResult:
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


def _assistant_retrieval_options(payload: AskRequest) -> dict[str, Any]:
    return {
        "top_k": payload.top_k,
        "candidate_k": payload.candidate_k,
        "use_hybrid_search": payload.use_hybrid_search,
        "keyword_candidate_k": payload.keyword_candidate_k,
        "hybrid_alpha": payload.hybrid_alpha,
        "hybrid_rrf_k": payload.hybrid_rrf_k,
        "use_rerank": payload.use_rerank,
        "use_llm_rerank": payload.use_llm_rerank,
        "llm_rerank_candidate_k": payload.llm_rerank_candidate_k,
        "llm_rerank_keep_k": payload.llm_rerank_keep_k,
        "use_auto_merging": payload.use_auto_merging,
        "auto_merge_max_gap": payload.auto_merge_max_gap,
        "auto_merge_max_chunks": payload.auto_merge_max_chunks,
        "use_sentence_window": payload.use_sentence_window,
        "sentence_window_size": payload.sentence_window_size,
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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


@app.post("/eligibility/check", response_model=EligibilityCheckResponse)
def check_policy_eligibility(payload: EligibilityCheckRequest) -> EligibilityCheckResponse:
    try:
        with tracer.start_as_current_span("eligibility_service.check_api"):
            result = check_eligibility(
                age=payload.age,
                has_preexisting_condition=payload.has_preexisting_condition,
                currently_hospitalized=payload.currently_hospitalized,
                policy_id=payload.policy_id,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"eligibility check failed: {exc}") from exc
    if payload.session_id:
        remember_session_profile(
            payload.session_id,
            {
                "age": payload.age,
                "policy_id": payload.policy_id,
                "has_preexisting_condition": payload.has_preexisting_condition,
                "currently_hospitalized": payload.currently_hospitalized,
            },
        )
    return EligibilityCheckResponse(**result)


@app.post("/quote/generate", response_model=QuoteResponse)
def generate_policy_quote(payload: QuoteRequest) -> QuoteResponse:
    try:
        with tracer.start_as_current_span("quote_service.generate_api"):
            result = generate_quote(
                age=payload.age,
                smoker=payload.smoker,
                riders=payload.riders,
                benefit_amount=payload.benefit_amount,
                has_preexisting_condition=payload.has_preexisting_condition,
                currently_hospitalized=payload.currently_hospitalized,
                policy_id=payload.policy_id,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"quote generation failed: {exc}") from exc
    if payload.session_id:
        remember_session_profile(
            payload.session_id,
            {
                "age": payload.age,
                "smoker": payload.smoker,
                "riders": payload.riders,
                "benefit_amount": payload.benefit_amount,
                "policy_id": payload.policy_id,
                "has_preexisting_condition": payload.has_preexisting_condition,
                "currently_hospitalized": payload.currently_hospitalized,
            },
        )

    return QuoteResponse(
        status=result["status"],
        message=result["message"],
        eligibility=EligibilityCheckResponse(**result["eligibility"]),
        rate_quote=(
            RateQuoteResponse(**result["rate_quote"]) if result.get("rate_quote") else None
        ),
        application_url=result.get("application_url"),
    )


@app.get("/application/complete", response_class=HTMLResponse)
def complete_application_page(
    age: int | None = Query(default=None, ge=0),
    smoker: bool | None = None,
    riders: str | None = None,
    benefit_amount: float | None = Query(default=None, gt=0),
    policy_id: str | None = None,
    has_preexisting_condition: bool | None = None,
    currently_hospitalized: bool | None = None,
) -> HTMLResponse:
    rider_values = [item.strip() for item in str(riders or "").split(",") if item.strip()]
    rider_text = ", ".join(rider_values) if rider_values else ""

    def _safe(value: Any) -> str:
        return escape(str(value if value is not None else ""))

    html = f"""
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Complete Your Application</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 0; background: #f4f6fb; color: #111827; }}
      .wrap {{ max-width: 760px; margin: 40px auto; background: #fff; border: 1px solid #dbe2f0; border-radius: 12px; padding: 24px; }}
      h1 {{ margin-top: 0; font-size: 28px; }}
      p {{ line-height: 1.5; }}
      .note {{ padding: 12px; border-radius: 8px; background: #eff6ff; border: 1px solid #bfdbfe; margin: 16px 0 22px; }}
      form {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
      label {{ display: block; font-size: 13px; margin-bottom: 6px; color: #374151; }}
      input {{ width: 100%; box-sizing: border-box; padding: 10px; border: 1px solid #cbd5e1; border-radius: 8px; }}
      .full {{ grid-column: 1 / -1; }}
      button {{ margin-top: 8px; padding: 12px 16px; border: 0; border-radius: 8px; background: #0f766e; color: #fff; cursor: pointer; }}
    </style>
  </head>
  <body>
    <main class="wrap">
      <h1>Complete your application</h1>
      <p class="note">
        This page is pre-populated with the information you provided during quote generation.
        Please review the details below and complete your application on this page.
      </p>
      <form>
        <div>
          <label for="age">Age</label>
          <input id="age" name="age" value="{_safe(age)}" />
        </div>
        <div>
          <label for="smoker">Smoker</label>
          <input id="smoker" name="smoker" value="{_safe(smoker)}" />
        </div>
        <div>
          <label for="benefit_amount">Benefit Amount</label>
          <input id="benefit_amount" name="benefit_amount" value="{_safe(benefit_amount)}" />
        </div>
        <div>
          <label for="policy_id">Policy ID</label>
          <input id="policy_id" name="policy_id" value="{_safe(policy_id)}" />
        </div>
        <div class="full">
          <label for="riders">Riders</label>
          <input id="riders" name="riders" value="{_safe(rider_text)}" />
        </div>
        <div>
          <label for="has_preexisting_condition">Has Pre-existing Condition</label>
          <input id="has_preexisting_condition" name="has_preexisting_condition" value="{_safe(has_preexisting_condition)}" />
        </div>
        <div>
          <label for="currently_hospitalized">Currently Hospitalized</label>
          <input id="currently_hospitalized" name="currently_hospitalized" value="{_safe(currently_hospitalized)}" />
        </div>
        <div class="full">
          <button type="button">Continue Application</button>
        </div>
      </form>
    </main>
  </body>
</html>
"""
    return HTMLResponse(content=html)


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


@app.post("/assistant/ask", response_model=AssistantAskResponse)
def assistant_ask(payload: AskRequest) -> AssistantAskResponse:
    question = _validate_question(payload.question)
    try:
        with tracer.start_as_current_span("assistant.ask_endpoint"):
            result = run_insurance_assistant(
                question=question,
                model=str(payload.model),
                session_id=payload.session_id,
                retrieval_options=_assistant_retrieval_options(payload),
                include_chunks=True,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"assistant failed: {exc}") from exc

    mapped = [_map_chunk(c) for c in list(result.get("chunks") or [])]
    route = str(result.get("route") or "rag")
    if route not in {"rag", "eligibility", "quote", "rate"}:
        route = "rag"
    return AssistantAskResponse(
        question=question,
        route=route,  # type: ignore[arg-type]
        answer=str(result.get("answer") or ""),
        needs_clarification=bool(result.get("needs_clarification") or False),
        retrieved_count=len(mapped),
        service_result=result.get("service_result") if payload.include_service_result else None,
        chunks=mapped if payload.include_chunks else None,
    )


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    question = _validate_question(payload.question)
    try:
        with tracer.start_as_current_span("assistant.ask_compat_endpoint"):
            result = run_insurance_assistant(
                question=question,
                model=str(payload.model),
                session_id=payload.session_id,
                retrieval_options=_assistant_retrieval_options(payload),
                include_chunks=True,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"answering failed: {exc}") from exc

    mapped = [_map_chunk(c) for c in list(result.get("chunks") or [])]
    route = str(result.get("route") or "rag")
    if route not in {"rag", "eligibility", "quote", "rate"}:
        route = "rag"
    return AskResponse(
        question=question,
        route=route,  # type: ignore[arg-type]
        answer=str(result.get("answer") or ""),
        needs_clarification=bool(result.get("needs_clarification") or False),
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
        service_result=result.get("service_result") if payload.include_service_result else None,
        chunks=mapped if payload.include_chunks else None,
    )
