# Health Insurance RAG

RAG prototype for answering health/critical-illness policy questions using:
- PostgreSQL + `pgvector` retrieval
- OpenAI embeddings + LLM answer generation
- LangChain + LangGraph assistant orchestration
- Ambiguity detection/clarification flow
- FastAPI backend
- Gradio frontend

The dataset in `data/policies/` is synthetic for demo/testing.

## Features

- Section-aware markdown chunking
- Vector or hybrid (vector + keyword) retrieval with optional embedding rerank + LLM rerank
- Clarifying-question behavior for ambiguous prompts
- Grounded answers with chunk-id citations
- Deterministic eligibility service
- DB-backed premium rating service + quote service (eligibility-gated)
- Agentic request routing across `rag`, `eligibility`, `quote`, and `rate` services
- Batch evaluation + reporting scripts
- API and UI for interactive usage

## Current Eval Snapshot

From `data/eval/eval_results.jsonl` (latest run):
- Retrieval hit: `20/20 (100.0%)`
- Grounded: `20/20 (100.0%)`
- Insufficient-context: `5/30 (16.7%)`
- Failures: `0/30 (0.0%)`
- Clarifying-question asked: `5/5 (100.0%)`

## Screenshots

### Gradio Ask

![Gradio Ask](docs/images/gradio-ask.png)

### Gradio Retrieve

![Gradio Retrieve](docs/images/gradio-retrieve.png)

### Eval Report

![Eval Report](docs/images/eval-report.png)

### Retriever Comparison

![Retriever Comparison](docs/images/retriever-comparison.png)

## Project Structure

```text
data/
  policies/                 # Source markdown policies
  chunks/                   # Generated chunk JSONL
  eval/                     # Eval questions + eval output JSONL
config/
  config.yaml               # Runtime defaults for models/retrieval/eval/ingest/index/ui
notes/                      # Project notes and experiment logs
src/
  api/
    app.py                  # FastAPI service
  assistant/
    orchestrator.py         # LangGraph assistant/router orchestration
  core/
    config.py               # Config loader + defaults + deep merge
    chunking.py             # Chunking logic
  rag/
    retrieve.py             # Retrieval orchestrator
    answer.py               # Single-question answer flow
    ambiguity.py            # Ambiguity detection/clarification prompts
    citation.py             # Citation normalization/validation
    retrieval/
      chunk_retriever.py      # Vector/keyword candidate retrieval + hybrid fusion
      rerank_retriever.py     # Embedding reranking logic
      llm_rerank_retriever.py # LLM-based reranking logic
      auto_merging_retriever.py   # Adjacent-chunk auto merge logic
      sentence_window_retriever.py # Sentence-window selection logic
  services/
    eligibility_service.py  # Deterministic eligibility rules
    quote_service.py        # Eligibility-gated quote generation
    rate_service.py         # DB-backed premium rating logic
    seed_rates.py           # Create/seed rate tables
  pipelines/
    ingest.py               # Build chunks JSONL from policies
    index.py                # Embed + index chunks into pgvector
    eval.py                 # Batch evaluator
    report_eval.py          # Eval summary report
  ui/
    gradio_app.py           # Gradio frontend
scripts/                    # Lightweight wrappers for CLI entrypoints
```

## Requirements

- Python 3.11+ (tested in local `venv`)
- PostgreSQL with `pgvector` extension
- OpenAI API key

## Docker (Recommended DB Setup)

This repo includes `docker-compose.yml` for:
- `postgres` (with `pgvector`)
- `pgadmin` (web UI)
- `otel-collector` (OTLP receiver/export pipeline)
- `jaeger` (trace UI)

Start services:

```powershell
docker compose up -d
```

Stop services:

```powershell
docker compose down
```

Default ports:
- Postgres: `localhost:5433`
- pgAdmin: `http://localhost:5050`
- Jaeger UI: `http://localhost:16686`
- OTLP gRPC: `localhost:4317`
- OTLP HTTP: `localhost:4318`

Update `.env` for Docker Postgres:

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5433/insurance_rag
```

pgAdmin login defaults (from compose):
- Email: `admin@local.dev`
- Password: `admin`

When adding server in pgAdmin:
- Host: `postgres`
- Port: `5432`
- Username: `postgres`
- Password: `postgres`
- Database: `insurance_rag`

Install dependencies:

```powershell
venv\Scripts\pip install -r requirements.txt
```

## Environment Variables

Copy `.env.example` to `.env`, then fill in real credentials:

```powershell
Copy-Item .env.example .env
```

Canonical env var definitions live in `.env.example`:

Optional Langfuse tracing:

- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_BASE_URL` (default: `https://cloud.langfuse.com`)
- `LANGFUSE_TRACING_ENABLED` (default: `true`)

When Langfuse keys are present, OpenAI SDK calls are automatically routed through
Langfuse instrumentation.

Note: On Python `3.14+`, this project pins Langfuse `4.x` prerelease for compatibility.

## Configuration (YAML)

Defaults are centralized in `config/config.yaml`. CLI wrappers in `scripts/`, FastAPI (`api.app`), and Gradio (`ui.gradio_app`) all read from it.

Optional override file location:

```powershell
$env:RAG_CONFIG_PATH="D:\path\to\custom-config.yaml"
```

CLI flags still override YAML values for one-off runs.

## End-to-End Pipeline

1. Generate chunks from policy markdown:

```powershell
venv\Scripts\python scripts\ingest.py
```

2. Index chunks into pgvector:

```powershell
venv\Scripts\python scripts\index.py
```

3. Create/seed DB rate tables (required for premium quotes):

```powershell
venv\Scripts\python scripts\seed_rates.py
```

4. Ask one question via CLI:

```powershell
venv\Scripts\python scripts\answer.py "What illnesses are covered by this policy?"
```

Example with custom retrieval depth:

```powershell
venv\Scripts\python scripts\answer.py --top-k 6 "What illnesses are covered by this policy?"
```

5. Run evaluation:

```powershell
venv\Scripts\python scripts\eval.py
venv\Scripts\python scripts\report_eval.py
```

Example with custom eval settings:

```powershell
venv\Scripts\python scripts\eval.py --top-k 6 --max-questions 30
```

## FastAPI Backend

Start API server:

```powershell
venv\Scripts\uvicorn api.app:app --app-dir src --reload
```

Default URL: `http://127.0.0.1:8000`

Endpoints:
- `GET /health`
- `POST /retrieve`
- `POST /ask`
- `POST /assistant/ask`
- `POST /eligibility/check`
- `POST /quote/generate`
- `POST /rates/quote`

Example:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/ask -Method Post -ContentType "application/json" -Body '{"question":"What are the main exclusions?"}'
```

Premium quote API example:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/rates/quote -Method Post -ContentType "application/json" -Body '{"age":42,"smoker":false,"riders":["early_stage_cancer"]}'
```

Eligibility API example:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/eligibility/check -Method Post -ContentType "application/json" -Body '{"age":42,"has_preexisting_condition":false,"currently_hospitalized":false}'
```

Quote API example (eligibility-gated):

```powershell
Invoke-RestMethod http://127.0.0.1:8000/quote/generate -Method Post -ContentType "application/json" -Body '{"age":42,"smoker":false,"riders":["early_stage_cancer"]}'
```

Assistant routing example via `/assistant/ask`:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/assistant/ask -Method Post -ContentType "application/json" -Body '{"session_id":"demo-user-1","question":"What is my monthly premium if I am 42, non-smoker, with early-stage cancer rider?","include_service_result":true}'
```

Tip: pass the same `session_id` across turns so the assistant can reuse previously provided profile fields (for example age/smoker).

## OpenTelemetry

Tracing is built in for FastAPI requests, OpenAI/httpx calls, Postgres (`psycopg2`), Redis, and retrieval pipeline stages.

1. Start supporting services:

```powershell
docker compose up -d otel-collector jaeger
```

2. In `.env`, set `OTEL_ENABLED=true`.

3. Start API as usual and open Jaeger:
- `http://127.0.0.1:16686`

## Gradio Frontend

Start UI:

```powershell
venv\Scripts\python scripts\gradio_app.py --host 127.0.0.1 --port 7860
```

Open: `http://127.0.0.1:7860`

Tabs:
- `Assistant`: agentic routing (`rag` / `eligibility` / `quote` / `rate`) with service payload visibility
- `Ask`: full QA flow
- `Retrieve`: retrieval-only debugging view

## Retrieval and Reranking

Retrieval flow (`src/rag/retrieve.py`):
1. Embed query with `text-embedding-3-small`
2. Fetch candidates from pgvector (`candidate_k`, default 12)
3. Optional hybrid mode: fetch keyword candidates and fuse vector + keyword ranks with RRF
4. Optional rerank by query-chunk cosine similarity using `text-embedding-3-large`
5. Optional LLM rerank on top candidates (default model `gpt-4o-mini`)
6. Optional auto-merging of adjacent chunks from the same source/section
7. Optional sentence-window selection (score sentences per chunk, return windowed text around best sentence)
8. Return final top `k` (default 4)

Hybrid search controls:
- `use_hybrid_search` (default: `false`)
- `keyword_candidate_k` (default: `12`)
- `hybrid_alpha` (default: `0.7`, vector-weight in RRF blend)
- `hybrid_rrf_k` (default: `60`)

Auto-merging controls:
- `use_auto_merging` (default: `false`)
- `auto_merge_max_gap` (default: `1`)
- `auto_merge_max_chunks` (default: `3`)

Sentence-window controls:
- `use_sentence_window` (default: `false`)
- `sentence_window_size` (default: `1`, meaning `best sentence +/- 1`)

Rerank control:
- `use_rerank` (default: `false`)

LLM rerank controls:
- `use_llm_rerank` (default: `false`)
- `llm_rerank_candidate_k` (default: `8`)
- `llm_rerank_keep_k` (default: `4`)

All default values above come from `config/config.yaml`.

Query-embedding cache (Redis):
- `cache.enabled` (default: `true`)
- `cache.backend` (default: `redis`)
- `cache.redis_url` (default: `redis://127.0.0.1:6379/0`)
- `cache.embedding_ttl_seconds` (default: `86400`)
- `cache.retrieval_enabled` (default: `true`)
- `cache.retrieval_ttl_seconds` (default: `300`)
- `cache.retrieval_version` (default: `v1`, bump to invalidate retrieval cache)
- `cache.key_prefix` (default: `health_rag`)

## Notes

- Eval output (`data/eval/eval_results.jsonl`) is generated and overwritten on each eval run.
- Indexing currently does a full table refresh before insert to avoid stale rows.

## Next Iteration Ideas

- Sentence-window retrieval experiments
- Larger/multi-policy eval set
- Latency/cost benchmark with rerank on/off
- Automated pipeline script (`ingest -> index -> eval -> report`)
