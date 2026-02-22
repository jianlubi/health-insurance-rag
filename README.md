# Health Insurance RAG

RAG prototype for answering health/critical-illness policy questions using:
- PostgreSQL + `pgvector` retrieval
- OpenAI embeddings + LLM answer generation
- Ambiguity detection/clarification flow
- FastAPI backend
- Gradio frontend

The dataset in `data/policies/` is synthetic for demo/testing.

## Features

- Section-aware markdown chunking
- Vector retrieval with optional reranking
- Clarifying-question behavior for ambiguous prompts
- Grounded answers with chunk-id citations
- Batch evaluation + reporting scripts
- API and UI for interactive usage

## Current Eval Snapshot

From `data/eval/eval_results.jsonl` (latest run):
- Retrieval hit: `20/20 (100.0%)`
- Grounded: `20/20 (100.0%)`
- Insufficient-context: `5/30 (16.7%)`
- Failures: `0/30 (0.0%)`
- Clarifying-question asked: `5/5 (100.0%)`

## Project Structure

```text
data/
  policies/                 # Source markdown policies
  chunks/                   # Generated chunk JSONL
  eval/                     # Eval questions + eval output JSONL
notes/                      # Project notes and experiment logs
src/
  chunking.py               # Chunking logic
  ingest.py                 # Build chunks JSONL from policies
  index.py                  # Embed + index chunks into pgvector
  retrieve.py               # Retrieval + reranking
  ambiguity.py              # Ambiguity detection/clarification prompts
  answer.py                 # Single-question answer flow
  eval.py                   # Batch evaluator
  report_eval.py            # Eval summary report
  api.py                    # FastAPI service
  gradio_app.py             # Gradio frontend
```

## Requirements

- Python 3.11+ (tested in local `venv`)
- PostgreSQL with `pgvector` extension
- OpenAI API key

Install dependencies:

```powershell
venv\Scripts\pip install -r requirements.txt
```

## Environment Variables

Create `.env` in project root:

```env
OPENAI_API_KEY=your_openai_api_key
DATABASE_URL=postgresql://user:password@host:5432/dbname
```

## End-to-End Pipeline

1. Generate chunks from policy markdown:

```powershell
venv\Scripts\python src\ingest.py
```

Current chunk settings (`src/ingest.py`):
- `chunk_size_tokens=400`
- `chunk_overlap_tokens=80`
- `min_chunk_tokens=40`

2. Index chunks into pgvector:

```powershell
venv\Scripts\python src\index.py
```

3. Ask one question via CLI:

```powershell
venv\Scripts\python src\answer.py "What illnesses are covered by this policy?"
```

4. Run evaluation:

```powershell
venv\Scripts\python src\eval.py
venv\Scripts\python src\report_eval.py
```

## FastAPI Backend

Start API server:

```powershell
venv\Scripts\uvicorn api:app --app-dir src --reload
```

Default URL: `http://127.0.0.1:8000`

Endpoints:
- `GET /health`
- `POST /retrieve`
- `POST /ask`

Example:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/ask -Method Post -ContentType "application/json" -Body '{"question":"What are the main exclusions?"}'
```

## Gradio Frontend

Start UI:

```powershell
venv\Scripts\python src\gradio_app.py --host 127.0.0.1 --port 7860
```

Open: `http://127.0.0.1:7860`

Tabs:
- `Ask`: full QA flow
- `Retrieve`: retrieval-only debugging view

## Screenshots

### Gradio Ask

![Gradio Ask](docs/images/gradio-ask.png)

### Gradio Retrieve

![Gradio Retrieve](docs/images/gradio-retrieve.png)

### Eval Report

![Eval Report](docs/images/eval-report.png)

## Retrieval and Reranking

Retrieval flow (`src/retrieve.py`):
1. Embed query with `text-embedding-3-small`
2. Fetch top candidates from pgvector (`candidate_k`, default 12)
3. Optional rerank by query-chunk cosine similarity using `text-embedding-3-large`
4. Return final top `k` (default 4)

## Notes

- Eval output (`data/eval/eval_results.jsonl`) is generated and overwritten on each eval run.
- Indexing currently does a full table refresh before insert to avoid stale rows.

## Next Iteration Ideas

- Sentence-window retrieval experiments
- Larger/multi-policy eval set
- Latency/cost benchmark with rerank on/off
- Automated pipeline script (`ingest -> index -> eval -> report`)
