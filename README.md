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
- Vector retrieval with optional embedding rerank + LLM rerank
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
  config.py                 # Config loader + defaults + deep merge
  retrieval/
    chunk_retriever.py      # Candidate chunk retrieval from pgvector
    rerank_retriever.py     # Reranking logic
    llm_rerank_retriever.py # LLM-based reranking logic
    auto_merging_retriever.py  # Adjacent-chunk auto merge logic
    sentence_window_retriever.py  # Sentence-window selection logic
  chunking.py               # Chunking logic
  ingest.py                 # Build chunks JSONL from policies
  index.py                  # Embed + index chunks into pgvector
  retrieve.py               # Retrieval orchestrator
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

## Configuration (YAML)

Defaults are centralized in `config/config.yaml`. The scripts (`ingest.py`, `index.py`, `retrieve.py`, `answer.py`, `eval.py`, `report_eval.py`), FastAPI (`api.py`), and Gradio (`gradio_app.py`) all read from it.

Optional override file location:

```powershell
$env:RAG_CONFIG_PATH="D:\path\to\custom-config.yaml"
```

CLI flags still override YAML values for one-off runs.

## End-to-End Pipeline

1. Generate chunks from policy markdown:

```powershell
venv\Scripts\python src\ingest.py
```

2. Index chunks into pgvector:

```powershell
venv\Scripts\python src\index.py
```

3. Ask one question via CLI:

```powershell
venv\Scripts\python src\answer.py "What illnesses are covered by this policy?"
```

Example with custom retrieval depth:

```powershell
venv\Scripts\python src\answer.py --top-k 6 "What illnesses are covered by this policy?"
```

4. Run evaluation:

```powershell
venv\Scripts\python src\eval.py
venv\Scripts\python src\report_eval.py
```

Example with custom eval settings:

```powershell
venv\Scripts\python src\eval.py --top-k 6 --max-questions 30
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

## Retrieval and Reranking

Retrieval flow (`src/retrieve.py`):
1. Embed query with `text-embedding-3-small`
2. Fetch top candidates from pgvector (`candidate_k`, default 12)
3. Optional rerank by query-chunk cosine similarity using `text-embedding-3-large`
4. Optional LLM rerank on top candidates (default model `gpt-4o-mini`)
5. Optional auto-merging of adjacent chunks from the same source/section
6. Optional sentence-window selection (score sentences per chunk, return windowed text around best sentence)
7. Return final top `k` (default 4)

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

## Notes

- Eval output (`data/eval/eval_results.jsonl`) is generated and overwritten on each eval run.
- Indexing currently does a full table refresh before insert to avoid stale rows.

## Next Iteration Ideas

- Sentence-window retrieval experiments
- Larger/multi-policy eval set
- Latency/cost benchmark with rerank on/off
- Automated pipeline script (`ingest -> index -> eval -> report`)
