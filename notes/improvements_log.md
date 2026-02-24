# RAG Improvement Log

## 1) Baseline: Basic RAG

- Implemented a basic RAG system.
- Initial retrieval hit rate was about **83%** overall.
- Retrieval for the **complex** category was about **67%**.

## 2) Retrieval Analysis and Reranking

- Analyzed eval outputs and found that, for several questions, relevant sections were often at rank **5-6**.
- Implemented a reranking stage.
- After reranking, both overall retrieval and complex-category retrieval improved to **90%+**.

## 3) Chunking Experiments

- Tweaked and tested chunking settings.
- Confirmed the best chunk size is **400 tokens** (current setting).

## 4) Groundedness Improvement

- Groundedness was around **82%**.
- After reviewing eval results, identified that when context was insufficient, the system could still answer from pre-trained knowledge.
- Updated prompts to strongly require refusal when context is insufficient.
- After this fix, groundedness improved to **100%**.

## 5) Retrieval Cache Benchmark (Redis)

- Goal: compare retrieval latency before vs after cache population.
- Setup:
  - Redis backend enabled (`cache.backend=redis`, `cache.retrieval_enabled=true`).
  - Same dataset: **30 eval queries**.
  - Same retrieval settings and `top_k`.
  - Run A (cold): `FLUSHDB` first, then run all 30 queries once.
  - Run B (warm): immediately rerun the same 30 queries.

- Results:
  - Cold cache total: **18.535 s**
  - Warm cache total: **0.046 s**
  - Speedup (total): **~403x**

- Per-query latency:
  - Cold avg / p50 / p95: **617.83 ms / 564.37 ms / 750.78 ms**
  - Warm avg / p50 / p95: **1.55 ms / 1.28 ms / 2.02 ms**

- Cache footprint after benchmark:
  - Retrieval keys: **30**
  - Query embedding keys: **30**
  - Total keys in Redis DB: **60**

- Conclusion:
  - Retrieval cache is working as expected.
  - Repeated queries now bypass retrieval compute and return near-instantly.
