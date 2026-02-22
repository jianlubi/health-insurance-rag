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
