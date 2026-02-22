# RAG Notes

- pgvector works, but raw SQL feels low-level because of vector literal formatting.
- Keep JSONL output. It makes debugging and re-indexing easier.
- For small datasets, retrieval can miss rows unless `ivfflat.probes` is increased.
- If I change embedding model, I probably need to re-index everything.
- Retrieval quality depends a lot on chunk settings and `top_k`.
- Next cleanup: create a small `vector_store.py` wrapper so index/retrieve code is cleaner.
