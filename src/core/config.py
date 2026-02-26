from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import os
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "models": {
        "answer_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "rerank_model": "text-embedding-3-large",
        "llm_rerank_model": "gpt-4o-mini",
    },
    "retrieval": {
        "table_name": "policy_chunks",
        "top_k": 4,
        "candidate_k": 12,
        "use_hybrid_search": False,
        "keyword_candidate_k": 12,
        "hybrid_alpha": 0.7,
        "hybrid_rrf_k": 60,
        "use_rerank": False,
        "use_llm_rerank": False,
        "llm_rerank_candidate_k": 8,
        "llm_rerank_keep_k": 4,
        "use_auto_merging": False,
        "auto_merge_max_gap": 1,
        "auto_merge_max_chunks": 3,
        "use_sentence_window": False,
        "sentence_window_size": 1,
    },
    "cache": {
        "enabled": True,
        "backend": "redis",
        "redis_url": "redis://127.0.0.1:6379/0",
        "embedding_ttl_seconds": 86400,
        "retrieval_enabled": True,
        "retrieval_ttl_seconds": 300,
        "retrieval_version": "v1",
        "key_prefix": "insurance_rag",
    },
    "answer": {
        "default_question": "What illnesses are covered by this policy?",
    },
    "eval": {
        "questions_path": "data/eval/test_questions.json",
        "output_path": "data/eval/eval_results.jsonl",
        "top_k": 4,
        "max_questions": 30,
    },
    "ingest": {
        "policies_dir": "data/policies",
        "output_path": "data/chunks/policy_chunks.jsonl",
        "chunk_size_tokens": 400,
        "chunk_overlap_tokens": 80,
        "min_chunk_tokens": 40,
        "model_name": "text-embedding-3-small",
    },
    "index": {
        "input_path": "data/chunks/policy_chunks.jsonl",
        "table_name": "policy_chunks",
        "embedding_model": "text-embedding-3-small",
        "embed_batch_size": 64,
    },
    "gradio": {
        "host": "127.0.0.1",
        "port": 7860,
        "share": False,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_user_config() -> dict[str, Any]:
    env_path = os.getenv("RAG_CONFIG_PATH")
    if env_path:
        config_path = Path(env_path)
    else:
        search_paths: list[Path] = [
            Path("config/config.yaml"),
            Path("config.yaml"),  # legacy path compatibility
        ]
        module_path = Path(__file__).resolve()
        for parent in module_path.parents:
            search_paths.append(parent / "config" / "config.yaml")
            search_paths.append(parent / "config.yaml")  # legacy path compatibility

        deduped_search_paths: list[Path] = []
        seen: set[str] = set()
        for candidate in search_paths:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            deduped_search_paths.append(candidate)

        search_paths = deduped_search_paths
        config_path = next((p for p in search_paths if p.exists()), search_paths[0])

    if not config_path.exists():
        return {}
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    return loaded


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    user_config = _load_user_config()
    return _deep_merge(DEFAULT_CONFIG, user_config)
