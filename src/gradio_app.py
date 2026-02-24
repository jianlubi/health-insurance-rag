from __future__ import annotations

import argparse
import json

from fastapi import HTTPException
import gradio as gr

from api import AskRequest, RetrieveRequest, ask, retrieve
from config import get_config


CFG = get_config()
RETRIEVAL_CFG = CFG["retrieval"]
MODELS_CFG = CFG["models"]
GRADIO_CFG = CFG["gradio"]


CHUNK_HEADERS = [
    "id",
    "distance",
    "source",
    "section",
    "index",
    "token_count",
    "initial_rank",
    "vector_rank",
    "keyword_rank",
    "hybrid_score",
    "rerank_score",
    "llm_rerank_rank",
    "sentence_window_score",
    "auto_merged",
    "merged_from_count",
]


def _chunk_rows(chunks: list | None) -> list[list]:
    if not chunks:
        return []

    rows: list[list] = []
    for chunk in chunks:
        # FastAPI handlers return pydantic models here.
        item = chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)
        rows.append(
            [
                item.get("id"),
                item.get("distance"),
                item.get("source"),
                item.get("section"),
                item.get("index"),
                item.get("token_count"),
                item.get("initial_rank"),
                item.get("vector_rank"),
                item.get("keyword_rank"),
                item.get("hybrid_score"),
                item.get("rerank_score"),
                item.get("llm_rerank_rank"),
                item.get("sentence_window_score"),
                item.get("auto_merged"),
                item.get("merged_from_count"),
            ]
        )
    return rows


def run_ask(
    question: str,
    top_k: int,
    candidate_k: int,
    use_hybrid_search: bool,
    keyword_candidate_k: int,
    hybrid_alpha: float,
    hybrid_rrf_k: int,
    use_rerank: bool,
    use_llm_rerank: bool,
    llm_rerank_candidate_k: int,
    llm_rerank_keep_k: int,
    use_auto_merging: bool,
    auto_merge_max_gap: int,
    auto_merge_max_chunks: int,
    use_sentence_window: bool,
    sentence_window_size: int,
    model: str,
    include_chunks: bool,
) -> tuple[str, str, list[list]]:
    try:
        response = ask(
            AskRequest(
                question=question,
                top_k=int(top_k),
                candidate_k=int(candidate_k),
                use_hybrid_search=use_hybrid_search,
                keyword_candidate_k=int(keyword_candidate_k),
                hybrid_alpha=float(hybrid_alpha),
                hybrid_rrf_k=int(hybrid_rrf_k),
                use_rerank=use_rerank,
                use_llm_rerank=use_llm_rerank,
                llm_rerank_candidate_k=int(llm_rerank_candidate_k),
                llm_rerank_keep_k=int(llm_rerank_keep_k),
                use_auto_merging=use_auto_merging,
                auto_merge_max_gap=int(auto_merge_max_gap),
                auto_merge_max_chunks=int(auto_merge_max_chunks),
                use_sentence_window=use_sentence_window,
                sentence_window_size=int(sentence_window_size),
                model=model,
                include_chunks=include_chunks,
            )
        )
    except HTTPException as exc:
        return "", json.dumps({"error": exc.detail}, indent=2), []
    except Exception as exc:
        return "", json.dumps({"error": str(exc)}, indent=2), []

    meta = {
        "needs_clarification": response.needs_clarification,
        "retrieved_count": response.retrieved_count,
        "top_k": response.top_k,
        "candidate_k": response.candidate_k,
        "use_hybrid_search": response.use_hybrid_search,
        "keyword_candidate_k": response.keyword_candidate_k,
        "hybrid_alpha": response.hybrid_alpha,
        "hybrid_rrf_k": response.hybrid_rrf_k,
        "use_rerank": response.use_rerank,
        "use_llm_rerank": response.use_llm_rerank,
        "llm_rerank_candidate_k": response.llm_rerank_candidate_k,
        "llm_rerank_keep_k": response.llm_rerank_keep_k,
        "use_auto_merging": response.use_auto_merging,
        "auto_merge_max_gap": response.auto_merge_max_gap,
        "auto_merge_max_chunks": response.auto_merge_max_chunks,
        "use_sentence_window": response.use_sentence_window,
        "sentence_window_size": response.sentence_window_size,
        "model": model,
    }
    rows = _chunk_rows(response.chunks) if include_chunks else []
    return response.answer, json.dumps(meta, indent=2), rows


def run_retrieve(
    question: str,
    top_k: int,
    candidate_k: int,
    use_hybrid_search: bool,
    keyword_candidate_k: int,
    hybrid_alpha: float,
    hybrid_rrf_k: int,
    use_rerank: bool,
    use_llm_rerank: bool,
    llm_rerank_candidate_k: int,
    llm_rerank_keep_k: int,
    use_auto_merging: bool,
    auto_merge_max_gap: int,
    auto_merge_max_chunks: int,
    use_sentence_window: bool,
    sentence_window_size: int,
) -> tuple[str, list[list]]:
    try:
        response = retrieve(
            RetrieveRequest(
                question=question,
                top_k=int(top_k),
                candidate_k=int(candidate_k),
                use_hybrid_search=use_hybrid_search,
                keyword_candidate_k=int(keyword_candidate_k),
                hybrid_alpha=float(hybrid_alpha),
                hybrid_rrf_k=int(hybrid_rrf_k),
                use_rerank=use_rerank,
                use_llm_rerank=use_llm_rerank,
                llm_rerank_candidate_k=int(llm_rerank_candidate_k),
                llm_rerank_keep_k=int(llm_rerank_keep_k),
                use_auto_merging=use_auto_merging,
                auto_merge_max_gap=int(auto_merge_max_gap),
                auto_merge_max_chunks=int(auto_merge_max_chunks),
                use_sentence_window=use_sentence_window,
                sentence_window_size=int(sentence_window_size),
            )
        )
    except HTTPException as exc:
        return json.dumps({"error": exc.detail}, indent=2), []
    except Exception as exc:
        return json.dumps({"error": str(exc)}, indent=2), []

    summary = {
        "retrieved_count": response.retrieved_count,
        "top_k": response.top_k,
        "candidate_k": response.candidate_k,
        "use_hybrid_search": response.use_hybrid_search,
        "keyword_candidate_k": response.keyword_candidate_k,
        "hybrid_alpha": response.hybrid_alpha,
        "hybrid_rrf_k": response.hybrid_rrf_k,
        "use_rerank": response.use_rerank,
        "use_llm_rerank": response.use_llm_rerank,
        "llm_rerank_candidate_k": response.llm_rerank_candidate_k,
        "llm_rerank_keep_k": response.llm_rerank_keep_k,
        "use_auto_merging": response.use_auto_merging,
        "auto_merge_max_gap": response.auto_merge_max_gap,
        "auto_merge_max_chunks": response.auto_merge_max_chunks,
        "use_sentence_window": response.use_sentence_window,
        "sentence_window_size": response.sentence_window_size,
    }
    return json.dumps(summary, indent=2), _chunk_rows(response.chunks)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Health Insurance RAG") as demo:
        gr.Markdown("# Health Insurance RAG")
        gr.Markdown("Ask policy questions or inspect retrieval output.")

        with gr.Tab("Ask"):
            ask_question = gr.Textbox(
                label="Question",
                lines=2,
                placeholder="e.g. What illnesses are covered by this policy?",
            )
            with gr.Row():
                ask_top_k = gr.Slider(
                    label="top_k",
                    minimum=1,
                    maximum=20,
                    value=int(RETRIEVAL_CFG["top_k"]),
                    step=1,
                )
                ask_candidate_k = gr.Slider(
                    label="candidate_k",
                    minimum=1,
                    maximum=100,
                    value=int(RETRIEVAL_CFG["candidate_k"]),
                    step=1,
                )
            with gr.Row():
                ask_use_hybrid_search = gr.Checkbox(
                    label="Use hybrid search",
                    value=bool(RETRIEVAL_CFG["use_hybrid_search"]),
                )
                ask_keyword_candidate_k = gr.Slider(
                    label="keyword_candidate_k",
                    minimum=1,
                    maximum=100,
                    value=int(RETRIEVAL_CFG["keyword_candidate_k"]),
                    step=1,
                )
                ask_hybrid_alpha = gr.Slider(
                    label="hybrid_alpha",
                    minimum=0.0,
                    maximum=1.0,
                    value=float(RETRIEVAL_CFG["hybrid_alpha"]),
                    step=0.05,
                )
                ask_hybrid_rrf_k = gr.Slider(
                    label="hybrid_rrf_k",
                    minimum=1,
                    maximum=200,
                    value=int(RETRIEVAL_CFG["hybrid_rrf_k"]),
                    step=1,
                )
            with gr.Row():
                ask_use_rerank = gr.Checkbox(
                    label="Use rerank", value=bool(RETRIEVAL_CFG["use_rerank"])
                )
                ask_use_llm_rerank = gr.Checkbox(
                    label="Use LLM rerank",
                    value=bool(RETRIEVAL_CFG["use_llm_rerank"]),
                )
                ask_llm_rerank_candidate_k = gr.Slider(
                    label="llm_rerank_candidate_k",
                    minimum=1,
                    maximum=50,
                    value=int(RETRIEVAL_CFG["llm_rerank_candidate_k"]),
                    step=1,
                )
                ask_llm_rerank_keep_k = gr.Slider(
                    label="llm_rerank_keep_k",
                    minimum=1,
                    maximum=20,
                    value=int(RETRIEVAL_CFG["llm_rerank_keep_k"]),
                    step=1,
                )
            with gr.Row():
                ask_use_auto_merging = gr.Checkbox(
                    label="Use auto merging",
                    value=bool(RETRIEVAL_CFG["use_auto_merging"]),
                )
                ask_auto_merge_max_gap = gr.Slider(
                    label="auto_merge_max_gap",
                    minimum=0,
                    maximum=5,
                    value=int(RETRIEVAL_CFG["auto_merge_max_gap"]),
                    step=1,
                )
                ask_auto_merge_max_chunks = gr.Slider(
                    label="auto_merge_max_chunks",
                    minimum=1,
                    maximum=10,
                    value=int(RETRIEVAL_CFG["auto_merge_max_chunks"]),
                    step=1,
                )
            with gr.Row():
                ask_use_sentence_window = gr.Checkbox(
                    label="Use sentence window",
                    value=bool(RETRIEVAL_CFG["use_sentence_window"]),
                )
                ask_sentence_window_size = gr.Slider(
                    label="sentence_window_size",
                    minimum=0,
                    maximum=5,
                    value=int(RETRIEVAL_CFG["sentence_window_size"]),
                    step=1,
                )
            with gr.Row():
                ask_include_chunks = gr.Checkbox(label="Include chunks", value=False)
            ask_model = gr.Textbox(
                label="LLM model", value=str(MODELS_CFG["answer_model"])
            )
            ask_btn = gr.Button("Ask")

            ask_answer = gr.Textbox(label="Answer", lines=10)
            ask_meta = gr.Code(label="Metadata", language="json")
            ask_chunks = gr.Dataframe(
                headers=CHUNK_HEADERS,
                datatype=[
                    "str",
                    "number",
                    "str",
                    "str",
                    "number",
                    "number",
                    "number",
                    "number",
                    "number",
                    "number",
                    "number",
                    "number",
                    "number",
                    "bool",
                    "number",
                ],
                label="Retrieved Chunks",
                interactive=False,
            )

            ask_btn.click(
                run_ask,
                inputs=[
                    ask_question,
                    ask_top_k,
                    ask_candidate_k,
                    ask_use_hybrid_search,
                    ask_keyword_candidate_k,
                    ask_hybrid_alpha,
                    ask_hybrid_rrf_k,
                    ask_use_rerank,
                    ask_use_llm_rerank,
                    ask_llm_rerank_candidate_k,
                    ask_llm_rerank_keep_k,
                    ask_use_auto_merging,
                    ask_auto_merge_max_gap,
                    ask_auto_merge_max_chunks,
                    ask_use_sentence_window,
                    ask_sentence_window_size,
                    ask_model,
                    ask_include_chunks,
                ],
                outputs=[ask_answer, ask_meta, ask_chunks],
            )

        with gr.Tab("Retrieve"):
            retrieve_question = gr.Textbox(
                label="Question",
                lines=2,
                placeholder="e.g. What are the main exclusions?",
            )
            with gr.Row():
                retrieve_top_k = gr.Slider(
                    label="top_k",
                    minimum=1,
                    maximum=20,
                    value=int(RETRIEVAL_CFG["top_k"]),
                    step=1,
                )
                retrieve_candidate_k = gr.Slider(
                    label="candidate_k",
                    minimum=1,
                    maximum=100,
                    value=int(RETRIEVAL_CFG["candidate_k"]),
                    step=1,
                )
            with gr.Row():
                retrieve_use_hybrid_search = gr.Checkbox(
                    label="Use hybrid search",
                    value=bool(RETRIEVAL_CFG["use_hybrid_search"]),
                )
                retrieve_keyword_candidate_k = gr.Slider(
                    label="keyword_candidate_k",
                    minimum=1,
                    maximum=100,
                    value=int(RETRIEVAL_CFG["keyword_candidate_k"]),
                    step=1,
                )
                retrieve_hybrid_alpha = gr.Slider(
                    label="hybrid_alpha",
                    minimum=0.0,
                    maximum=1.0,
                    value=float(RETRIEVAL_CFG["hybrid_alpha"]),
                    step=0.05,
                )
                retrieve_hybrid_rrf_k = gr.Slider(
                    label="hybrid_rrf_k",
                    minimum=1,
                    maximum=200,
                    value=int(RETRIEVAL_CFG["hybrid_rrf_k"]),
                    step=1,
                )
            with gr.Row():
                retrieve_use_rerank = gr.Checkbox(
                    label="Use rerank", value=bool(RETRIEVAL_CFG["use_rerank"])
                )
                retrieve_use_llm_rerank = gr.Checkbox(
                    label="Use LLM rerank",
                    value=bool(RETRIEVAL_CFG["use_llm_rerank"]),
                )
                retrieve_llm_rerank_candidate_k = gr.Slider(
                    label="llm_rerank_candidate_k",
                    minimum=1,
                    maximum=50,
                    value=int(RETRIEVAL_CFG["llm_rerank_candidate_k"]),
                    step=1,
                )
                retrieve_llm_rerank_keep_k = gr.Slider(
                    label="llm_rerank_keep_k",
                    minimum=1,
                    maximum=20,
                    value=int(RETRIEVAL_CFG["llm_rerank_keep_k"]),
                    step=1,
                )
            with gr.Row():
                retrieve_use_auto_merging = gr.Checkbox(
                    label="Use auto merging",
                    value=bool(RETRIEVAL_CFG["use_auto_merging"]),
                )
                retrieve_auto_merge_max_gap = gr.Slider(
                    label="auto_merge_max_gap",
                    minimum=0,
                    maximum=5,
                    value=int(RETRIEVAL_CFG["auto_merge_max_gap"]),
                    step=1,
                )
                retrieve_auto_merge_max_chunks = gr.Slider(
                    label="auto_merge_max_chunks",
                    minimum=1,
                    maximum=10,
                    value=int(RETRIEVAL_CFG["auto_merge_max_chunks"]),
                    step=1,
                )
            with gr.Row():
                retrieve_use_sentence_window = gr.Checkbox(
                    label="Use sentence window",
                    value=bool(RETRIEVAL_CFG["use_sentence_window"]),
                )
                retrieve_sentence_window_size = gr.Slider(
                    label="sentence_window_size",
                    minimum=0,
                    maximum=5,
                    value=int(RETRIEVAL_CFG["sentence_window_size"]),
                    step=1,
                )
            retrieve_btn = gr.Button("Retrieve")

            retrieve_meta = gr.Code(label="Summary", language="json")
            retrieve_chunks = gr.Dataframe(
                headers=CHUNK_HEADERS,
                datatype=[
                    "str",
                    "number",
                    "str",
                    "str",
                    "number",
                    "number",
                    "number",
                    "number",
                    "number",
                    "number",
                    "number",
                    "number",
                    "number",
                    "bool",
                    "number",
                ],
                label="Retrieved Chunks",
                interactive=False,
            )

            retrieve_btn.click(
                run_retrieve,
                inputs=[
                    retrieve_question,
                    retrieve_top_k,
                    retrieve_candidate_k,
                    retrieve_use_hybrid_search,
                    retrieve_keyword_candidate_k,
                    retrieve_hybrid_alpha,
                    retrieve_hybrid_rrf_k,
                    retrieve_use_rerank,
                    retrieve_use_llm_rerank,
                    retrieve_llm_rerank_candidate_k,
                    retrieve_llm_rerank_keep_k,
                    retrieve_use_auto_merging,
                    retrieve_auto_merge_max_gap,
                    retrieve_auto_merge_max_chunks,
                    retrieve_use_sentence_window,
                    retrieve_sentence_window_size,
                ],
                outputs=[retrieve_meta, retrieve_chunks],
            )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gradio UI for Health Insurance RAG.")
    parser.add_argument("--host", default=str(GRADIO_CFG["host"]))
    parser.add_argument("--port", type=int, default=int(GRADIO_CFG["port"]))
    parser.add_argument(
        "--share",
        action=argparse.BooleanOptionalAction,
        default=bool(GRADIO_CFG["share"]),
    )
    args = parser.parse_args()

    demo = build_demo()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
