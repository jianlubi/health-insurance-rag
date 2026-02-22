from __future__ import annotations

import argparse
import json

from fastapi import HTTPException
import gradio as gr

from api import AskRequest, RetrieveRequest, ask, retrieve


CHUNK_HEADERS = [
    "id",
    "distance",
    "source",
    "section",
    "index",
    "token_count",
    "initial_rank",
    "rerank_score",
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
                item.get("rerank_score"),
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
    use_rerank: bool,
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
                use_rerank=use_rerank,
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
        "use_rerank": response.use_rerank,
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
    use_rerank: bool,
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
                use_rerank=use_rerank,
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
        "use_rerank": response.use_rerank,
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
                ask_top_k = gr.Slider(label="top_k", minimum=1, maximum=20, value=4, step=1)
                ask_candidate_k = gr.Slider(
                    label="candidate_k",
                    minimum=1,
                    maximum=100,
                    value=12,
                    step=1,
                )
            with gr.Row():
                ask_use_rerank = gr.Checkbox(label="Use rerank", value=False)
                ask_use_auto_merging = gr.Checkbox(
                    label="Use auto merging", value=False
                )
                ask_auto_merge_max_gap = gr.Slider(
                    label="auto_merge_max_gap",
                    minimum=0,
                    maximum=5,
                    value=1,
                    step=1,
                )
                ask_auto_merge_max_chunks = gr.Slider(
                    label="auto_merge_max_chunks",
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                )
            with gr.Row():
                ask_use_sentence_window = gr.Checkbox(
                    label="Use sentence window", value=False
                )
                ask_sentence_window_size = gr.Slider(
                    label="sentence_window_size",
                    minimum=0,
                    maximum=5,
                    value=1,
                    step=1,
                )
            with gr.Row():
                ask_include_chunks = gr.Checkbox(label="Include chunks", value=False)
            ask_model = gr.Textbox(label="LLM model", value="gpt-4o-mini")
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
                    ask_use_rerank,
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
                    label="top_k", minimum=1, maximum=20, value=4, step=1
                )
                retrieve_candidate_k = gr.Slider(
                    label="candidate_k",
                    minimum=1,
                    maximum=100,
                    value=12,
                    step=1,
                )
            with gr.Row():
                retrieve_use_rerank = gr.Checkbox(label="Use rerank", value=False)
                retrieve_use_auto_merging = gr.Checkbox(
                    label="Use auto merging", value=False
                )
                retrieve_auto_merge_max_gap = gr.Slider(
                    label="auto_merge_max_gap",
                    minimum=0,
                    maximum=5,
                    value=1,
                    step=1,
                )
                retrieve_auto_merge_max_chunks = gr.Slider(
                    label="auto_merge_max_chunks",
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                )
            with gr.Row():
                retrieve_use_sentence_window = gr.Checkbox(
                    label="Use sentence window", value=False
                )
                retrieve_sentence_window_size = gr.Slider(
                    label="sentence_window_size",
                    minimum=0,
                    maximum=5,
                    value=1,
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
                    retrieve_use_rerank,
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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_demo()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
