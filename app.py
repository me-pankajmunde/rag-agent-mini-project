"""
RAG-Based AI Assistant
A simple Retrieval-Augmented Generation system for document Q&A.
"""

import os
import gradio as gr
from dotenv import load_dotenv
import rag_engine

# Load environment variables from .env file
load_dotenv()

# ── Global Resources (loaded once at startup) ─────────────────────────────────

model = rag_engine.get_embedding_model()
collection = rag_engine.get_vector_db()


def distance_to_confidence(distance: float) -> str:
    """Map vector distance to a simple confidence label for demo readability."""
    if distance <= 0.25:
        return "High"
    if distance <= 0.45:
        return "Medium"
    return "Low"


# ── Helper ────────────────────────────────────────────────────────────────────

def get_indexed_docs_text() -> str:
    """Return a formatted markdown string of indexed documents."""
    docs = rag_engine.list_indexed_documents(collection)
    if docs:
        doc_list = "\n".join(f"- {doc}" for doc in docs)
        total = collection.count()
        return f"{doc_list}\n\n*Total chunks in DB: {total}*"
    return "No documents indexed yet. Upload a file above."


# ── Event Handlers ────────────────────────────────────────────────────────────

def update_overlap_max(chunk_size: int):
    """Dynamically cap the overlap slider when chunk size changes."""
    max_overlap = max(20, chunk_size // 2)
    return gr.update(maximum=max_overlap, value=min(50, max_overlap))


def index_document(file, chunk_size: int, overlap: int, api_key_input: str):
    """Index the uploaded document into ChromaDB."""
    if file is None:
        return "No file uploaded.", get_indexed_docs_text()

    api_key = os.getenv("GEMINI_API_KEY", "") or api_key_input
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key

    try:
        text = rag_engine.load_document(file.name)
        if not text:
            return "Could not extract text from the file.", get_indexed_docs_text()

        chunks = rag_engine.chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        doc_name = os.path.basename(file.name)
        count = rag_engine.add_document(collection, model, chunks, doc_name)
        return (
            f"Indexed **{count}** chunks from **{doc_name}** "
            f"(chunk_size={chunk_size}, overlap={overlap})",
            get_indexed_docs_text(),
        )
    except Exception as e:
        return f"Error: {e}", get_indexed_docs_text()


def clear_data():
    """Delete all indexed documents and reset the chat."""
    deleted = rag_engine.clear_indexed_documents(collection)
    return f"Cleared **{deleted}** indexed chunks and reset chat history.", get_indexed_docs_text(), []


def chat(message: str, history: list, top_k: int, api_key_input: str):
    """Handle a chat message and stream the response."""
    api_key = os.getenv("GEMINI_API_KEY", "") or api_key_input
    if not api_key:
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Please enter your Gemini API key in the sidebar."},
        ]
        return

    os.environ["GEMINI_API_KEY"] = api_key

    if collection.count() == 0:
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Please upload and index at least one document first."},
        ]
        return

    chunks = rag_engine.search_similar(collection, model, message, n_results=top_k)

    new_history = history + [{"role": "user", "content": message}]

    if not chunks:
        yield new_history + [
            {
                "role": "assistant",
                "content": "I couldn't find relevant information in the indexed documents for that question.",
            }
        ]
        return

    context_parts = [f"[{i}. From: {c['source']}]\n{c['text']}" for i, c in enumerate(chunks, 1)]
    context = "\n\n".join(context_parts)

    rag_history = [{"role": m["role"], "content": m["content"]} for m in history]

    answer = ""
    try:
        for text_chunk in rag_engine.ask_gemini_stream(api_key, context, message, rag_history):
            answer += text_chunk
            yield new_history + [{"role": "assistant", "content": answer}]
    except Exception as e:
        answer = f"Error calling Gemini API: {e}"
        yield new_history + [{"role": "assistant", "content": answer}]
        return

    # Append sources below the answer
    sources_md = "\n\n---\n**Sources used:**\n"
    for src in chunks:
        confidence = distance_to_confidence(src["distance"])
        sources_md += (
            f"\n**{src['source']}** | Confidence: **{confidence}** | Distance: {src['distance']}\n\n"
            f"{src['text'][:300]}...\n"
        )
    yield new_history + [{"role": "assistant", "content": answer + sources_md}]


def submit_message(message: str, history: list, top_k: int, api_key_input: str):
    """Wrapper so the text box is cleared after submission."""
    if not message.strip():
        yield history
        return
    for updated_history in chat(message, history, top_k, api_key_input):
        yield updated_history


# ── UI Layout ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="RAG AI Assistant") as demo:
    gr.Markdown("# RAG AI Assistant\nUpload documents and ask questions — answers are grounded in your files.")

    with gr.Row():
        # ── Left panel ────────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("## Upload Documents")

            gr.Markdown("### RAG Settings")
            top_k = gr.Slider(
                minimum=1, maximum=10, value=5, step=1,
                label="Top-K Retrieval",
                info="Number of most relevant chunks retrieved for each question",
            )
            chunk_size = gr.Slider(
                minimum=200, maximum=1000, value=500, step=50,
                label="Chunk Size",
                info="Number of characters per chunk during indexing",
            )
            overlap = gr.Slider(
                minimum=20, maximum=250, value=50, step=10,
                label="Chunk Overlap",
                info="Overlap between consecutive chunks during indexing",
            )

            env_key = os.getenv("GEMINI_API_KEY", "")
            api_key_input = gr.Textbox(
                label="Gemini API Key",
                type="password",
                placeholder="Get your key from aistudio.google.com",
                value=env_key,
                visible=not bool(env_key),
            )

            file_upload = gr.File(
                label="Choose a PDF or TXT file",
                file_types=[".pdf", ".txt"],
            )
            index_btn = gr.Button("Index Document", variant="primary")
            index_status = gr.Markdown("")

            gr.Markdown("---")
            gr.Markdown("### Indexed Documents")
            docs_display = gr.Markdown(get_indexed_docs_text())
            clear_btn = gr.Button("Clear Indexed Data", variant="stop")

            gr.Markdown("---")
            gr.Markdown("*Built with Gradio + ChromaDB + Gemini*")

        # ── Right panel (chat) ────────────────────────────────────────────────
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="RAG Chat",
                height=520,
                type="messages",
                show_copy_button=True,
            )
            msg_input = gr.Textbox(
                label="Ask a question about your documents...",
                placeholder="Type your question and press Enter or click Send",
                lines=2,
            )
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                clear_chat_btn = gr.Button("Clear Chat")

    # ── Wire up events ────────────────────────────────────────────────────────

    chunk_size.change(fn=update_overlap_max, inputs=[chunk_size], outputs=[overlap])

    index_btn.click(
        fn=index_document,
        inputs=[file_upload, chunk_size, overlap, api_key_input],
        outputs=[index_status, docs_display],
    )

    clear_btn.click(
        fn=clear_data,
        outputs=[index_status, docs_display, chatbot],
    )

    send_btn.click(
        fn=submit_message,
        inputs=[msg_input, chatbot, top_k, api_key_input],
        outputs=[chatbot],
    ).then(fn=lambda: "", outputs=[msg_input])

    msg_input.submit(
        fn=submit_message,
        inputs=[msg_input, chatbot, top_k, api_key_input],
        outputs=[chatbot],
    ).then(fn=lambda: "", outputs=[msg_input])

    clear_chat_btn.click(fn=lambda: [], outputs=[chatbot])


if __name__ == "__main__":
    demo.launch()
