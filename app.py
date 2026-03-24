"""
RAG-Based AI Assistant
A simple Retrieval-Augmented Generation system for document Q&A.
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
import rag_engine

# Load environment variables from .env file
load_dotenv()

# ── Page Configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG AI Assistant")
st.caption("Upload documents and ask questions — answers are grounded in your files.")

# ── Cached Resources (loaded once, not on every click) ────────────────────────

@st.cache_resource
def load_model():
    """Load the embedding model once."""
    with st.spinner("Loading embedding model for the first time..."):
        return rag_engine.get_embedding_model()

@st.cache_resource
def load_db():
    """Connect to ChromaDB once."""
    return rag_engine.get_vector_db()

model = load_model()
collection = load_db()

# ── Session State ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": ..., "content": ...}

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Upload Documents")

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        api_key = st.text_input("Anthropic API Key", type="password",
                                help="Get your key from console.anthropic.com")
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key

    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=["pdf", "txt"],
        help="Upload a document to ask questions about"
    )

    if uploaded_file is not None:
        if st.button("Index Document", type="primary"):
            with st.spinner(f"Processing {uploaded_file.name}..."):
                # Save to a temp file so we can read it
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                try:
                    text = rag_engine.load_document(tmp_path)
                    if not text:
                        st.error("Could not extract text from the file.")
                    else:
                        chunks = rag_engine.chunk_text(text)
                        count = rag_engine.add_document(collection, model, chunks, uploaded_file.name)
                        st.success(f"Indexed {count} chunks from **{uploaded_file.name}**")
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    os.unlink(tmp_path)  # clean up temp file

    st.divider()

    # Show indexed documents
    st.subheader("Indexed Documents")
    docs = rag_engine.list_indexed_documents(collection)
    if docs:
        for doc in docs:
            st.write(f"- {doc}")
        st.caption(f"Total chunks in DB: {collection.count()}")
    else:
        st.info("No documents indexed yet. Upload a file above.")

    st.divider()
    st.caption("Built with Streamlit + ChromaDB + Claude")

# ── Main Chat Area ────────────────────────────────────────────────────────────

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("Sources used"):
                for src in msg["sources"]:
                    st.caption(f"**{src['source']}** (distance: {src['distance']})\n\n{src['text'][:300]}...")

# Chat input
question = st.chat_input("Ask a question about your documents...")

if question:
    # Validate prerequisites
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.error("Please enter your Anthropic API key in the sidebar.")
        st.stop()

    if collection.count() == 0:
        st.warning("Please upload and index at least one document first.")
        st.stop()

    # Show user message
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Retrieve relevant chunks
    with st.spinner("Searching documents..."):
        chunks = rag_engine.search_similar(collection, model, question, n_results=5)

    if not chunks:
        answer = "I couldn't find any relevant information in the indexed documents."
        sources = []
    else:
        # Format context for Claude
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[{i}. From: {chunk['source']}]\n{chunk['text']}")
        context = "\n\n".join(context_parts)

        # Build history for multi-turn (exclude source metadata)
        history = [{"role": m["role"], "content": m["content"]}
                   for m in st.session_state.messages[:-1]]  # exclude current question

        # Ask Claude
        with st.spinner("Generating answer..."):
            try:
                answer = rag_engine.ask_claude(api_key, context, question, history)
            except Exception as e:
                answer = f"Error calling Claude API: {e}"
        sources = chunks

    # Show assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)
        if sources:
            with st.expander("Sources used"):
                for src in sources:
                    st.caption(f"**{src['source']}** (distance: {src['distance']})\n\n{src['text'][:300]}...")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
