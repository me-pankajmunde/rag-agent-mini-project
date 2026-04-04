"""
RAG-Based AI Assistant – Streamlit UI
"""

import os
import sys
import tempfile
import streamlit as st
from dotenv import load_dotenv

# Support both `streamlit run src/rag_agent/app.py` and `python -m rag_agent.app`
try:
    from . import rag_engine
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from rag_agent import rag_engine

load_dotenv()

# ── Page config (must be the very first Streamlit call) ───────────────────────

st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Hide Streamlit chrome ─────────────────────────────────────────────────────

st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; max-width: 100% !important; }
</style>
""", unsafe_allow_html=True)

# ── Cached resources (initialised once per process) ───────────────────────────

@st.cache_resource(show_spinner="Loading embedding model…")
def _load_model():
    return rag_engine.get_embedding_model()


@st.cache_resource(show_spinner="Connecting to vector DB…")
def _load_collection():
    return rag_engine.get_vector_db()


model = _load_model()
collection = _load_collection()

# ── Session-state defaults ────────────────────────────────────────────────────

_DEFAULTS: dict = {
    "chat_history":        [],
    "sources_html":        "",
    "preview_doc":         None,
    "active_doc":          None,   # doc currently scoping retrieval
    "preview_excerpt":     "",
    "preview_chunk_count": 0,
    "preview_word_count":  0,
    "preview_char_count":  0,
    "preview_chunks_md":   "",
    "index_status":        None,   # {"type": "success"|"error"|"warning", "msg": str}
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Helpers ───────────────────────────────────────────────────────────────────

def _confidence(distance: float) -> str:
    if distance <= 0.25:
        return "High"
    if distance <= 0.45:
        return "Medium"
    return "Low"


def _build_sources_md(chunks: list) -> str:
    if not chunks:
        return ""
    parts = []
    for i, src in enumerate(chunks, 1):
        conf = _confidence(src["distance"])
        dot  = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(conf, "")
        snippet = src["text"][:400] + ("…" if len(src["text"]) > 400 else "")
        parts.append(
            f"**Source {i} — {src['source']}**  \n"
            f"{dot} Confidence: **{conf}** · Distance: `{src['distance']}`\n\n"
            f"{snippet}"
        )
    return "\n\n---\n\n".join(parts)


def _build_chunk_samples_md(chunks: list) -> str:
    if not chunks:
        return ""
    parts = []
    for i, c in enumerate(chunks[:3], 1):
        preview = c[:350] + ("…" if len(c) > 350 else "")
        parts.append(f"**Chunk {i}**\n\n> {preview}")
    return "\n\n---\n\n".join(parts)


def _set_preview(doc_name: str, excerpt: str, chunk_count: int,
                 chunks_md: str, word_count: int = 0, char_count: int = 0):
    st.session_state.preview_doc         = doc_name
    st.session_state.preview_excerpt     = excerpt
    st.session_state.preview_chunk_count = chunk_count
    st.session_state.preview_word_count  = word_count
    st.session_state.preview_char_count  = char_count
    st.session_state.preview_chunks_md   = chunks_md


def _load_preview_from_db(doc_name: str):
    try:
        data  = rag_engine.get_document_preview(collection, doc_name)
        _set_preview(
            doc_name,
            excerpt     = data["excerpt"],
            chunk_count = data["chunk_count"],
            chunks_md   = _build_chunk_samples_md(data["sample_chunks"]),
        )
    except Exception as e:
        _set_preview(doc_name, excerpt=f"Error loading preview: {e}", chunk_count=0, chunks_md="")


# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:

    st.title("📚 RAG AI Assistant")
    st.caption("Retrieval-Augmented Generation · Document Q&A")
    st.divider()

    # ── Configuration ──────────────────────────────────────────────────────────
    st.subheader("⚙ Configuration")

    top_k = st.slider("Top-K Retrieval", 1, 10, 5,
                      help="Most relevant chunks retrieved per question.")
    chunk_size = st.slider("Chunk Size", 200, 1000, 500, step=50,
                           help="Characters per chunk during indexing.")
    max_overlap = max(20, chunk_size // 2)
    overlap = st.slider("Chunk Overlap", 20, max_overlap, min(50, max_overlap), step=10,
                        help="Character overlap between consecutive chunks.")

    env_key = os.getenv("OPENAI_API_KEY", "")
    if not env_key:
        api_key = st.text_input("OpenAI API Key", type="password",
                                placeholder="sk-…  (platform.openai.com)",
                                key="api_key_input")
    else:
        api_key = env_key

    # ── Upload ──────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📤 Upload Document")

    uploaded_file = st.file_uploader(
        "PDF or TXT", type=["pdf", "txt"], label_visibility="collapsed"
    )

    if st.button("Index Document", type="primary", use_container_width=True):
        if uploaded_file is None:
            st.session_state.index_status = {"type": "warning", "msg": "⚠️ No file selected."}
        else:
            resolved_key = os.getenv("OPENAI_API_KEY", "") or api_key
            if resolved_key:
                os.environ["OPENAI_API_KEY"] = resolved_key
            try:
                suffix = os.path.splitext(uploaded_file.name)[1]
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    text = rag_engine.load_document(tmp_path)
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

                if not text:
                    st.session_state.index_status = {
                        "type": "error",
                        "msg": "❌ Could not extract text from the file.",
                    }
                else:
                    chunks = rag_engine.chunk_text(text, chunk_size=chunk_size, overlap=overlap)
                    count  = rag_engine.add_document(collection, model, chunks, uploaded_file.name)
                    _set_preview(
                        uploaded_file.name,
                        excerpt     = text[:1500].strip(),
                        chunk_count = count,
                        chunks_md   = _build_chunk_samples_md(chunks[:3]),
                        word_count  = len(text.split()),
                        char_count  = len(text),
                    )
                    st.session_state.index_status = {
                        "type": "success",
                        "msg": f"✅ Indexed **{count}** chunks from **{uploaded_file.name}**",
                    }
            except Exception as e:
                st.session_state.index_status = {"type": "error", "msg": f"❌ {e}"}

    if st.session_state.index_status:
        s = st.session_state.index_status
        if s["type"] == "success":
            st.success(s["msg"])
        elif s["type"] == "error":
            st.error(s["msg"])
        else:
            st.warning(s["msg"])

    # ── Indexed documents ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("📚 Indexed Documents")

    docs = rag_engine.list_indexed_documents(collection)
    if docs:
        selected_doc = st.radio(
            "Select to preview",
            options=docs,
            label_visibility="collapsed",
            key="doc_radio",
        )
        # Load preview when selection changes
        if selected_doc != st.session_state.preview_doc:
            _load_preview_from_db(selected_doc)

        # Switching the active doc clears chat so history isn't from the old scope
        if selected_doc != st.session_state.active_doc:
            st.session_state.active_doc = selected_doc
            st.session_state.chat_history = []
            st.session_state.sources_html = ""

        st.caption(
            f"{len(docs)} document{'s' if len(docs) != 1 else ''}"
            f" · {collection.count()} total chunks"
        )
    else:
        st.caption("No documents indexed yet. Upload a file above.")

    if st.button("🗑️ Clear All Data", use_container_width=True):
        deleted = rag_engine.clear_indexed_documents(collection)
        for k, v in _DEFAULTS.items():
            st.session_state[k] = v
        st.session_state.index_status = {
            "type": "success", "msg": f"🗑️ Cleared {deleted} indexed chunks."
        }
        st.rerun()

    st.markdown("---")
    st.caption("Built with **Streamlit** · **ChromaDB** · **OpenAI**")

# ── MAIN AREA ─────────────────────────────────────────────────────────────────

st.title("💬 RAG AI Assistant")
st.caption("Ask questions grounded in your indexed documents · ChromaDB · OpenAI · Streamlit")
if st.session_state.active_doc:
    st.info(f"🔍 Searching in: **{st.session_state.active_doc}**")
st.divider()

# Render accumulated chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input + streaming ────────────────────────────────────────────────────

if prompt := st.chat_input("Ask a question about your documents…"):
    resolved_key = os.getenv("OPENAI_API_KEY", "") or api_key
    if not resolved_key:
        st.warning("Please enter your OpenAI API key in the sidebar ⚙ Configuration section.")
    elif collection.count() == 0:
        st.warning("No documents indexed yet. Upload a file and click **Index Document** first.")
    else:
        os.environ["OPENAI_API_KEY"] = resolved_key

        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        chunks = rag_engine.search_similar(
            collection, model, prompt,
            n_results=top_k,
            source_filter=st.session_state.active_doc or None,
        )

        with st.chat_message("assistant"):
            if not chunks:
                answer = "I couldn't find relevant content for that question in the indexed documents."
                st.markdown(answer)
            else:
                context = "\n\n".join(
                    f"[{i}. From: {c['source']}]\n{c['text']}"
                    for i, c in enumerate(chunks, 1)
                )
                rag_history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.chat_history[:-1]   # exclude the just-added user msg
                ]
                try:
                    answer = st.write_stream(
                        rag_engine.ask_openai_stream(resolved_key, context, prompt, rag_history)
                    )
                except Exception as e:
                    answer = f"❌ Error calling OpenAI API: {e}"
                    st.markdown(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        if chunks:
            st.session_state.sources_html = _build_sources_md(chunks)

# ── Info tabs (rendered every run, after chat block so sources are current) ───

st.divider()
tab_sources, tab_preview = st.tabs(["🔍 Retrieved Sources", "📄 Document Preview"])

with tab_sources:
    if st.session_state.sources_html:
        st.markdown(st.session_state.sources_html)
    else:
        st.caption("Ask a question to see the retrieved source chunks here.")

with tab_preview:
    if st.session_state.preview_doc:
        st.markdown(f"📄 **{st.session_state.preview_doc}**")
        cols = st.columns(3 if st.session_state.preview_word_count else 1)
        cols[0].metric("Chunks", st.session_state.preview_chunk_count)
        if st.session_state.preview_word_count:
            cols[1].metric("Words", f"{st.session_state.preview_word_count:,}")
            cols[2].metric("Chars", f"{st.session_state.preview_char_count:,}")
        st.text_area(
            "Document Excerpt",
            value=st.session_state.preview_excerpt,
            height=200,
            disabled=True,
        )
        if st.session_state.preview_chunks_md:
            st.caption("SAMPLE CHUNKS")
            st.markdown(st.session_state.preview_chunks_md)
    else:
        st.caption("Index a document to see a text preview here.")


