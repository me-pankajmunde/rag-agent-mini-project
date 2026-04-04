"""
RAG Engine - Core logic for document processing, embedding, and retrieval.
"""

import os
import shutil
from datetime import datetime
import uuid
import chromadb
from chromadb.config import Settings
from chromadb.telemetry.product import ProductTelemetryClient, ProductTelemetryEvent
from overrides import override
import PyPDF2
from sentence_transformers import SentenceTransformer
from openai import OpenAI


DEFAULT_OPENAI_MODEL = "gpt-4o"


class NoOpTelemetry(ProductTelemetryClient):
    """Disable Chroma product telemetry to avoid runtime posthog issues."""

    @override
    def capture(self, event: ProductTelemetryEvent) -> None:
        return


def get_openai_model_name() -> str:
    """Return OpenAI model name from env or default."""
    return os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)


# ── Document Loading ──────────────────────────────────────────────────────────

def extract_text_from_pdf(file_path: str) -> str:
    """Read all text from a PDF file."""
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def extract_text_from_txt(file_path: str) -> str:
    """Read all text from a plain text file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def load_document(file_path: str) -> str:
    """Load document text based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".txt":
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Please upload PDF or TXT files.")


# ── Text Chunking ─────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Split text into overlapping chunks.
    Returns a list of text strings.
    """
    chunks = []
    step = chunk_size - overlap
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if len(chunk) > 20:  # skip very short/empty chunks
            chunks.append(chunk)
        start += step
    return chunks


# ── Embedding Model ───────────────────────────────────────────────────────────

def get_embedding_model():
    """Load the sentence-transformers model (384-dim embeddings)."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(model, texts: list) -> list:
    """Embed a list of texts and return as plain Python list."""
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


# ── Vector Database ───────────────────────────────────────────────────────────

def get_vector_db(persist_dir: str = "./data/chroma_db"):
    """Connect to (or create) the ChromaDB collection.

    If an incompatible persisted configuration is detected (for example
    KeyError: '_type' after Chroma upgrades), backup the old directory and
    recreate a fresh store so the app can still start.
    """
    os.makedirs(persist_dir, exist_ok=True)
    def _rotate_persist_dir() -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"{persist_dir}_backup_{timestamp}"
        if os.path.exists(persist_dir):
            shutil.move(persist_dir, backup_dir)
        os.makedirs(persist_dir, exist_ok=True)
        print(
            "Detected incompatible ChromaDB persisted config. "
            f"Moved old DB to: {backup_dir}. A new empty DB has been created."
        )
        return backup_dir

    settings = Settings(
        anonymized_telemetry=False,
        chroma_product_telemetry_impl="rag_agent.rag_engine.NoOpTelemetry",
        chroma_telemetry_impl="rag_agent.rag_engine.NoOpTelemetry",
    )

    for attempt in range(2):
        try:
            client = chromadb.PersistentClient(path=persist_dir, settings=settings)
            return client.get_or_create_collection(
                name="rag_documents",
                metadata={"hnsw:space": "cosine"}
            )
        except KeyError as e:
            if "_type" not in str(e) or attempt == 1:
                if "_type" in str(e):
                    print(
                        "ChromaDB persistent mode is incompatible in this environment. "
                        "Falling back to in-memory storage for this session."
                    )
                    client = chromadb.EphemeralClient(settings=settings)
                    return client.get_or_create_collection(
                        name="rag_documents",
                        metadata={"hnsw:space": "cosine"}
                    )
                raise
            _rotate_persist_dir()

    raise RuntimeError("Unable to initialize ChromaDB collection")


def add_document(collection, model, chunks: list, doc_name: str) -> int:
    """
    Embed chunks and store them in ChromaDB.
    Returns the number of chunks added.
    """
    if not chunks:
        return 0

    doc_id = str(uuid.uuid4())[:8]
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    embeddings = embed_texts(model, chunks)
    metadatas = [{"source": doc_name, "doc_id": doc_id, "chunk_index": i}
                 for i in range(len(chunks))]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )
    return len(chunks)


def search_similar(collection, model, query: str, n_results: int = 5,
                   source_filter: str | None = None) -> list:
    """
    Search for the most relevant chunks for a given query.
    When source_filter is set, retrieval is scoped to that document only.
    Returns a list of dicts with 'text', 'source', and 'distance'.
    """
    total = collection.count()
    if total == 0:
        return []

    # Count chunks available for this scope so n never exceeds it.
    if source_filter:
        scoped = collection.get(where={"source": source_filter}, include=[])
        available = len(scoped.get("ids", []))
    else:
        available = total

    if available == 0:
        return []

    n = min(n_results, available)
    query_embedding = embed_texts(model, [query])[0]

    query_kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": n,
        "include": ["documents", "metadatas", "distances"],
    }
    if source_filter:
        query_kwargs["where"] = {"source": source_filter}

    results = collection.query(**query_kwargs)

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "text": doc,
            "source": meta.get("source", "Unknown"),
            "distance": round(dist, 4)
        })
    return chunks


def list_indexed_documents(collection) -> list:
    """Return unique document names currently stored in the DB."""
    if collection.count() == 0:
        return []
    all_items = collection.get(include=["metadatas"])
    sources = set()
    for meta in all_items["metadatas"]:
        sources.add(meta.get("source", "Unknown"))
    return sorted(sources)


def get_document_preview(collection, doc_name: str, n_chunks: int = 3) -> dict:
    """Fetch stored chunks for a named document and return preview data.

    Returns a dict with:
        chunk_count   – total number of stored chunks for this document
        excerpt       – text of the first chunk (up to 1500 chars)
        sample_chunks – list of up to *n_chunks* chunk strings
    """
    if collection.count() == 0:
        return {"chunk_count": 0, "excerpt": "", "sample_chunks": []}

    results = collection.get(
        where={"source": doc_name},
        include=["documents", "metadatas"],
    )

    docs = results.get("documents") or []
    metadatas = results.get("metadatas") or []

    # Re-order chunks by their original index so the excerpt is the doc start.
    paired = sorted(zip(metadatas, docs), key=lambda x: x[0].get("chunk_index", 0))
    chunks = [doc for _, doc in paired]

    excerpt = chunks[0][:1500] if chunks else ""
    return {
        "chunk_count": len(chunks),
        "excerpt": excerpt,
        "sample_chunks": chunks[:n_chunks],
    }


def clear_indexed_documents(collection) -> int:
    """Delete all indexed chunks and return number of deleted items."""
    total = collection.count()
    if total == 0:
        return 0

    all_items = collection.get(include=[])
    ids = all_items.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    return len(ids)


# ── LLM (OpenAI) ─────────────────────────────────────────────────────────────

def _build_messages(context: str, question: str, history: list) -> list:
    """Build the OpenAI messages list from context, history, and the current question."""
    system_prompt = (
        "You are a helpful AI assistant that answers questions based on the provided document context.\n"
        "Use ONLY the information given in the context to answer the question.\n"
        "If the context does not contain enough information, say so clearly.\n"
        "Do not mention document names or file names in your answer.\n\n"
        f"Context from documents:\n{context}"
    )
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})
    return messages


def ask_openai(api_key: str, context: str, question: str, history: list) -> str:
    """
    Send the question + retrieved context to OpenAI and return the answer.

    history: list of {"role": "user"|"assistant", "content": str}
    """
    client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_API_BASE_URL", "https://ai.jdfy.in/v1"))
    messages = _build_messages(context, question, history)
    response = client.chat.completions.create(
        model=get_openai_model_name(),
        messages=messages,
        max_tokens=1024,
    )
    return response.choices[0].message.content


def ask_openai_stream(api_key: str, context: str, question: str, history: list):
    """
    Send the question + retrieved context to OpenAI and stream the answer.
    Yields text chunks as they arrive.
    """
    client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_API_BASE_URL", "https://ai.jdfy.in/v1") )
    messages = _build_messages(context, question, history)
    stream = client.chat.completions.create(
        model=get_openai_model_name(),
        messages=messages,
        max_tokens=1024,
        stream=True,
    )
    for chunk in stream:
        # Some providers can emit housekeeping chunks with empty choices.
        if not getattr(chunk, "choices", None):
            continue
        delta = getattr(chunk.choices[0], "delta", None)
        text = getattr(delta, "content", None) if delta is not None else None
        if text:
            yield text
