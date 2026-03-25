"""
RAG Engine - Core logic for document processing, embedding, and retrieval.
"""

import os
import uuid
import chromadb
import PyPDF2
from sentence_transformers import SentenceTransformer
import google.generativeai as genai


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

def get_vector_db(persist_dir: str = "./chroma_db"):
    """Connect to (or create) the ChromaDB collection."""
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name="rag_documents",
        metadata={"hnsw:space": "cosine"}
    )
    return collection


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


def search_similar(collection, model, query: str, n_results: int = 5) -> list:
    """
    Search for the most relevant chunks for a given query.
    Returns a list of dicts with 'text' and 'source'.
    """
    total = collection.count()
    if total == 0:
        return []

    n = min(n_results, total)
    query_embedding = embed_texts(model, [query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        include=["documents", "metadatas", "distances"]
    )

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


# ── LLM (Gemini) ─────────────────────────────────────────────────────────────

def ask_gemini(api_key: str, context: str, question: str, history: list) -> str:
    """
    Send the question + retrieved context to Gemini and return the answer.

    history: list of {"role": "user"|"assistant", "content": str}
    """
    genai.configure(api_key=api_key)

    system_prompt = (
        "You are a helpful AI assistant that answers questions based on the provided document context.\n"
        "Use ONLY the information given in the context to answer the question.\n"
        "If the context does not contain enough information, say so clearly.\n"
        "Always mention which document your answer comes from when possible.\n\n"
        f"Context from documents:\n{context}"
    )

    llm = genai.GenerativeModel(
        model_name="gemini-3.1-flash-lite-preview",
        system_instruction=system_prompt
    )

    # Convert history to Gemini format (role "assistant" -> "model")
    gemini_history = []
    for msg in history:
        role = "model" if msg["role"] == "assistant" else "user"
        gemini_history.append({"role": role, "parts": [msg["content"]]})

    chat = llm.start_chat(history=gemini_history)
    response = chat.send_message(
        question,
        generation_config=genai.GenerationConfig(max_output_tokens=1024)
    )
    return response.text
