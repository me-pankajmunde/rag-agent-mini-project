# Project Report

## RAG-Based AI Assistant
### Retrieval-Augmented Generation for Document Question Answering

---

| | |
|---|---|
| **Project Title** | RAG-Based AI Assistant |
| **Submitted By** | Pankaj Munde |
| **Date** | March 2026 |
| **Technology Domain** | Artificial Intelligence / Natural Language Processing |

---

## Table of Contents

1. [Abstract](#abstract)
2. [Introduction](#introduction)
3. [Problem Statement](#problem-statement)
4. [Objectives](#objectives)
5. [Literature Review / Background](#literature-review--background)
6. [System Architecture](#system-architecture)
7. [Technology Stack](#technology-stack)
8. [Implementation Details](#implementation-details)
9. [Project Structure](#project-structure)
10. [System Workflow](#system-workflow)
11. [Key Features](#key-features)
12. [Results and Evaluation](#results-and-evaluation)
13. [Challenges and Solutions](#challenges-and-solutions)
14. [Future Enhancements](#future-enhancements)
15. [Conclusion](#conclusion)
16. [References](#references)

---

## Abstract

This project presents a **Retrieval-Augmented Generation (RAG) AI Assistant** — an intelligent document question-answering system that grounds its responses strictly in user-uploaded documents. The system accepts PDF and plain text files, processes them through a multi-stage pipeline involving text extraction, chunking, vector embedding, and semantic similarity search, and uses the Google Gemini large language model (LLM) to generate accurate, source-cited answers. By combining a vector database (ChromaDB) with a generative AI model, the system significantly reduces hallucinations — a common problem in standard LLM deployments — and provides transparent, verifiable answers with source attribution. The application is deployed as an interactive web interface built with Streamlit, enabling real-time conversational interaction with uploaded documents.

---

## Introduction

The rapid advancement of large language models (LLMs) has unlocked transformative capabilities in natural language understanding and generation. However, standalone LLMs have a critical limitation: they generate responses from parametric knowledge encoded during training, which can lead to factual inaccuracies ("hallucinations"), outdated information, and an inability to answer questions about private or domain-specific documents.

Retrieval-Augmented Generation (RAG) is a hybrid architecture that addresses this limitation by pairing a generative model with an information retrieval component. Instead of relying solely on training data, a RAG system first retrieves relevant information from a document store and then uses that retrieved context to generate a grounded, accurate response.

This project implements a fully functional RAG pipeline as a mini-project, demonstrating core AI engineering concepts including vector embeddings, semantic search, prompt engineering, and LLM API integration — packaged in a user-friendly web application.

---

## Problem Statement

Traditional search tools rely on keyword matching and cannot understand the semantic intent of a query. General-purpose LLMs, while capable of natural language dialogue, cannot answer questions about private documents and are prone to fabricating information. There is a need for a system that:

- Allows users to upload their own documents (PDFs, text files)
- Understands the semantic meaning of questions
- Retrieves the most relevant portions of uploaded documents
- Generates concise, accurate answers grounded only in those documents
- Clearly cites the source of each answer for verification

---

## Objectives

1. Build an end-to-end RAG pipeline from document ingestion to answer generation.
2. Implement efficient document chunking with configurable overlap to preserve context across chunk boundaries.
3. Use a pre-trained sentence embedding model to convert text into high-dimensional vectors for semantic search.
4. Integrate a persistent vector database (ChromaDB) for scalable and fast similarity retrieval.
5. Connect to the Google Gemini API for high-quality language generation.
6. Develop a clean, interactive web interface using Streamlit.
7. Provide source attribution and confidence scoring for every generated answer.
8. Support multi-turn conversation with maintained chat history.

---

## Literature Review / Background

### Retrieval-Augmented Generation (RAG)

RAG was introduced by Lewis et al. (2020) in the paper *"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"*. The key insight is that a retrieval component can supply up-to-date, domain-specific context to a generative model, improving factual accuracy without requiring model retraining.

### Vector Embeddings

Sentence embedding models convert text into dense numerical vectors (embeddings) such that semantically similar sentences map to nearby points in the vector space. This project uses the `all-MiniLM-L6-v2` model from the Sentence-Transformers library, which produces 384-dimensional embeddings. It is lightweight, fast, and achieves strong performance on semantic textual similarity benchmarks.

### Vector Databases

Vector databases are purpose-built for storing and querying high-dimensional vectors. ChromaDB is an open-source, embeddable vector database that supports persistent storage, cosine/L2 distance metrics, and metadata filtering. It is widely used in RAG prototypes due to its simplicity and Python-native API.

### Large Language Models — Google Gemini

Google Gemini is a family of multimodal LLMs developed by Google DeepMind. This project uses `gemini-2.0-flash`, a fast and cost-efficient model well-suited for context-grounded question answering. Gemini is accessed via the `google-generativeai` Python SDK.

---

## System Architecture

The system is composed of two logical layers:

### 1. Indexing Pipeline (Offline Phase)

```
User uploads PDF/TXT file
        |
        v
  Text Extraction
  (PyPDF2 / plain text reader)
        |
        v
  Text Chunking
  (500-char chunks, 50-char overlap)
        |
        v
  Embedding Generation
  (all-MiniLM-L6-v2 → 384-dim vectors)
        |
        v
  Vector Storage
  (ChromaDB — persistent on disk)
```

### 2. Query Pipeline (Online Phase)

```
User submits a question
        |
        v
  Query Embedding
  (same all-MiniLM-L6-v2 model)
        |
        v
  Semantic Search
  (ChromaDB cosine distance → Top-K chunks)
        |
        v
  Context Injection
  (retrieved chunks + conversation history → prompt)
        |
        v
  LLM Generation
  (Google Gemini API — streaming)
        |
        v
  Answer with Source Citations
  (displayed in Streamlit chat interface)
```

---

## Technology Stack

| Component | Technology | Version |
|---|---|---|
| User Interface | Streamlit | 1.42.2 |
| Vector Database | ChromaDB | 0.5.23 |
| Embedding Model | Sentence-Transformers (`all-MiniLM-L6-v2`) | 3.4.1 |
| Language Model | Google Gemini (`gemini-2.0-flash`) | google-generativeai 0.8.5 |
| PDF Parsing | PyPDF2 | 3.0.1 |
| Environment Config | python-dotenv | 1.0.1 |
| Programming Language | Python | 3.10+ |

---

## Implementation Details

### Module 1: `rag_engine.py` — Core RAG Logic

This module encapsulates all backend processing, keeping it cleanly separated from the UI layer.

#### 1.1 Document Loading

```python
def load_document(file_path: str) -> str:
    # Auto-detects file type (.pdf or .txt) and delegates
    # to extract_text_from_pdf() or extract_text_from_txt()
```

PDFs are parsed using `PyPDF2.PdfReader`, iterating over all pages to concatenate extracted text. Plain text files are read with UTF-8 encoding.

#### 1.2 Text Chunking

```python
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    step = chunk_size - overlap
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), step)]
    return [c for c in chunks if len(c) >= 20]
```

Overlapping chunks ensure that information near chunk boundaries is not lost during retrieval. The default configuration of 500-character chunks with 50-character overlap balances retrieval granularity and context preservation.

#### 1.3 Embedding

```python
def embed_texts(model, texts: list) -> list:
    return model.encode(texts).tolist()
```

The `all-MiniLM-L6-v2` model converts text to 384-dimensional floating-point vectors. These vectors capture semantic meaning, enabling similarity-based retrieval that outperforms keyword search.

#### 1.4 Vector Storage

```python
def get_vector_db(persist_dir: str = "./chroma_db"):
    # Returns a ChromaDB collection with cosine distance metric
    # Persists to disk automatically

def add_document(collection, model, chunks, doc_name) -> int:
    # Assigns UUID-based IDs, embeds chunks, stores with metadata
    # Metadata: {source: doc_name, doc_id: uuid, chunk_index: int}
```

Chunks are stored with metadata to enable source attribution during retrieval.

#### 1.5 Retrieval

```python
def search_similar(collection, model, query, n_results=5) -> list:
    # Embeds the query, queries ChromaDB, returns list of:
    # {"text": chunk_text, "source": doc_name, "distance": float}
```

Cosine distance is used as the similarity metric. Lower distance values indicate higher semantic similarity.

#### 1.6 LLM Generation

```python
def ask_gemini_stream(api_key, context, question, history):
    # Streams response from Gemini API
    # System instruction: answer ONLY from provided context
    # Conversation history preserved for multi-turn dialogue
```

The system prompt instructs Gemini to answer exclusively from the retrieved context, preventing hallucinations.

---

### Module 2: `app.py` — Streamlit Web Interface

The interface is divided into a sidebar (configuration and document management) and a main panel (chat interface).

#### 2.1 Sidebar Controls

| Control | Purpose |
|---|---|
| Gemini API Key | Authenticates with Google Generative AI |
| Top-K Retrieval | Number of chunks to retrieve (1–10, default 5) |
| Chunk Size | Characters per chunk (200–1000, default 500) |
| Chunk Overlap | Overlap between chunks (20–max, default 50) |
| File Uploader | Accepts PDF or TXT files |
| Index Document | Triggers the indexing pipeline |
| Indexed Documents | Lists all documents in ChromaDB |
| Clear All Data | Wipes the vector database |

#### 2.2 Confidence Scoring

Retrieved chunks are assigned a confidence label based on their cosine distance to the query:

```python
def distance_to_confidence(distance: float) -> str:
    if distance <= 0.25:  return "High"
    elif distance <= 0.45: return "Medium"
    else:                  return "Low"
```

This provides users with a qualitative indicator of how well each source chunk matches their query.

#### 2.3 Chat Interface

- Displays full conversation history with role-based formatting (user/assistant)
- Streams responses in real-time using `st.write_stream()`
- Shows expandable source citations below each answer
- Maintains multi-turn context by passing full chat history to Gemini

---

## Project Structure

```
rag-agent-mini-project/
├── app.py              # Streamlit UI — chat interface, sidebar controls
├── rag_engine.py       # Core RAG logic — document loading, chunking,
│                       # embedding, vector DB, Gemini integration
├── requirements.txt    # Python dependency list
├── .env.example        # API key configuration template
├── .gitignore          # Excludes .env, chroma_db/, uploads/, __pycache__/
└── README.md           # Project setup and usage documentation
```

The `chroma_db/` directory (created at runtime) stores the persistent vector database. The `uploads/` directory (created at runtime) holds temporary files during document processing.

---

## System Workflow

### Document Indexing

1. User uploads a PDF or TXT file via the Streamlit sidebar.
2. The file is saved temporarily to the `uploads/` directory.
3. Text is extracted using PyPDF2 (PDF) or a file reader (TXT).
4. The text is split into overlapping chunks using the user-configured chunk size and overlap.
5. Each chunk is embedded into a 384-dimensional vector using `all-MiniLM-L6-v2`.
6. Chunks, their embeddings, and metadata (source document name) are stored in ChromaDB.
7. The temporary file is deleted; the document name appears in the "Indexed Documents" list.

### Question Answering

1. User types a question in the chat input.
2. The question is embedded using the same `all-MiniLM-L6-v2` model.
3. ChromaDB performs a cosine similarity search and returns the top-K most relevant chunks.
4. Retrieved chunks are formatted into a context block.
5. The context, question, and conversation history are sent to the Gemini API.
6. The response is streamed token-by-token to the chat interface.
7. Source citations with confidence labels are displayed in an expandable section below the answer.

---

## Key Features

| Feature | Description |
|---|---|
| Multi-format Upload | Supports both PDF and plain text (TXT) files |
| Persistent Vector DB | Documents remain indexed across sessions (ChromaDB on disk) |
| Configurable Chunking | Chunk size and overlap adjustable via UI sliders |
| Semantic Search | Cosine similarity over 384-dim embeddings (not keyword matching) |
| Streaming Responses | Real-time token-by-token answer generation |
| Source Attribution | Every answer cites the source chunk(s) it was derived from |
| Confidence Labels | High / Medium / Low confidence based on vector distance |
| Multi-turn Dialogue | Full conversation history preserved and sent to Gemini |
| Hallucination Prevention | System prompt restricts Gemini to retrieved context only |
| Multiple Documents | Multiple files can be indexed and queried simultaneously |
| Clear Data | One-click wipe of all indexed documents |

---

## Results and Evaluation

### Evaluation Methodology

The system was evaluated against a test set of 15–20 questions across three categories:
- **40% Factual** — Direct answers present in a single chunk
- **40% Multi-sentence** — Answers requiring combination of nearby context
- **20% Out-of-context** — Answers not present in indexed documents

### Metrics

| Metric | Description |
|---|---|
| **Retrieval Hit@5** | Percentage of questions where at least one of the top-5 retrieved chunks contained correct evidence |
| **Answer Groundedness** | Percentage of answers that stayed within retrieved context (no hallucinations) |
| **Source Quality** | Relevance and interpretability of cited source chunks |
| **Failure Handling** | Clarity of "information not available" responses for out-of-context questions |

### Scoring Rubric

Each question is scored on a 0–2 scale:

| Score | Meaning |
|---|---|
| 2 | Correct answer, clearly grounded in source document |
| 1 | Partially correct or weakly grounded |
| 0 | Incorrect or hallucinated |

**Final Score** = Total Score / (2 × Number of Questions)

### Expected Outcomes

- High Retrieval Hit@5 (>80%) for factual and multi-sentence questions
- Near-zero hallucination rate (system prompt enforcement)
- Clear failure messages for out-of-context questions
- Response latency under 3 seconds for typical documents

---

## Challenges and Solutions

| Challenge | Solution |
|---|---|
| Long documents exceed LLM context window | Chunking divides documents into manageable segments; only top-K are sent |
| Chunk boundaries split sentences mid-context | Overlapping chunks (default 50 chars) preserve cross-boundary information |
| Keyword search misses semantic intent | Vector embeddings + cosine similarity enable intent-aware retrieval |
| LLM hallucinations on private documents | System prompt strictly restricts generation to retrieved context |
| Slow re-loading of embedding model | Streamlit `@st.cache_resource` ensures model loads only once per session |
| API key management | `.env` file with `python-dotenv`; sidebar fallback for convenience |
| Large PDF text extraction | PyPDF2 iterates all pages and concatenates; robust for multi-page PDFs |

---

## Future Enhancements

1. **Re-ranking**: Apply a cross-encoder re-ranker on retrieved chunks before sending to the LLM for improved precision.
2. **Hybrid Search**: Combine dense vector search with BM25 sparse retrieval for better recall on keyword-specific queries.
3. **Multi-modal Support**: Extend ingestion to handle images, tables, and scanned PDFs via OCR.
4. **Document Summarization**: Provide an auto-summary of uploaded documents before the Q&A session.
5. **User Authentication**: Add login system to isolate document stores per user.
6. **Cloud Deployment**: Deploy on a cloud platform (e.g., Google Cloud Run, AWS Lambda) for shared access.
7. **Fine-tuned Embeddings**: Fine-tune the embedding model on domain-specific corpora for higher retrieval accuracy.
8. **Evaluation Dashboard**: Build an automated evaluation interface for measuring RAG performance metrics.

---

## Conclusion

This project successfully demonstrates the design and implementation of a complete Retrieval-Augmented Generation (RAG) pipeline. By integrating semantic vector search (ChromaDB + Sentence-Transformers) with a large language model (Google Gemini), the system delivers accurate, grounded answers to user questions about uploaded documents — without hallucinating information that is not present.

The project covers multiple important concepts in modern AI engineering:
- Embedding-based semantic search
- Vector database management
- Prompt engineering for constrained generation
- LLM API integration with streaming
- Full-stack AI application development with Python

The clean modular architecture (separate `rag_engine.py` and `app.py`) ensures maintainability and extensibility. The configurable parameters (chunk size, overlap, top-K) allow experimentation and understanding of how each component affects system performance.

This RAG assistant serves as a practical foundation for production-grade document intelligence applications in domains such as legal research, academic study, enterprise knowledge management, and customer support.

---

## References

1. Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020. https://arxiv.org/abs/2005.11401

2. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP 2019. https://arxiv.org/abs/1908.10084

3. Google DeepMind. (2024). *Gemini: A Family of Highly Capable Multimodal Models*. https://deepmind.google/technologies/gemini/

4. Chroma. (2024). *ChromaDB — The AI-native open-source embedding database*. https://www.trychroma.com/

5. Streamlit Inc. (2024). *Streamlit — A faster way to build and share data apps*. https://streamlit.io/

6. Hugging Face. (2024). *Sentence-Transformers Documentation*. https://www.sbert.net/

7. PyPDF2 Contributors. (2024). *PyPDF2 — A pure-python PDF library*. https://pypdf2.readthedocs.io/

---

*Report generated: March 2026*
