# Project Report

## RAG-Based AI Assistant
### Retrieval-Augmented Generation for Document Question Answering

| | |
|---|---|
| Project Title | RAG-Based AI Assistant |
| Submitted By | Pankaj Munde |
| Date | April 2026 |
| Domain | AI / NLP |

## Abstract

This project implements a compact, end-to-end Retrieval-Augmented Generation (RAG) assistant for document question answering. Users upload PDF or TXT files, the system extracts and chunks text, generates dense embeddings, stores them in ChromaDB, retrieves the most relevant chunks for each query, and produces grounded responses through the OpenAI chat API. The interface is built with Gradio and supports streaming responses with source visibility. The project follows a standard Python src-based directory layout to improve maintainability and extensibility.

## Problem Statement

Standalone LLMs may hallucinate, and traditional keyword search lacks semantic understanding. The goal is to build a practical system that:

- Accepts user documents (PDF/TXT)
- Performs semantic retrieval over document chunks
- Generates answers grounded in retrieved evidence
- Provides transparent source snippets

## Objectives

1. Build a modular RAG pipeline from ingestion to response generation.
2. Use sentence-transformer embeddings for semantic similarity retrieval.
3. Persist vectors in a local ChromaDB store.
4. Provide an interactive chat UI for end users.
5. Maintain a standard project structure for clean development workflow.

## System Architecture

### Indexing Pipeline

```text
Upload File -> Text Extraction -> Chunking -> Embedding -> ChromaDB Persistence
```

### Query Pipeline

```text
User Question -> Query Embedding -> Top-K Retrieval -> Prompt Assembly -> OpenAI Response (streaming)
```

## Technology Stack

| Component | Technology |
|---|---|
| UI | Gradio |
| Vector Database | ChromaDB |
| Embedding Model | sentence-transformers (`all-MiniLM-L6-v2`) |
| LLM API | OpenAI Chat Completions (`gpt-4o` default) |
| PDF Parsing | PyPDF2 |
| Config | python-dotenv |
| Language | Python 3.10+ |

## Implementation Overview

### Core Module: `src/rag_agent/rag_engine.py`

Responsibilities:

- Load document text from PDF/TXT
- Chunk text with configurable `chunk_size` and `overlap`
- Generate embeddings using sentence-transformers
- Add and query vectors in ChromaDB
- Construct prompt messages and stream OpenAI responses

### UI Module: `src/rag_agent/app.py`

Responsibilities:

- Render Gradio interface
- Capture indexing and query settings
- Trigger indexing and similarity search
- Stream model responses into chat
- Display source snippets and confidence labels

## Standard Directory Structure

```text
rag-agent-mini-project/
├── data/
│   └── chroma_db/
├── docs/
│   └── PROJECT_REPORT.md
├── src/
│   └── rag_agent/
│       ├── __init__.py
│       ├── app.py
│       └── rag_engine.py
├── tests/
├── .env.example
├── README.md
├── requirements.txt
└── SETUP_GUIDE.md
```

## Workflow

### Document Indexing

1. User uploads PDF/TXT file.
2. Text is extracted and chunked.
3. Chunks are embedded and inserted into ChromaDB.
4. Source metadata is stored with each chunk.

### Question Answering

1. User submits a question.
2. Question embedding is compared against stored chunk embeddings.
3. Top-k relevant chunks are retrieved.
4. Retrieved context and chat history are sent to OpenAI.
5. Response is streamed back to the UI with source snippets.

## Key Features

- Persistent local vector storage (`data/chroma_db`)
- Configurable retrieval behavior (Top-K, chunk size, overlap)
- Streaming chat responses
- Basic confidence heuristic from vector distance
- Multi-document retrieval support

## Evaluation Plan

Suggested evaluation set:

- 15-20 questions over 2-3 documents
- Mix of factual, synthesis, and out-of-context prompts

Suggested metrics:

- Retrieval Hit@K
- Groundedness of final answers
- Source relevance
- Failure behavior for missing context

## Challenges and Mitigations

| Challenge | Mitigation |
|---|---|
| Hallucination risk | System prompt constrains model to provided context |
| Chunk boundary information loss | Overlap-based chunking |
| Query/document mismatch | Semantic embeddings instead of keyword-only retrieval |
| Local persistence management | Standardized `data/chroma_db` location |

## Future Enhancements

1. Add automated tests under `tests/` for chunking, retrieval, and prompt assembly.
2. Add re-ranking for improved precision.
3. Support OCR for scanned PDFs.
4. Add per-document filtering and metadata search.
5. Package as installable module with CLI entry points.

## Conclusion

The project demonstrates a practical RAG system with clear modular separation between UI and engine logic, persistent vector storage, and grounded generation behavior. With the updated standard directory structure, the codebase is now better positioned for testing, packaging, and future production-oriented improvements.

## References

1. Lewis et al. (2020), Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
2. Reimers and Gurevych (2019), Sentence-BERT.
3. ChromaDB Documentation.
4. OpenAI API Documentation.
5. Gradio Documentation.
