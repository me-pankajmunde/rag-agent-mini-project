# RAG-Based AI Assistant

A lightweight Retrieval-Augmented Generation (RAG) application for document Q&A. Upload PDF or TXT files, index them into ChromaDB, and ask grounded questions through a Gradio interface.

## Features

- PDF and TXT ingestion
- Configurable chunk size, overlap, and top-k retrieval
- Semantic retrieval using `all-MiniLM-L6-v2` embeddings
- OpenAI chat-completions streaming responses
- Source snippets with confidence labels

## Tech Stack

| Component | Technology |
|---|---|
| UI | Gradio |
| Vector DB | ChromaDB |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| LLM API | OpenAI (`gpt-4o` default, configurable) |
| Document Parsing | PyPDF2 |

## Standard Project Structure

```
rag-agent-mini-project/
├── data/
│   └── chroma_db/            # Persistent vector store
├── docs/
│   └── PROJECT_REPORT.md
├── src/
│   └── rag_agent/
│       ├── __init__.py
│       ├── app.py            # Gradio application
│       └── rag_engine.py     # RAG core logic
├── tests/
├── .env.example
├── README.md
├── requirements.txt
└── SETUP_GUIDE.md
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables:

```bash
cp .env.example .env
```

Set at least:

```env
OPENAI_API_KEY=your_api_key
```

Optional:

```env
OPENAI_MODEL=gpt-4o
```

## Run

Run from the repository root:

```bash
PYTHONPATH=src python -m rag_agent.app
```

The Gradio app starts on a local URL shown in terminal output.

## Usage

1. Enter your OpenAI API key in the sidebar if not set in `.env`.
2. Upload a PDF or TXT document.
3. Click Index Document.
4. Ask questions in the chat panel.
5. Review the source snippets shown with each answer.

## Notes

- Vector data is now persisted under `data/chroma_db`.
- Existing indexed data from earlier layout has been moved into this directory.
