# Step-by-Step Setup Guide

This guide is for the current project structure and stack:
- UI: Gradio
- LLM API: OpenAI-compatible chat completions
- Vector DB: ChromaDB (persisted under data/chroma_db)

## Prerequisites

- Python 3.10+
- pip
- Internet access (for first-time model download)

## 1. Clone and enter the repo

```bash
git clone https://github.com/me-pankajmunde/rag-agent-mini-project.git
cd rag-agent-mini-project
```

## 2. Install dependencies

Preferred (virtual environment):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If your environment blocks installation with an "externally managed environment" error (PEP 668), use this fallback:

```bash
python3 -m pip install --break-system-packages -r requirements.txt
```

## 3. Configure environment variables

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
OPENAI_API_BASE_URL=https://api.openai.com/v1
```

Note: If OPENAI_API_BASE_URL is omitted, the app uses its configured default.

## 4. Run the app

From repo root:

```bash
PYTHONPATH=src python3 -m rag_agent.app
```

The terminal will print a local Gradio URL you can open in your browser.

## 5. Use the app

1. Enter API key in the sidebar if not already set in .env.
2. Upload a PDF or TXT document.
3. Click Index Document.
4. Ask questions in chat.
5. Review the source snippets and confidence labels.

## Troubleshooting

- ModuleNotFoundError for rag_engine:
  - Use module mode from repo root: PYTHONPATH=src python3 -m rag_agent.app
- No module named gradio (or similar):
  - Reinstall dependencies from requirements.txt in the same Python environment you run.
- Externally managed environment error:
  - Use the --break-system-packages fallback shown above, or install python3-venv and use a virtual environment.
- KeyError: '_type' from chromadb:
  - This indicates an incompatible persisted Chroma config.
  - The app now auto-backs up incompatible data directory as data/chroma_db_backup_<timestamp> and retries.
  - If persistent mode still fails in your environment, it falls back to in-memory Chroma for the current session.
- First startup is slow:
  - sentence-transformers downloads model files on first run.
