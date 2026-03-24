# RAG-Based AI Assistant

A simple **Retrieval-Augmented Generation (RAG)** system that lets you upload documents (PDFs or text files) and ask questions about them. Answers are grounded in your documents — not hallucinated.

## How It Works

```
Upload Document
    → Extract text from PDF/TXT
    → Split into chunks (500 chars each)
    → Convert to embeddings (all-MiniLM-L6-v2)
    → Store in ChromaDB (vector database)

Ask a Question
    → Convert question to embedding
    → Find most similar chunks in ChromaDB
    → Send chunks as context to Claude (LLM)
    → Display grounded answer with source citations
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| User Interface | Streamlit |
| Vector Database | ChromaDB |
| Embedding Model | sentence-transformers (`all-MiniLM-L6-v2`) |
| Language Model | Claude (Anthropic API) |
| PDF Parsing | PyPDF2 |

## Setup & Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Set your API key**
```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
```

**3. Run the app**
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

## Usage

1. Enter your Anthropic API key in the sidebar (or set it in `.env`)
2. Upload a PDF or TXT file using the sidebar uploader
3. Click **Index Document** to process it
4. Type your question in the chat box
5. The assistant will answer using only the content from your documents

## Project Structure

```
rag-agent-mini-project/
├── app.py           # Streamlit user interface
├── rag_engine.py    # Core RAG logic (load, chunk, embed, search, answer)
├── requirements.txt
├── .env.example     # Template for API key
└── README.md
```

## Key Concepts Demonstrated

- **Document Chunking**: Long documents are split into 500-character overlapping chunks for better retrieval
- **Vector Embeddings**: Text is converted to numerical vectors so semantic similarity can be measured
- **Similarity Search**: ChromaDB finds the most relevant chunks using cosine distance
- **Prompt Engineering**: Retrieved chunks are injected into Claude's context to produce accurate, grounded answers
- **RAG Architecture**: Combines retrieval (ChromaDB) with generation (Claude) to reduce hallucinations
