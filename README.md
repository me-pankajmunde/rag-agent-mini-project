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
    → Send chunks as context to Gemini (LLM)
    → Display grounded answer with source citations
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| User Interface | Streamlit |
| Vector Database | ChromaDB |
| Embedding Model | sentence-transformers (`all-MiniLM-L6-v2`) |
| Language Model | Gemini (Google Generative AI API) |
| PDF Parsing | PyPDF2 |

## Setup & Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Set your API key**
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

Optional model override:
```bash
# Stable default in code is gemini-2.0-flash
GEMINI_MODEL=gemini-2.0-flash
```

**3. Run the app**
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

## Usage

1. Enter your Gemini API key in the sidebar (or set it in `.env`)
2. (Optional) Adjust **Top-K Retrieval**, **Chunk Size**, and **Chunk Overlap** in the sidebar
3. Upload a PDF or TXT file using the sidebar uploader
4. Click **Index Document** to process it
5. Type your question in the chat box
6. The assistant will answer using only the content from your documents

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
- **Prompt Engineering**: Retrieved chunks are injected into Gemini's context to produce accurate, grounded answers
- **RAG Architecture**: Combines retrieval (ChromaDB) with generation (Gemini) to reduce hallucinations

## Evaluation (Final Year Project)

Use this section during demo/viva to show measurable performance.

### Suggested Test Setup

- Prepare 2-3 documents (course notes, technical report, policy PDF)
- Create 15-20 questions:
    - 40% factual (direct answer present in one chunk)
    - 40% multi-sentence (requires combining nearby context)
    - 20% out-of-context (answer not present in documents)

### Metrics to Report

- **Retrieval Hit@5**: Did at least one returned chunk contain the correct evidence?
- **Answer Groundedness**: Did the answer stay within retrieved context?
- **Source Quality**: Were cited source chunks relevant and interpretable?
- **Failure Handling**: For out-of-context questions, did the system clearly say information was unavailable?

### Simple Scoring Rubric

For each question, score 0-2:

- **2** = Correct and clearly grounded in source
- **1** = Partially correct or weak grounding
- **0** = Incorrect or hallucinated

Report final score as:

- `Total Score / (2 × Number of Questions)`
- Plus `%` of questions with correct retrieval evidence (Hit@5)
