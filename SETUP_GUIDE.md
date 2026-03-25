# Step-by-Step Guide to Run the RAG-Based AI Assistant

---

## Prerequisites

Before you begin, make sure the following are installed on your system:

| Requirement | Minimum Version | Check Command |
|---|---|---|
| Python | 3.10+ | `python --version` |
| pip | 22+ | `pip --version` |
| Git | Any | `git --version` |
| Internet connection | — | Required for API calls and first-time model download |

> **Note:** The embedding model (`all-MiniLM-L6-v2`, ~90 MB) is downloaded automatically on first run.

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/me-pankajmunde/rag-agent-mini-project.git
cd rag-agent-mini-project
```

---

## Step 2 — Create a Virtual Environment

Creating a virtual environment keeps dependencies isolated from your system Python.

**On Linux / macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

**On Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

You should see `(.venv)` at the start of your terminal prompt, confirming the environment is active.

---

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs the following packages:

| Package | Purpose |
|---|---|
| `streamlit` | Web application interface |
| `google-generativeai` | Google Gemini LLM API |
| `chromadb` | Vector database for storing embeddings |
| `sentence-transformers` | Generates text embeddings (`all-MiniLM-L6-v2`) |
| `PyPDF2` | Extracts text from PDF files |
| `python-dotenv` | Loads API keys from `.env` file |

> Installation may take 2–5 minutes. The `sentence-transformers` and `chromadb` packages are the largest.

---

## Step 4 — Get a Gemini API Key

1. Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the generated key (it looks like `AIzaSy...`)

> The free tier is sufficient for development and testing.

---

## Step 5 — Configure the API Key

Copy the example environment file and add your key:

```bash
cp .env.example .env
```

Open `.env` in any text editor and replace the placeholder:

```
GEMINI_API_KEY=your_actual_api_key_here
```

**Optional** — to use a specific Gemini model:
```
GEMINI_API_KEY=your_actual_api_key_here
GEMINI_MODEL=gemini-2.0-flash
```

> If you skip this step, you can also enter the API key directly in the app's sidebar each time you run it.

---

## Step 6 — Run the Application

```bash
streamlit run app.py
```

You will see output like:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

The app opens automatically in your default browser. If it does not, navigate to `http://localhost:8501` manually.

---

## Step 7 — Using the Application

### 7.1 — Enter Your API Key (if not set in `.env`)

In the left sidebar, find the **"Gemini API Key"** field and paste your key.

---

### 7.2 — (Optional) Adjust RAG Settings

In the sidebar, you can tune three parameters before uploading:

| Setting | Default | What it controls |
|---|---|---|
| **Top-K Retrieval** | 5 | How many document chunks are retrieved per question |
| **Chunk Size** | 500 | Characters per chunk when splitting the document |
| **Chunk Overlap** | 50 | Overlapping characters between consecutive chunks |

> Start with defaults. Increase **Chunk Size** for longer-context documents; increase **Top-K** if answers seem incomplete.

---

### 7.3 — Upload a Document

1. In the sidebar, click **"Browse files"** under the file uploader.
2. Select a **PDF** or **TXT** file from your computer.
3. Click the **"Index Document"** button that appears after the file is selected.
4. Wait for the success message: `"Indexed X chunks from filename.pdf"`

You can repeat this step to upload and index multiple documents.

---

### 7.4 — Ask Questions

1. Type your question in the chat box at the bottom of the main panel.
2. Press **Enter** or click the send button.
3. The assistant will stream its answer in real time.
4. Expand the **"Sources"** section below each answer to see:
   - Which document the answer came from
   - The exact chunk of text used
   - A confidence label (**High** / **Medium** / **Low**)

---

### 7.5 — Continue the Conversation

You can ask follow-up questions — the assistant remembers the full conversation history within the session.

---

### 7.6 — Manage Documents

- **View indexed documents**: The sidebar lists all files currently in the vector database.
- **Clear all data**: Click the **"Clear All Indexed Data"** button in the sidebar to wipe the database and start fresh.

---

## Directory Structure After First Run

```
rag-agent-mini-project/
├── app.py
├── rag_engine.py
├── requirements.txt
├── .env                  ← Created by you (contains API key)
├── .env.example
├── README.md
├── chroma_db/            ← Created automatically (vector database)
└── uploads/              ← Created automatically (temp file storage)
```

> `chroma_db/` persists your indexed documents between sessions. Delete this folder to start with a clean database.

---

## Stopping the Application

Press `Ctrl + C` in the terminal to stop the Streamlit server.

To deactivate the virtual environment when done:

```bash
deactivate
```

---

## Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| `ModuleNotFoundError` | Dependencies not installed or wrong environment active | Re-run `pip install -r requirements.txt` with venv active |
| `GEMINI_API_KEY not set` | Missing or wrong key | Check `.env` file or enter key in sidebar |
| `Invalid API Key` error | Incorrect key value | Re-copy the key from Google AI Studio |
| App doesn't open in browser | Port 8501 in use | Run `streamlit run app.py --server.port 8502` |
| Slow first startup | Embedding model downloading | Wait ~1 minute; only happens on first run |
| Empty or irrelevant answers | Chunk size too large or Top-K too low | Reduce chunk size to 300, increase Top-K to 7–10 |
| PDF text not extracted | Scanned/image-based PDF | Use a text-based PDF; OCR not supported |

---

## Quick Reference — Common Commands

```bash
# Activate virtual environment
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

# Run on a specific port
streamlit run app.py --server.port 8502

# Deactivate virtual environment
deactivate

# Reset the vector database
rm -rf chroma_db/
```
