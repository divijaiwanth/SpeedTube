# ⚡ SpeedTube

> Get up to speed on any YouTube video — instantly.

SpeedTube is a production-grade RAG (Retrieval-Augmented Generation) system that lets you ask questions, get summaries, and extract insights from any YouTube video using its transcript. Built with advanced retrieval techniques, a FastAPI backend, and a React frontend.

---

## What it does

Paste any YouTube URL. SpeedTube fetches the transcript, builds a semantic search index, and lets you have a conversation with the video — asking questions, getting summaries, or pulling key quotes — all without watching a single second of it.

---

## Architecture

```
YouTube URL
    ↓
Transcript fetch (youtube-transcript-api)
    ↓
Parent-child chunking (LangChain)
    ↓
┌─────────────────────────┐
│   Dual Index            │
│   FAISS (semantic)      │
│   BM25  (keyword)       │
└─────────────────────────┘
    ↓
Query comes in
    ↓
HyDE — generate hypothetical answer, embed it
    ↓
Hybrid retrieval — RRF fusion of FAISS + BM25
    ↓
Cross-encoder reranking (BAAI/bge-reranker-base)
    ↓
Parent chunk lookup — fetch full context
    ↓
LLM answer generation (Groq / llama-3.1-8b)
    ↓
FastAPI response → React UI
```

---

## Tech Stack

| Layer | Tech |
|---|---|
| LLM | Groq (llama-3.1-8b-instant) |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Reranker | BAAI/bge-reranker-base (cross-encoder) |
| Vector store | FAISS |
| Keyword search | BM25 (rank-bm25) |
| RAG framework | LangChain |
| Backend | FastAPI + Uvicorn |
| Frontend | React |
| Evaluation | RAGAS + MLflow |

---

## Project Structure

```
SpeedTube/
├── backend/
│   └── main.py              # FastAPI app — all API endpoints
├── frontend/
│   └── src/
│       ├── App.js            # React app — UI components
│       └── App.css           # Styles
├── langchain_helper.py       # Core RAG pipeline (Phase 1)
├── evaluate.py               # RAGAS evaluation pipeline (Phase 2)
├── diagnose.py               # YouTube transcript diagnostics
├── requirements.txt
└── .env                      # API keys (never committed)
```

---

## API Endpoints

The FastAPI backend runs at `http://localhost:8000`. Full interactive docs at `/docs`.

### `POST /ingest`
Load a YouTube video and build the RAG index.

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=zjkBMFhNj_g"}'
```

```json
{
  "session_id": "92de3076a6db",
  "video_id": "zjkBMFhNj_g",
  "chunk_count": 320,
  "message": "Video loaded successfully",
  "cached": false
}
```

### `POST /query`
Ask a question about a loaded video.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"session_id": "92de3076a6db", "question": "What is a large language model?"}'
```

```json
{
  "answer": "A large language model is...",
  "sources": ["...chunk 1...", "...chunk 2...", "...chunk 3..."],
  "num_sources": 3
}
```

### `GET /summary`
Get a summary of a loaded video.

```bash
curl "http://localhost:8000/summary?session_id=92de3076a6db&summary_type=bullets"
```

`summary_type` options: `concise`, `detailed`, `bullets`

### `GET /health`
Health check — returns uptime and active session count.

### `DELETE /session/{session_id}`
Clear a session from memory.

---

## Advanced RAG Techniques (Phase 1)

### Parent-Child Chunking
Small chunks (300 chars) are used for retrieval — precise matching. When a child chunk is retrieved, its parent (1200 chars) is sent to the LLM — full context. Prevents relevant context from being split across chunk boundaries.

### Hybrid Search + RRF
BM25 handles exact keyword matches (names, technical terms). FAISS handles semantic similarity. Both run in parallel and are fused using Reciprocal Rank Fusion:

```
score = Σ 1 / (rank + 60)
```

Higher-ranked results from either retriever get boosted. The constant 60 prevents top-ranked results from completely dominating.

### HyDE (Hypothetical Document Embeddings)
Instead of embedding the raw question, the LLM generates a hypothetical answer first, then that answer is embedded. A hypothetical answer is lexically closer to the actual transcript than a short question, improving cosine similarity matching significantly.

Paper: [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496)

### Cross-Encoder Reranking
10 candidate chunks are retrieved via hybrid search. All 10 are passed through `BAAI/bge-reranker-base` — a cross-encoder that scores (query, document) pairs jointly instead of independently. Top 3 are kept. Cross-encoders are more accurate than cosine similarity but too slow to run on the full index, hence the two-stage approach.

---

## Evaluation (Phase 2)

SpeedTube includes a full evaluation pipeline using [RAGAS](https://docs.ragas.io) and [MLflow](https://mlflow.org).

### Metrics

| Metric | What it measures |
|---|---|
| Faithfulness | Are all claims in the answer supported by the retrieved context? |
| Answer Relevancy | Is the answer actually relevant to the question asked? |
| Context Precision | Are relevant chunks ranked before irrelevant ones? |
| Context Recall | Did retrieval find all the chunks needed to answer? |

### Running evaluation

```bash
python evaluate.py <youtube_video_id> <n_questions>

# Example
python evaluate.py zjkBMFhNj_g 10
```

### Viewing results in MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Open `http://localhost:5001` to compare runs, view per-question scores, and track how pipeline changes affect metrics.

### Sample results

| Video type | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| Conversational | 0.25 | 0.82 | 0.00 | 0.00 |
| Technical tutorial | 0.33 | 0.56 | 0.40 | 0.50 |

Context metrics improve significantly on structured technical content. All metrics will improve further after Phase 5 (Bedrock migration) when a stronger judge model is used for evaluation.

---

## Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- A [Groq API key](https://console.groq.com) (free)

### Backend

```bash
git clone https://github.com/yourusername/SpeedTube.git
cd SpeedTube

# Create virtual environment
uv venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

# Install dependencies
uv pip install -r requirements.txt

# Set environment variable
$env:GROQ_API_KEY = "gsk_your_key_here"  # Windows PowerShell
export GROQ_API_KEY="gsk_your_key_here"  # Mac/Linux

# Run backend
cd backend
uvicorn main:app --reload --port 8000
```

API docs available at `http://localhost:8000/docs`

### Frontend

```bash
cd frontend
npm install
npm start
```

App available at `http://localhost:3000`

---

## Roadmap

- [x] Phase 1 — Advanced RAG pipeline (hybrid search, reranking, HyDE, parent-child chunking)
- [x] Phase 2 — RAGAS evaluation + MLflow experiment tracking
- [x] Phase 3 — FastAPI backend + React frontend
- [ ] Phase 4 — Guardrails, rate limiting, Redis caching, structured logging
- [ ] Phase 5 — AWS Bedrock migration (Claude Haiku + Titan Embeddings + Lambda deployment)

---

## Known Limitations

- Context precision and recall metrics require a strong judge model — currently 0 on conversational videos with llama-3.1-8b. Will be resolved in Phase 5 with Bedrock.
- YouTube transcript API requires browser cookies for some videos due to IP-based rate limiting.
- In-memory session store resets on server restart — Redis persistence coming in Phase 4.
- HyDE disabled by default locally due to inference latency — will be re-enabled on Bedrock.