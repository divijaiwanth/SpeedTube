"""
RAGTube - Phase 3: FastAPI Backend
===================================
Endpoints:
  POST /ingest          - Load a YouTube video and build the RAG index
  POST /query           - Ask a question about a loaded video
  GET  /summary         - Get a summary of a loaded video
  GET  /health          - Health check

How sessions work:
  - Each video load creates a session_id (hash of the video ID)
  - The index is cached in memory so the same video isn't re-embedded twice
  - session_id is passed with every query and summary request
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional
import hashlib
import time
import logging
from urllib.parse import urlparse, parse_qs

# Import your Phase 1 RAG pipeline
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_helper import create_video_index, get_answer, get_summary, VideoIndex

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ragtube")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAGTube API",
    description="YouTube transcript RAG pipeline — ask questions about any YouTube video",
    version="2.0.0",
)

# CORS — allow React frontend (localhost:3000) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory session store
# session_id -> VideoIndex
# ---------------------------------------------------------------------------

sessions: dict[str, VideoIndex] = {}
session_metadata: dict[str, dict] = {}  # stores video_id, created_at, chunk_count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from any URL format."""
    try:
        parsed = urlparse(url)
        if parsed.hostname in ("youtu.be",):
            return parsed.path.lstrip("/")
        if parsed.hostname in ("www.youtube.com", "youtube.com"):
            qs = parse_qs(parsed.query)
            return qs.get("v", [None])[0]
    except Exception:
        return None
    return None


def make_session_id(video_id: str) -> str:
    """Deterministic session ID from video ID — same video always gets same session."""
    return hashlib.md5(video_id.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Request / Response schemas (Pydantic)
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    url: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "url": "https://www.youtube.com/watch?v=zjkBMFhNj_g"
            }
        }
    }


class IngestResponse(BaseModel):
    session_id: str
    video_id: str
    chunk_count: int
    message: str
    cached: bool  # True if index was already in memory


class QueryRequest(BaseModel):
    session_id: str
    question: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "abc123def456",
                "question": "What is the main topic of this video?"
            }
        }
    }


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    num_sources: int


class SummaryResponse(BaseModel):
    summary: str
    summary_type: str


class HealthResponse(BaseModel):
    status: str
    active_sessions: int
    uptime_seconds: float


# ---------------------------------------------------------------------------
# Track startup time for health check
# ---------------------------------------------------------------------------

START_TIME = time.time()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    Use this to verify the API is running before making other calls.
    """
    return HealthResponse(
        status="ok",
        active_sessions=len(sessions),
        uptime_seconds=round(time.time() - START_TIME, 2),
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Video"])
async def ingest_video(request: IngestRequest):
    """
    Load a YouTube video and build the RAG index.

    - Extracts the transcript from the YouTube URL
    - Builds FAISS vector index + BM25 keyword index
    - Returns a session_id to use in subsequent /query and /summary calls
    - If the same video was already loaded, returns the cached session instantly

    This is idempotent — calling it twice with the same URL is safe.
    """
    # Extract video ID from URL
    video_id = extract_video_id(request.url)
    if not video_id:
        raise HTTPException(
            status_code=400,
            detail="Could not extract video ID from URL. Make sure it's a valid YouTube URL."
        )

    session_id = make_session_id(video_id)

    # Return cached session if already loaded
    if session_id in sessions:
        logger.info(f"Cache hit for video {video_id}, session {session_id}")
        meta = session_metadata[session_id]
        return IngestResponse(
            session_id=session_id,
            video_id=video_id,
            chunk_count=meta["chunk_count"],
            message="Video already loaded — using cached index",
            cached=True,
        )

    # Build the index
    logger.info(f"Building index for video {video_id}")
    try:
        index = create_video_index(video_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to build index for {video_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process video: {str(e)}"
        )

    # Store in session cache
    sessions[session_id] = index
    session_metadata[session_id] = {
        "video_id": video_id,
        "chunk_count": len(index.child_docs),
        "created_at": time.time(),
    }

    logger.info(f"Index built: {len(index.child_docs)} chunks, session {session_id}")

    return IngestResponse(
        session_id=session_id,
        video_id=video_id,
        chunk_count=len(index.child_docs),
        message="Video loaded successfully",
        cached=False,
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_video(request: QueryRequest):
    """
    Ask a question about a loaded video.

    Requires a valid session_id from a previous /ingest call.
    Uses the full advanced RAG pipeline:
      - Hybrid search (BM25 + FAISS)
      - Cross-encoder reranking
      - Parent-child chunk lookup
      - Conversational memory (remembers last 5 turns)
    """
    if request.session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please call /ingest first to load the video."
        )

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    index = sessions[request.session_id]

    try:
        result = get_answer(index, request.question)
    except Exception as e:
        logger.error(f"Query failed for session {request.session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        num_sources=result["num_sources"],
    )


@app.get("/summary", response_model=SummaryResponse, tags=["RAG"])
async def get_video_summary(
    session_id: str,
    summary_type: str = "concise"
):
    """
    Get a summary of a loaded video.

    summary_type options:
      - concise  : 3-5 sentence overview (default)
      - detailed : paragraph per major topic
      - bullets  : key takeaways as bullet points

    Uses map-reduce over the full transcript — takes 30-60 seconds.
    """
    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please call /ingest first to load the video."
        )

    valid_types = ["concise", "detailed", "bullets"]
    if summary_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid summary_type. Must be one of: {valid_types}"
        )

    index = sessions[session_id]

    try:
        summary = get_summary(index, summary_type)
    except Exception as e:
        logger.error(f"Summary failed for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Summary failed: {str(e)}")

    return SummaryResponse(
        summary=summary,
        summary_type=summary_type,
    )


@app.delete("/session/{session_id}", tags=["System"])
async def clear_session(session_id: str):
    """
    Clear a session from memory.
    Call this when you're done with a video to free up RAM.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    del sessions[session_id]
    del session_metadata[session_id]
    logger.info(f"Session {session_id} cleared")

    return {"message": "Session cleared successfully"}