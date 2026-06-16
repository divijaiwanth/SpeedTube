"""
SpeedTube - Phase 4: Production Guardrails
==========================================
What's new in Phase 4:
  1. Rate limiting (slowapi)        — 10 req/min on /query, 5 req/min on /ingest
  2. Redis caching                  — FAISS indexes survive server restarts
  3. Input guardrails               — reject irrelevant/empty/malicious queries
  4. Circuit breaker                — stop hammering Groq when it's failing
  5. Structured logging (loguru)    — proper log levels, searchable output

Why each one matters:
  - Rate limiting   : prevents abuse, protects your Groq free tier quota
  - Redis caching   : same video = instant response, no re-embedding
  - Guardrails      : LLM calls cost time/money, don't waste them on garbage input
  - Circuit breaker : fail fast instead of queuing up 50 requests to a dead API
  - Loguru          : in prod you need searchable logs, not print statements
"""

import os
import time
import pickle
import hashlib
from typing import Optional
from urllib.parse import urlparse, parse_qs

# ---------------------------------------------------------------------------
# Loguru — structured logging (replaces standard logging)
# ---------------------------------------------------------------------------
# Why loguru over standard logging:
#   - Automatic log levels with color in terminal
#   - One-line setup instead of 5 lines of boilerplate
#   - Structured JSON output for production (CloudWatch, Datadog etc.)
#   - Better exception formatting

from loguru import logger

logger.add(
    "../logs/speedtube.log",          # log to file
    rotation="10 MB",              # rotate when file hits 10MB
    retention="7 days",            # keep 7 days of logs
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} | {message}",
)

# ---------------------------------------------------------------------------
# FastAPI + rate limiting
# ---------------------------------------------------------------------------

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Rate limiter — uses client IP address as the key
# In production you'd use user ID or API key instead
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="SpeedTube API",
    description="Get up to speed on any YouTube video — advanced RAG pipeline",
    version="3.0.0",
)

# Attach rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Redis client — for persistent session caching
# ---------------------------------------------------------------------------
# Why Redis over in-memory dict:
#   - Survives server restarts (in-memory dict resets every restart)
#   - Can be shared across multiple server instances (horizontal scaling)
#   - TTL support — automatically expire old sessions
#   - In Phase 5 you'll use AWS ElastiCache (managed Redis) instead of local

import redis

try:
    redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=False)
    redis_client.ping()
    REDIS_AVAILABLE = True
    logger.info("Redis connected successfully")
except Exception as e:
    REDIS_AVAILABLE = False
    logger.warning(f"Redis not available, falling back to in-memory cache: {e}")

# Fallback in-memory cache if Redis isn't available
_memory_cache: dict = {}

SESSION_TTL = 60 * 60 * 3  # 3 hours — sessions expire after 3 hours of inactivity


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------
# Problem it solves:
#   If Groq is down and you keep sending requests, you're:
#     - Wasting time waiting for timeouts (30s each)
#     - Potentially getting rate limited even harder
#     - Queuing up requests that will all fail anyway
#
# The circuit breaker has 3 states:
#   CLOSED   → normal operation, requests go through
#   OPEN     → too many failures, requests blocked immediately (fail fast)
#   HALF-OPEN → after cooldown, try one request to see if service recovered
#
# Think of it like a physical circuit breaker in your house —
# too much current → breaker trips → you reset it → normal again

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, cooldown_seconds: int = 60):
        self.failure_threshold = failure_threshold  # failures before opening
        self.cooldown_seconds = cooldown_seconds    # seconds to wait before trying again
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"   # CLOSED, OPEN, HALF_OPEN

    def call_succeeded(self):
        """Call this when an LLM call succeeds."""
        self.failure_count = 0
        self.state = "CLOSED"
        logger.debug("Circuit breaker: call succeeded, state = CLOSED")

    def call_failed(self):
        """Call this when an LLM call fails."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(
                f"Circuit breaker OPENED after {self.failure_count} failures. "
                f"Blocking requests for {self.cooldown_seconds}s"
            )

    def can_proceed(self) -> bool:
        """Returns True if the request should proceed, False if it should be blocked."""
        if self.state == "CLOSED":
            return True

        if self.state == "OPEN":
            # Check if cooldown period has passed
            time_since_failure = time.time() - self.last_failure_time
            if time_since_failure >= self.cooldown_seconds:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker: cooldown passed, state = HALF_OPEN, trying one request")
                return True
            return False

        if self.state == "HALF_OPEN":
            return True  # Let one request through to test

        return True


# One circuit breaker per external service
groq_circuit_breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=60)


# ---------------------------------------------------------------------------
# Input Guardrails
# ---------------------------------------------------------------------------
# Why guardrails matter:
#   - LLM calls take 2-5 seconds and consume API quota
#   - Don't waste them on empty questions, gibberish, or prompt injections
#   - Catch obviously bad inputs before they hit the expensive pipeline
#
# This is a lightweight rule-based guardrail.
# In Phase 5 you could add an LLM-based guardrail that checks
# if the question is actually related to the video topic.

class InputGuardrail:
    # Questions that are clearly trying to jailbreak or misuse the system
    BLOCKED_PATTERNS = [
        "ignore previous instructions",
        "ignore all instructions",
        "you are now",
        "pretend you are",
        "forget everything",
        "system prompt",
        "jailbreak",
    ]

    # Minimum meaningful question length
    MIN_LENGTH = 5
    MAX_LENGTH = 500

    @classmethod
    def validate_question(cls, question: str) -> tuple[bool, str]:
        """
        Returns (is_valid, error_message).
        is_valid = True means the question passed all checks.
        """
        # Check empty
        if not question or not question.strip():
            return False, "Question cannot be empty."

        question_lower = question.lower().strip()

        # Check length
        if len(question_lower) < cls.MIN_LENGTH:
            return False, "Question is too short. Please ask a complete question."

        if len(question_lower) > cls.MAX_LENGTH:
            return False, f"Question is too long. Keep it under {cls.MAX_LENGTH} characters."

        # Check for prompt injection attempts
        for pattern in cls.BLOCKED_PATTERNS:
            if pattern in question_lower:
                logger.warning(f"Blocked potential prompt injection: {question[:50]}")
                return False, "That type of question isn't supported."

        # Check it's actually a question or statement (not just symbols/numbers)
        alpha_chars = sum(1 for c in question if c.isalpha())
        if alpha_chars < 3:
            return False, "Please ask a question in plain text."

        return True, ""

    @classmethod
    def validate_url(cls, url: str) -> tuple[bool, str]:
        """Validate that the URL is actually a YouTube URL."""
        if not url or not url.strip():
            return False, "URL cannot be empty."

        url_lower = url.lower().strip()

        if "youtube.com" not in url_lower and "youtu.be" not in url_lower:
            return False, "Only YouTube URLs are supported."

        if len(url) > 200:
            return False, "URL is too long."

        return True, ""


guardrail = InputGuardrail()


# ---------------------------------------------------------------------------
# Session cache helpers (Redis + memory fallback)
# ---------------------------------------------------------------------------

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_helper import create_video_index, get_answer, get_summary, VideoIndex


def make_session_id(video_id: str) -> str:
    return hashlib.md5(video_id.encode()).hexdigest()[:12]


def extract_video_id(url: str) -> Optional[str]:
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


def cache_set(session_id: str, index: VideoIndex, metadata: dict):
    """Store index in Redis (pickled) or memory fallback."""
    if REDIS_AVAILABLE:
        try:
            # Pickle the VideoIndex object to store in Redis
            # We store index and metadata separately
            redis_client.setex(
                f"index:{session_id}",
                SESSION_TTL,
                pickle.dumps(index)
            )
            redis_client.setex(
                f"meta:{session_id}",
                SESSION_TTL,
                pickle.dumps(metadata)
            )
            logger.info(f"Session {session_id} cached in Redis (TTL: {SESSION_TTL}s)")
        except Exception as e:
            logger.error(f"Redis write failed: {e}, falling back to memory")
            _memory_cache[session_id] = {"index": index, "metadata": metadata}
    else:
        _memory_cache[session_id] = {"index": index, "metadata": metadata}


def cache_get(session_id: str) -> tuple[Optional[VideoIndex], Optional[dict]]:
    """Retrieve index from Redis or memory fallback."""
    if REDIS_AVAILABLE:
        try:
            index_bytes = redis_client.get(f"index:{session_id}")
            meta_bytes = redis_client.get(f"meta:{session_id}")

            if index_bytes and meta_bytes:
                # Reset TTL on access (sliding expiry)
                redis_client.expire(f"index:{session_id}", SESSION_TTL)
                redis_client.expire(f"meta:{session_id}", SESSION_TTL)
                return pickle.loads(index_bytes), pickle.loads(meta_bytes)
        except Exception as e:
            logger.error(f"Redis read failed: {e}")

    # Fallback to memory
    if session_id in _memory_cache:
        cached = _memory_cache[session_id]
        return cached["index"], cached["metadata"]

    return None, None


def cache_delete(session_id: str):
    """Delete session from Redis or memory."""
    if REDIS_AVAILABLE:
        redis_client.delete(f"index:{session_id}")
        redis_client.delete(f"meta:{session_id}")
    if session_id in _memory_cache:
        del _memory_cache[session_id]


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    url: str

class IngestResponse(BaseModel):
    session_id: str
    video_id: str
    chunk_count: int
    message: str
    cached: bool

class QueryRequest(BaseModel):
    session_id: str
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    num_sources: int

class SummaryResponse(BaseModel):
    summary: str
    summary_type: str

class HealthResponse(BaseModel):
    status: str
    redis_connected: bool
    circuit_breaker_state: str
    uptime_seconds: float


START_TIME = time.time()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(
        status="ok",
        redis_connected=REDIS_AVAILABLE,
        circuit_breaker_state=groq_circuit_breaker.state,
        uptime_seconds=round(time.time() - START_TIME, 2),
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Video"])
@limiter.limit("5/minute")   # max 5 video loads per minute per IP
async def ingest_video(request: Request, body: IngestRequest):
    """
    Load a YouTube video and build the RAG index.
    Rate limited to 5 requests/minute per IP.
    Idempotent — same URL always returns same session_id.
    """
    # Guardrail — validate URL
    is_valid, error = guardrail.validate_url(body.url)
    if not is_valid:
        logger.warning(f"Invalid URL rejected: {body.url[:50]} — {error}")
        raise HTTPException(status_code=400, detail=error)

    video_id = extract_video_id(body.url)
    if not video_id:
        raise HTTPException(
            status_code=400,
            detail="Could not extract video ID. Make sure it's a valid YouTube URL."
        )

    session_id = make_session_id(video_id)

    # Check cache first (Redis or memory)
    cached_index, cached_meta = cache_get(session_id)
    if cached_index is not None:
        logger.info(f"Cache HIT for video {video_id}, session {session_id}")
        return IngestResponse(
            session_id=session_id,
            video_id=video_id,
            chunk_count=cached_meta["chunk_count"],
            message="Video already loaded — using cached index",
            cached=True,
        )

    # Build the index
    logger.info(f"Cache MISS — building index for video {video_id}")
    try:
        index = create_video_index(video_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to build index for {video_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process video: {str(e)}")

    metadata = {
        "video_id": video_id,
        "chunk_count": len(index.child_docs),
        "created_at": time.time(),
    }
    cache_set(session_id, index, metadata)

    logger.info(f"Index built and cached: {len(index.child_docs)} chunks, session {session_id}")

    return IngestResponse(
        session_id=session_id,
        video_id=video_id,
        chunk_count=len(index.child_docs),
        message="Video loaded successfully",
        cached=False,
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
@limiter.limit("10/minute")   # max 10 questions per minute per IP
async def query_video(request: Request, body: QueryRequest):
    """
    Ask a question about a loaded video.
    Rate limited to 10 requests/minute per IP.
    Input guardrails reject empty/malicious questions.
    Circuit breaker blocks requests if Groq is failing.
    """
    # Guardrail — validate question
    is_valid, error = guardrail.validate_question(body.question)
    if not is_valid:
        logger.warning(f"Invalid question rejected: {body.question[:50]} — {error}")
        raise HTTPException(status_code=400, detail=error)

    # Circuit breaker check
    if not groq_circuit_breaker.can_proceed():
        logger.warning("Circuit breaker OPEN — blocking request to Groq")
        raise HTTPException(
            status_code=503,
            detail="AI service temporarily unavailable. Please try again in a minute."
        )

    # Get session from cache
    index, _ = cache_get(body.session_id)
    if index is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please call /ingest first to load the video."
        )

    logger.info(f"Query: session={body.session_id}, question={body.question[:50]}")

    try:
        result = get_answer(index, body.question)
        groq_circuit_breaker.call_succeeded()
    except Exception as e:
        groq_circuit_breaker.call_failed()
        logger.error(f"Query failed: {e}, circuit breaker failures={groq_circuit_breaker.failure_count}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        num_sources=result["num_sources"],
    )


@app.get("/summary", response_model=SummaryResponse, tags=["RAG"])
@limiter.limit("5/minute")
async def get_video_summary(
    request: Request,
    session_id: str,
    summary_type: str = "concise"
):
    """
    Get a summary of a loaded video.
    Rate limited to 5 requests/minute per IP.
    """
    # Circuit breaker check
    if not groq_circuit_breaker.can_proceed():
        raise HTTPException(
            status_code=503,
            detail="AI service temporarily unavailable. Please try again in a minute."
        )

    valid_types = ["concise", "detailed", "bullets"]
    if summary_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid summary_type. Must be one of: {valid_types}"
        )

    index, _ = cache_get(session_id)
    if index is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please call /ingest first."
        )

    logger.info(f"Summary request: session={session_id}, type={summary_type}")

    try:
        summary = get_summary(index, summary_type)
        groq_circuit_breaker.call_succeeded()
    except Exception as e:
        groq_circuit_breaker.call_failed()
        logger.error(f"Summary failed: {e}")
        raise HTTPException(status_code=500, detail=f"Summary failed: {str(e)}")

    return SummaryResponse(summary=summary, summary_type=summary_type)


@app.delete("/session/{session_id}", tags=["System"])
async def clear_session(session_id: str):
    """Clear a session from cache."""
    index, _ = cache_get(session_id)
    if index is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    cache_delete(session_id)
    logger.info(f"Session {session_id} cleared")
    return {"message": "Session cleared successfully"}