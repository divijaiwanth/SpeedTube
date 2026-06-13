"""
RAGTube - Advanced RAG Pipeline
Upgrades over v1:
  - Parent-child chunking (better context preservation)
  - Hybrid search: BM25 (keyword) + FAISS (semantic), fused with RRF
  - HyDE: Hypothetical Document Embedding for better query representation
  - Cross-encoder reranking (BGE reranker) to re-score top candidates
  - Map-reduce summarisation (not just Q&A)
  - Multi-turn conversational memory
"""

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_classic.memory import ConversationBufferWindowMemory
from sentence_transformers import CrossEncoder
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Model setup  (loaded once at module level — not inside functions)
# ---------------------------------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

import os
from dotenv import load_dotenv

load_dotenv()
# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
GrokAPI = os.getenv("GROQ_API_KEY")
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GrokAPI, temperature=0.7)

# Cross-encoder for reranking — local, no API key needed
# BGE reranker is small (~120MB) and very accurate
reranker = CrossEncoder("BAAI/bge-reranker-base")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VideoIndex:
    """Everything we need to query a single video."""
    video_id: str
    parent_docs: list[Document]      # large chunks sent to LLM
    child_docs: list[Document]       # small chunks used for retrieval
    faiss_db: FAISS                  # vector index over child chunks
    bm25_retriever: BM25Retriever    # keyword index over child chunks
    memory: ConversationBufferWindowMemory = field(
        default_factory=lambda: ConversationBufferWindowMemory(k=5, return_messages=True)
    )


# ---------------------------------------------------------------------------
# Step 1 — Ingest
# ---------------------------------------------------------------------------

def create_video_index(video_id: str) -> VideoIndex:
    """
    Ingests a YouTube video and builds the full retrieval index.

    Parent-child chunking strategy:
      - Child chunks (300 chars, 50 overlap)  → used for retrieval (precise matching)
      - Parent chunks (1200 chars, 200 overlap) → sent to LLM (full context)

    Each child doc stores the index of its parent so we can look it up after retrieval.
    """
    # --- 1a. Fetch transcript ---
    from youtube_transcript_api.proxies import GenericProxyConfig
    ytt_api = YouTubeTranscriptApi()
    try:
        import xml.etree.ElementTree as ET
        transcript = ytt_api.fetch(video_id)
    except ET.ParseError:
        raise ValueError(
            "Could not parse the transcript. This usually happens if the video "
            "has no subtitles available, or if YouTube returned an empty response. "
            "Try a different video."
        )

    # Preserve timestamps as metadata — useful later for chapter detection
    segments = [
        Document(
            page_content=t.text,
            metadata={"start": t.start, "duration": t.duration}
        )
        for t in transcript
    ]
    full_text = " ".join([t.text for t in transcript])
    raw_doc = Document(page_content=full_text, metadata={"video_id": video_id})

    # --- 1b. Create parent chunks (large, for LLM context) ---
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    parent_docs = parent_splitter.split_documents([raw_doc])
    for i, doc in enumerate(parent_docs):
        doc.metadata["parent_id"] = i

    # --- 1c. Create child chunks (small, for precise retrieval) ---
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    child_docs = []
    for parent_id, parent_doc in enumerate(parent_docs):
        children = child_splitter.split_documents([parent_doc])
        for child in children:
            child.metadata["parent_id"] = parent_id   # link back to parent
        child_docs.extend(children)

    # --- 1d. Build FAISS index on child chunks ---
    faiss_db = FAISS.from_documents(child_docs, embeddings)

    # --- 1e. Build BM25 index on child chunks ---
    bm25_retriever = BM25Retriever.from_documents(child_docs)
    bm25_retriever.k = 60

    return VideoIndex(
        video_id=video_id,
        parent_docs=parent_docs,
        child_docs=child_docs,
        faiss_db=faiss_db,
        bm25_retriever=bm25_retriever,
    )


# ---------------------------------------------------------------------------
# Step 2 — HyDE (Hypothetical Document Embedding)
# ---------------------------------------------------------------------------

def generate_hypothetical_answer(query: str) -> str:
    """
    HyDE: Ask the LLM to generate a *hypothetical* answer to the query,
    then embed that answer instead of the raw query.

    Why: A hypothetical answer is lexically closer to the actual transcript
    text than a short question, so cosine similarity works better.
    """
    hyde_prompt = PromptTemplate.from_template(
        """Write a short paragraph (3-5 sentences) that would answer this question 
about a YouTube video transcript. Write as if you are excerpting from the transcript.
Do NOT mention that this is hypothetical.

Question: {question}

Hypothetical excerpt:"""
    )
    chain = hyde_prompt | llm | StrOutputParser()
    return chain.invoke({"question": query})


# ---------------------------------------------------------------------------
# Step 3 — Hybrid retrieval (BM25 + FAISS via RRF fusion)
# ---------------------------------------------------------------------------

def hybrid_retrieve(index: VideoIndex, query: str, k: int = 60) -> list[Document]:
    """
    Reciprocal Rank Fusion (RRF) of BM25 and FAISS results.

    RRF score = sum over each ranker of: 1 / (rank + 60)
    The constant 60 dampens the effect of high ranks.

    Returns the top-k child docs after fusion.
    """
    # Vector search using HyDE-transformed query
    hypothetical = generate_hypothetical_answer(query)
    # To this — skip HyDE, use query directly
    vector_docs = index.faiss_db.similarity_search(hypothetical, k=k)

    # Keyword search using original query (BM25 is better with exact terms)
    keyword_docs = index.bm25_retriever.invoke(query)[:k]

    # RRF fusion
    doc_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    def rrf_score(rank: int, k: int = 60) -> float:
        return 1.0 / (rank + k)

    for rank, doc in enumerate(vector_docs):
        key = doc.page_content[:100]   # use first 100 chars as dedup key
        doc_scores[key] = doc_scores.get(key, 0) + rrf_score(rank)
        doc_map[key] = doc

    for rank, doc in enumerate(keyword_docs):
        key = doc.page_content[:100]
        doc_scores[key] = doc_scores.get(key, 0) + rrf_score(rank)
        doc_map[key] = doc

    # Sort by fused score descending
    sorted_keys = sorted(doc_scores, key=lambda x: doc_scores[x], reverse=True)
    return [doc_map[k] for k in sorted_keys[:k]]


# ---------------------------------------------------------------------------
# Step 4 — Cross-encoder reranking
# ---------------------------------------------------------------------------

def rerank(query: str, docs: list[Document], top_n: int = 3) -> list[Document]:
    """
    Cross-encoder reranking: pass every (query, doc) pair through a small
    transformer that scores relevance jointly (not just cosine similarity).

    We retrieve 10 via hybrid search, rerank all 10, keep top_n=3.
    """
    if not docs:
        return []

    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)                  # shape: (len(docs),)

    # Attach scores and sort
    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_n]]


# ---------------------------------------------------------------------------
# Step 5 — Parent lookup (child → parent)
# ---------------------------------------------------------------------------

def fetch_parent_docs(index: VideoIndex, child_docs: list[Document]) -> list[Document]:
    """
    Given retrieved child docs, return their parent docs (deduplicated).
    This gives the LLM more context than the small child chunks.
    """
    seen_parent_ids = set()
    parents = []
    for child in child_docs:
        parent_id = child.metadata.get("parent_id")
        if parent_id is not None and parent_id not in seen_parent_ids:
            seen_parent_ids.add(parent_id)
            parents.append(index.parent_docs[parent_id])
    return parents


# ---------------------------------------------------------------------------
# Step 6 — Q&A with conversational memory
# ---------------------------------------------------------------------------

def get_answer(index: VideoIndex, query: str) -> dict:
    """
    Full advanced RAG pipeline:
      query → HyDE → hybrid retrieval → reranking → parent lookup → LLM

    Returns dict with answer + retrieved source snippets for transparency.
    """
    # Retrieve + rerank
    candidate_docs = hybrid_retrieve(index, query, k=60)
    top_child_docs = rerank(query, candidate_docs, top_n=3)
    context_docs = fetch_parent_docs(index, top_child_docs)

    context_text = "\n\n---\n\n".join([doc.page_content for doc in context_docs])

    # Pull conversation history from memory
    history = index.memory.load_memory_variables({})
    chat_history = history.get("history", "")

    qa_prompt = PromptTemplate.from_template(
        """You are a helpful YouTube assistant that answers questions about video transcripts.

Conversation so far:
{chat_history}

Relevant transcript excerpts:
{context}

Current question: {question}

Instructions:
- Answer only from the transcript excerpts above.
- If the answer is not in the transcript, say "I couldn't find that in this video."
- Be concise and direct.

Answer:"""
    )

    chain = qa_prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "question": query,
        "context": context_text,
        "chat_history": chat_history,
    })
    answer = answer.strip()

    # Save to memory
    index.memory.save_context({"input": query}, {"output": answer})

    return {
        "answer": answer,
        "sources": [doc.page_content[:200] + "..." for doc in context_docs],
        "num_sources": len(context_docs),
    }


# ---------------------------------------------------------------------------
# Step 7 — Summarisation (map-reduce)
# ---------------------------------------------------------------------------

def get_summary(index: VideoIndex, summary_type: str = "concise") -> str:
    """
    Map-reduce summarisation over the full transcript.

    summary_type options:
      - "concise"   : 3-5 sentence overview
      - "detailed"  : paragraph per main topic
      - "bullets"   : bullet-point key takeaways
    """
    style_instructions = {
        "concise": "Summarise in 3-5 sentences covering the main topic and key points.",
        "detailed": "Write a detailed summary with one paragraph per major topic covered.",
        "bullets": "List the 5-7 most important takeaways as bullet points.",
    }
    instruction = style_instructions.get(summary_type, style_instructions["concise"])

    map_prompt = PromptTemplate.from_template(
        """Summarise this transcript excerpt briefly:

{text}

Brief summary:"""
    )

    combine_prompt = PromptTemplate.from_template(
        f"""You have been given summaries of different parts of a YouTube video transcript.
{instruction}

Summaries:
{{text}}

Final summary:"""
    )

    summarise_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        verbose=False,
    )

    result = summarise_chain.invoke({"input_documents": index.parent_docs})
    return result["output_text"].strip()


# ---------------------------------------------------------------------------
# Step 8 — Key quotes extraction
# ---------------------------------------------------------------------------

def get_key_quotes(index: VideoIndex, n: int = 5) -> list[str]:
    """
    Extract the most memorable or insightful quotes from the transcript.
    """
    # Use a sample of parent docs to stay within context limits
    sample_docs = index.parent_docs[:min(10, len(index.parent_docs))]
    context = "\n\n".join([doc.page_content for doc in sample_docs])

    prompt = PromptTemplate.from_template(
        """From the following YouTube transcript, extract the {n} most notable, 
insightful, or memorable quotes. Return them as a numbered list.
Only use exact words from the transcript.

Transcript:
{context}

Top {n} quotes:"""
    )

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"context": context, "n": n})
    return result.strip()