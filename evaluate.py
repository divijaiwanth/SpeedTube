"""
RAGTube - Phase 2: RAG Evaluation
==================================
What this file does:
  1. Generates a synthetic Q&A dataset from the video transcript using an LLM
  2. Runs your Phase 1 RAG pipeline against every question
  3. Scores each answer using RAGAS metrics:
       - Faithfulness        : does the answer only use information from the retrieved context?
       - Answer Relevancy    : is the answer actually relevant to the question?
       - Context Precision   : are the retrieved chunks ranked well (relevant ones first)?
       - Context Recall      : did retrieval actually find the chunks needed to answer?
  4. Logs every run + config to MLflow so you can compare experiments
"""

import json
import time
from dataclasses import dataclass, asdict
from typing import Optional

import mlflow
import mlflow.data
import pandas as pd
from datasets import Dataset
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_helper import create_video_index, get_answer, VideoIndex
import os
from dotenv import load_dotenv

load_dotenv()
# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
GrokAPI = os.getenv("GROQ_API_KEY")
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GrokAPI, temperature=0.7)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvalSample:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str


@dataclass
class EvalConfig:
    video_id: str
    retrieval_k: int = 10
    rerank_top_n: int = 3
    chunk_size_child: int = 300
    chunk_size_parent: int = 1200
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "BAAI/bge-reranker-base"
    llm_model: str = "llama-3.1-8b-instant"
    hyde_enabled: bool = True
    hybrid_search_enabled: bool = True
    num_eval_questions: int = 10


# ---------------------------------------------------------------------------
# Step 1 — Synthetic dataset generation
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(index: VideoIndex, n_questions: int = 10) -> list[dict]:
    parent_docs = index.parent_docs
    total_chunks = len(parent_docs)

    if total_chunks <= n_questions:
        sampled_chunks = parent_docs
    else:
        step = total_chunks // n_questions
        sampled_chunks = [parent_docs[i * step] for i in range(n_questions)]

    generation_prompt = PromptTemplate.from_template(
        """You are creating an evaluation dataset for a RAG system.

Given this excerpt from a YouTube video transcript, generate ONE factual question
that can be answered directly from this excerpt, and provide the correct answer.

Rules:
- The question must be answerable from ONLY this excerpt
- The answer must be a complete sentence, not just a word or phrase
- Do not ask vague questions like "What is this about?"
- Focus on specific facts, explanations, or claims made in the excerpt

Transcript excerpt:
{chunk}

Respond in this exact JSON format (no other text):
{{
  "question": "your question here",
  "ground_truth": "the correct answer here"
}}"""
    )

    chain = generation_prompt | llm | StrOutputParser()
    dataset = []
    print(f"Generating {len(sampled_chunks)} synthetic Q&A pairs...")

    for i, chunk in enumerate(sampled_chunks):
        try:
            raw = chain.invoke({"chunk": chunk.page_content})
            raw = raw.strip().strip("```json").strip("```").strip()
            parsed = json.loads(raw)

            if "question" in parsed and "ground_truth" in parsed:
                dataset.append({
                    "question": parsed["question"],
                    "ground_truth": parsed["ground_truth"],
                    "source_chunk": chunk.page_content[:200],
                })
                print(f"  [{i+1}/{len(sampled_chunks)}] ✓ {parsed['question'][:60]}...")
            else:
                print(f"  [{i+1}/{len(sampled_chunks)}] ✗ Malformed response, skipping")

        except Exception as e:
            print(f"  [{i+1}/{len(sampled_chunks)}] ✗ Error: {e}, skipping")
            continue

    print(f"Generated {len(dataset)} valid Q&A pairs\n")
    return dataset


# ---------------------------------------------------------------------------
# Step 2 — Run pipeline against each question
# ---------------------------------------------------------------------------

def collect_rag_responses(index: VideoIndex, dataset: list[dict]) -> list[EvalSample]:
    samples = []
    print(f"Running RAG pipeline on {len(dataset)} questions...")

    for i, item in enumerate(dataset):
        question = item["question"]
        ground_truth = item["ground_truth"]

        try:
            result = get_answer(index, question)
            sample = EvalSample(
                question=question,
                answer=result["answer"],
                contexts=result["sources"],
                ground_truth=ground_truth,
            )
            samples.append(sample)
            print(f"  [{i+1}/{len(dataset)}] ✓ Answered: {question[:50]}...")

        except Exception as e:
            print(f"  [{i+1}/{len(dataset)}] ✗ Pipeline error: {e}, skipping")
            continue

    print(f"Collected {len(samples)} samples for evaluation\n")
    return samples


# ---------------------------------------------------------------------------
# Step 3 — RAGAS scoring
# ---------------------------------------------------------------------------

def run_ragas_evaluation(samples: list[EvalSample]) -> tuple[dict, pd.DataFrame]:
    from ragas import EvaluationDataset, SingleTurnSample, evaluate as ragas_evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.run_config import RunConfig
    from openai import OpenAI
    import instructor
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI
    ragas_llm = LangchainLLMWrapper(
    ChatOpenAI(
        model="llama-3.1-8b-instant",
        api_key=GrokAPI,
        base_url="https://api.groq.com/openai/v1"
    )
    )
    '''
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    ragas_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="gpt-4.1",  # or gpt-4.1-mini, gpt-4o
            api_key=OPENAI_API_KEY,
            temperature=0
        )
    )
    '''

    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    ragas_samples = [
        SingleTurnSample(
            user_input=s.question,
            response=s.answer,
            retrieved_contexts=s.contexts,
            reference=s.ground_truth,
        )
        for s in samples
    ]
    dataset = EvaluationDataset(samples=ragas_samples)

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]

    print("Running RAGAS evaluation...")
    result = ragas_evaluate(
        dataset, metrics=metrics,
        run_config=RunConfig(max_workers=1, timeout=120)
    )
    results_df = result.to_pandas()

    scores = {
        "faithfulness":      round(float(results_df["faithfulness"].mean()), 4),
        "answer_relevancy":  round(float(results_df["answer_relevancy"].mean()), 4),
        "context_precision": round(float(results_df["context_precision"].mean()), 4),
        "context_recall":    round(float(results_df["context_recall"].mean()), 4),
    }
    scores["composite_score"] = round(sum(scores.values()) / len(scores), 4)
    return scores, results_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_evaluation(
    video_id: str,
    n_questions: int = 10,
    run_name: Optional[str] = None,
) -> dict:
    print(f"\n{'='*60}")
    print(f"RAGTube Evaluation Pipeline")
    print(f"Video: {video_id} | Questions: {n_questions}")
    print(f"{'='*60}\n")

    config = EvalConfig(video_id=video_id, num_eval_questions=n_questions)

    print("Building video index...")
    index = create_video_index(video_id)
    print(f"Index built: {len(index.child_docs)} child chunks\n")

    dataset = generate_synthetic_dataset(index, n_questions=n_questions)
    if not dataset:
        raise ValueError("No Q&A pairs generated.")

    samples = collect_rag_responses(index, dataset)
    if not samples:
        raise ValueError("No RAG responses collected.")

    scores, results_df = run_ragas_evaluation(samples)
    run_id = log_to_mlflow(config, scores, results_df, samples, run_name)

    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    for metric, score in scores.items():
        try:
            bar = "█" * int(float(score) * 20)
        except (ValueError, TypeError):
            bar = "N/A"
        print(f"  {metric:<22} {score}  {bar}")
    print(f"{'='*60}\n")

    return {**scores, "run_id": run_id}

def log_to_mlflow(config, scores, results_df, samples, run_name=None):
    import tempfile, os
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("RAGTube-Evaluation")

    with mlflow.start_run(run_name=run_name or f"eval_{config.video_id}_{int(time.time())}"):
        mlflow.log_params(asdict(config))
        mlflow.log_metrics(scores)

        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = os.path.join(tmpdir, f"ragas_results.csv")
            results_df.to_csv(results_path, index=False)
            mlflow.log_artifact(results_path, artifact_path="evaluation")

        run_id = mlflow.active_run().info.run_id

    print(f"\nMLflow run logged: {run_id}")
    return run_id

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <youtube_video_id> [n_questions]")
        print("Example: python evaluate.py zjkBMFhNj_g 4")
        sys.exit(1)

    video_id = sys.argv[1]
    n_questions = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    scores = run_evaluation(video_id, n_questions=n_questions)