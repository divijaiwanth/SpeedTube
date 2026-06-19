"""
SpeedTube - Phase 5: AWS Bedrock Integration
=============================================
This module provides Bedrock-backed LLM and embedding clients that are
drop-in compatible with LangChain's interfaces — meaning langchain_helper.py
can swap between Groq/local and Bedrock with a single flag, no other
code changes required.

Why Bedrock for production:
  - Managed infrastructure — no GPU servers to provision or maintain
  - Pay-per-token — scales to zero when idle, scales up under load
  - IAM-based security — fits into existing AWS account permissions
  - SLA-backed availability — unlike a free-tier API like Groq

Models used:
  - amazon.titan-embed-text-v2:0   — embeddings (1024-dim)
  - amazon.nova-micro-v1:0          — generation (Amazon's own model,
                                       no AWS Marketplace subscription needed)

Note on model choice:
  Third-party models (Anthropic Claude, Meta Llama, etc.) on Bedrock go
  through AWS Marketplace and require a valid payment instrument on file,
  even when using free credits. Amazon's own models (Titan, Nova) do not —
  they're billed directly through your AWS account like any other AWS
  service. Nova Micro was chosen here specifically to avoid that
  requirement while still using genuine, working Bedrock infrastructure.
"""

import boto3
import json
import os
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from typing import Optional, Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
TITAN_EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
NOVA_MICRO_MODEL_ID = "amazon.nova-micro-v1:0"


# ---------------------------------------------------------------------------
# Bedrock client (shared across embeddings + LLM)
# ---------------------------------------------------------------------------

def get_bedrock_client():
    """
    Creates a boto3 Bedrock runtime client.
    Credentials are picked up automatically from environment variables
    (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) or from ~/.aws/credentials.
    """
    return boto3.client("bedrock-runtime", region_name=AWS_REGION)


# ---------------------------------------------------------------------------
# Titan Embeddings — LangChain-compatible wrapper
# ---------------------------------------------------------------------------

class BedrockTitanEmbeddings(Embeddings):
    """
    Drop-in replacement for HuggingFaceEmbeddings, backed by Amazon Titan
    Embeddings V2 on Bedrock.

    Usage is identical to any other LangChain Embeddings class:
        embeddings = BedrockTitanEmbeddings()
        vector = embeddings.embed_query("some text")
        vectors = embeddings.embed_documents(["doc1", "doc2"])
    """

    def __init__(self):
        self.client = get_bedrock_client()
        self.model_id = TITAN_EMBED_MODEL_ID

    def _embed(self, text: str) -> List[float]:
        body = json.dumps({"inputText": text})
        response = self.client.invoke_model(modelId=self.model_id, body=body)
        result = json.loads(response["body"].read())
        return result["embedding"]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string. Used at retrieval time."""
        return self._embed(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents. Used at index-build time.

        Note: Titan doesn't support true batch embedding in one call,
        so we loop. For large document sets you'd parallelize this
        with a thread pool to avoid sequential latency — not needed
        at this project's scale (a few hundred chunks per video).
        """
        return [self._embed(text) for text in texts]


# ---------------------------------------------------------------------------
# Claude Haiku — LangChain-compatible LLM wrapper
# ---------------------------------------------------------------------------

class BedrockNovaLLM(LLM):
    """
    Drop-in replacement for OllamaLLM / ChatGroq, backed by Amazon Nova Micro
    on Bedrock. Nova is Amazon's own model family — no AWS Marketplace
    subscription required (unlike third-party models such as Anthropic's
    Claude), which means no payment card needs to be on file.

    Usage is identical to any other LangChain LLM:
        llm = BedrockNovaLLM()
        response = llm.invoke("What is a large language model?")
    """

    model_id: str = NOVA_MICRO_MODEL_ID
    max_tokens: int = 1024
    temperature: float = 0.7

    @property
    def _llm_type(self) -> str:
        return "bedrock-nova-micro"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        client = get_bedrock_client()

        # Nova models use a different request schema than Anthropic models
        body = json.dumps({
            "messages": [
                {"role": "user", "content": [{"text": prompt}]}
            ],
            "inferenceConfig": {
                "maxTokens": self.max_tokens,
                "temperature": self.temperature,
            }
        })

        response = client.invoke_model(modelId=self.model_id, body=body)
        result = json.loads(response["body"].read())
        return result["output"]["message"]["content"][0]["text"]


# ---------------------------------------------------------------------------
# Factory functions — used by langchain_helper.py
# ---------------------------------------------------------------------------

def get_bedrock_embeddings() -> BedrockTitanEmbeddings:
    """Returns a Titan embeddings client ready to use."""
    return BedrockTitanEmbeddings()


def get_bedrock_llm(temperature: float = 0.7) -> BedrockNovaLLM:
    """Returns a Nova Micro LLM client ready to use."""
    return BedrockNovaLLM(temperature=temperature)


# ---------------------------------------------------------------------------
# Standalone test — run this file directly to verify everything works
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing Bedrock Titan embeddings...")
    embeddings = get_bedrock_embeddings()
    vec = embeddings.embed_query("This is a test sentence.")
    print(f"  Embedding dimension: {len(vec)}")

    print("\nTesting Bedrock Nova Micro...")
    llm = get_bedrock_llm()
    response = llm.invoke("Say hello in exactly 5 words.")
    print(f"  Response: {response}")

    print("\nBoth Bedrock components working correctly.")