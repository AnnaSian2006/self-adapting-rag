# retrieval/hyde.py

import os
import requests
from dotenv import load_dotenv
from ingestion.embedder import embed_text

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3")

HYDE_PROMPT = """You are a financial document expert.
Write a short, factual paragraph (3-5 sentences) that would DIRECTLY answer this question.
Write it as if it came from a professional financial document or textbook.
Be specific and use domain terminology.
Do NOT say you don't know — always write a plausible answer.

Question: {query}

Paragraph:"""


def generate_hypothetical_document(query: str) -> str:
    """
    Generate a hypothetical answer document for the query.
    This hypothetical doc is then embedded and used for retrieval
    instead of the raw query — dramatically improves recall.
    """
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model":  OLLAMA_MODEL,
            "stream": False,
            "messages": [
                {
                    "role":    "user",
                    "content": HYDE_PROMPT.format(query=query),
                }
            ],
        },
    )
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


def hyde_embed(query: str) -> tuple:
    """
    Generate hypothetical document and embed it.
    Returns (hypothetical_doc, hyde_embedding).
    The hyde_embedding is used for retrieval instead of query embedding.
    """
    hyp_doc  = generate_hypothetical_document(query)
    hyp_emb  = embed_text(hyp_doc)
    return hyp_doc, hyp_emb


if __name__ == "__main__":
    from ingestion.embedder import embed_text as raw_embed
    import numpy as np

    test_queries = [
        "How do taxes work for a sole proprietorship?",
        "What is dollar cost averaging?",
        "How does inflation affect bond prices?",
    ]

    print("HyDE Test\n" + "="*40)

    for query in test_queries:
        print(f"\nQuery: {query}")

        # generate hypothetical document
        hyp_doc, hyp_emb = hyde_embed(query)
        print(f"Hypothetical doc:\n  {hyp_doc[:200]}...")

        # compare raw query embedding vs HyDE embedding
        raw_emb = raw_embed(query)
        similarity = float(
            np.dot(raw_emb, hyp_emb) /
            (np.linalg.norm(raw_emb) * np.linalg.norm(hyp_emb) + 1e-8)
        )
        print(f"Similarity (raw vs HyDE): {similarity:.3f}")
        print()