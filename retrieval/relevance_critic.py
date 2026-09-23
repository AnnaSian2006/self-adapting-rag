# retrieval/relevance_critic.py

import os
import json
import requests
from typing import List
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3")

CRITIC_PROMPT = """You are a relevance judge for a RAG system.
Given a query and a retrieved text chunk, score how relevant the chunk is for answering the query.

Respond ONLY with a JSON object in this exact format, nothing else:
{"score": 0.85, "relevant": true, "reason": "one line explanation"}

Scoring guide:
- 0.0 to 0.3: Not relevant at all
- 0.4 to 0.6: Partially relevant
- 0.7 to 0.9: Relevant
- 0.9 to 1.0: Highly relevant, directly answers the query

"relevant" should be true if score >= 0.5, false otherwise.
"""


def score_chunk(query: str, chunk_text: str) -> dict:
    """Score a single chunk against a query."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model":  OLLAMA_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": CRITIC_PROMPT},
                {"role": "user",   "content": f"Query: {query}\n\nChunk: {chunk_text[:500]}"},
            ],
        },
    )
    response.raise_for_status()
    raw = response.json()["message"]["content"].strip()

    # strip markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        result = json.loads(raw.strip())
    except json.JSONDecodeError:
        result = {"score": 0.5, "relevant": True, "reason": "parse failed"}

    result["chunk_text"] = chunk_text
    return result


def score_chunks(query: str, chunks: List[dict], threshold: float = 0.5) -> List[dict]:
    """
    Score all retrieved chunks and filter by threshold.
    Returns only chunks that pass the relevance threshold,
    sorted by score descending.
    """
    scored = []
    for chunk in chunks:
        critique = score_chunk(query, chunk["text"])
        chunk["relevance_score"] = critique["score"]
        chunk["relevant"]        = critique["relevant"]
        chunk["critic_reason"]   = critique.get("reason", "")
        scored.append(chunk)

    # filter + sort
    passed = [c for c in scored if c["relevance_score"] >= threshold]
    passed.sort(key=lambda x: x["relevance_score"], reverse=True)

    return passed


if __name__ == "__main__":
    from evaluation.beir_loader import load_fiqa
    from ingestion.loader import load_from_beir
    from ingestion.chunker import chunk_corpus
    from ingestion.embedder import embed_chunks, embed_text
    from retrieval.retriever_pool import RetrieverPool

    print("Setting up pipeline...")
    corpus, _, _ = load_fiqa()
    docs   = list(load_from_beir(corpus))[:50]
    chunks = chunk_corpus(docs, strategy="adaptive")
    chunks = embed_chunks(chunks)

    pool = RetrieverPool(in_memory=True)
    pool.index(chunks)

    query = "How does dollar cost averaging reduce investment risk?"
    qvec  = embed_text(query)

    # retrieve
    raw_chunks = pool.hybrid_retrieve(query, qvec, top_k=5)
    print(f"\nQuery: {query}")
    print(f"Retrieved {len(raw_chunks)} chunks, now scoring...\n")

    # score
    scored = score_chunks(query, raw_chunks, threshold=0.4)

    print(f"Chunks passing threshold: {len(scored)}/{len(raw_chunks)}\n")
    for c in scored:
        print(f"  Score: {c['relevance_score']} | Relevant: {c['relevant']}")
        print(f"  Reason: {c['critic_reason']}")
        print(f"  Text: {c['text'][:100]}...")
        print()