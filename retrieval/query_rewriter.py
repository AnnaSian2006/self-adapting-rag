# retrieval/query_rewriter.py

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3")

REWRITER_PROMPT = """You are a query rewriting assistant for a RAG system.
The original query failed to retrieve relevant results.
Rewrite it to be more specific, use different keywords, or break it into a clearer question.

Respond ONLY with a JSON object in this exact format, nothing else:
{"rewritten": "the new query", "strategy": "one line on what you changed"}
"""


def rewrite_query(query: str, reason: str = "") -> dict:
    """
    Rewrite a query that failed retrieval.
    reason: optional context on why retrieval failed.
    """
    user_msg = f"Original query: {query}"
    if reason:
        user_msg += f"\nReason retrieval failed: {reason}"

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model":  OLLAMA_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": REWRITER_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
        },
    )
    response.raise_for_status()
    raw = response.json()["message"]["content"].strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        result = json.loads(raw.strip())
    except json.JSONDecodeError:
        result = {"rewritten": query, "strategy": "parse failed, using original"}

    result["original"] = query
    return result


def rewrite_if_needed(query: str, scored_chunks: list, threshold: float = 0.5) -> dict:
    """
    Check if retrieval quality is poor and rewrite if needed.
    Returns dict with: should_retry, original, rewritten, reason
    """
    if not scored_chunks:
        result = rewrite_query(query, reason="no chunks retrieved at all")
        return {"should_retry": True, **result}

    avg_score = sum(c["relevance_score"] for c in scored_chunks) / len(scored_chunks)
    top_score = max(c["relevance_score"] for c in scored_chunks)

    if top_score < threshold:
        reason = f"best chunk score was only {top_score:.2f}, below threshold {threshold}"
        result = rewrite_query(query, reason=reason)
        return {"should_retry": True, "avg_score": avg_score, **result}

    return {
        "should_retry": False,
        "original":     query,
        "rewritten":    query,
        "avg_score":    avg_score,
        "top_score":    top_score,
    }


if __name__ == "__main__":
    # test rewriting on deliberately bad/vague queries
    test_cases = [
        {
            "query": "money stuff",
            "scored_chunks": [],  # simulate no results
        },
        {
            "query": "how invest",
            "scored_chunks": [
                {"relevance_score": 0.2},
                {"relevance_score": 0.15},
            ],  # simulate poor results
        },
        {
            "query": "What is dollar cost averaging?",
            "scored_chunks": [
                {"relevance_score": 0.85},
                {"relevance_score": 0.75},
            ],  # simulate good results
        },
    ]

    print("Query Rewriter Test\n" + "="*40)
    for case in test_cases:
        result = rewrite_if_needed(case["query"], case["scored_chunks"])
        print(f"Original:     {result['original']}")
        print(f"Should retry: {result['should_retry']}")
        if result["should_retry"]:
            print(f"Rewritten:    {result['rewritten']}")
            print(f"Strategy:     {result.get('strategy', '')}")
        else:
            print(f"Top score:    {result.get('top_score', '')} — no rewrite needed")
        print()