# retrieval/query_analyzer.py

import os
import json
import requests
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.1")

QueryType = Literal["factual", "reasoning", "comparative", "ambiguous"]


SYSTEM_PROMPT = """You are a query classifier for a RAG system.
Classify the user query into exactly one of these types:

- factual: Simple fact lookup. Single, direct answer exists.
  Example: "What is dollar cost averaging?"

- reasoning: Requires multi-step thinking or explanation.
  Example: "Why does inflation affect bond prices?"

- comparative: Compares two or more things.
  Example: "What is the difference between ETFs and mutual funds?"

- ambiguous: Unclear, too vague, or needs clarification.
  Example: "Tell me about money"

Respond ONLY with a JSON object in this exact format, nothing else:
{"type": "factual", "confidence": 0.95, "reasoning": "one line explanation"}
"""


def classify_query(query: str) -> dict:
    """
    Classify a query into one of four types using Ollama.
    Returns a dict with keys: type, confidence, reasoning
    """
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model":  OLLAMA_MODEL,
            "stream": False,
            "messages": [
                {"role": "system",  "content": SYSTEM_PROMPT},
                {"role": "user",    "content": f"Classify this query: {query}"},
            ],
        },
    )
    response.raise_for_status()
    raw = response.json()["message"]["content"].strip()

    # strip markdown fences if model wraps in ```json ... ```
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    
    try:
        result = json.loads(raw.strip())
    except json.JSONDecodeError:
        # fallback if model doesn't follow instructions
        result = {"type": "factual", "confidence": 0.5, "reasoning": "parse failed, defaulting to factual"}

    # validate type
    valid_types = {"factual", "reasoning", "comparative", "ambiguous"}
    if result.get("type") not in valid_types:
        result["type"] = "factual"

    result["query"] = query
    return result


if __name__ == "__main__":
    test_queries = [
        "What is dollar cost averaging?",
        "Why does inflation reduce the value of bonds?",
        "What is the difference between a Roth IRA and a traditional IRA?",
        "Tell me about money",
        "How should I invest my savings given current market conditions?",
    ]

    print(f"Using model: {OLLAMA_MODEL}\n")
    for query in test_queries:
        result = classify_query(query)
        print(f"Q: {query}")
        print(f"   Type: {result['type']} (confidence: {result.get('confidence', '?')})")
        print(f"   Reason: {result.get('reasoning', '')}")
        print()