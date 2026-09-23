# retrieval/strategy_router.py

from typing import Literal
from retrieval.query_analyzer import classify_query

Strategy = Literal["dense", "sparse", "hybrid"]

# mapping: query type → retrieval strategy + top_k
STRATEGY_MAP = {
    "factual":     {"strategy": "dense",  "top_k": 3},
    "reasoning":   {"strategy": "hybrid", "top_k": 5},
    "comparative": {"strategy": "hybrid", "top_k": 6},
    "ambiguous":   {"strategy": "sparse", "top_k": 3},
}


def route(query: str) -> dict:
    """
    Classify query and return the retrieval strategy to use.
    Returns dict with: query, type, strategy, top_k, confidence
    """
    classification = classify_query(query)
    query_type     = classification["type"]
    config         = STRATEGY_MAP[query_type]

    return {
        "query":      query,
        "type":       query_type,
        "strategy":   config["strategy"],
        "top_k":      config["top_k"],
        "confidence": classification.get("confidence", 0.5),
        "reasoning":  classification.get("reasoning", ""),
    }


if __name__ == "__main__":
    test_queries = [
        "What is dollar cost averaging?",
        "Why does inflation reduce the value of bonds?",
        "What is the difference between a Roth IRA and a traditional IRA?",
        "Tell me about money",
    ]

    print("Strategy Router Test\n" + "="*40)
    for query in test_queries:
        result = route(query)
        print(f"Q: {query}")
        print(f"   Type:     {result['type']}")
        print(f"   Strategy: {result['strategy']} (top_k={result['top_k']})")
        print(f"   Confidence: {result['confidence']}")
        print()