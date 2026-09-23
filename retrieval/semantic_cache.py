# retrieval/semantic_cache.py

import numpy as np
from typing import Optional
from ingestion.embedder import embed_text


class SemanticCache:
    """
    Caches query results by semantic similarity.
    If a new query is similar enough to a cached one,
    return the cached answer instantly — no retrieval needed.
    """

    def __init__(self, similarity_threshold: float = 0.92):
        self.threshold = similarity_threshold
        self.cache: list = []  # list of {embedding, query, result}
        self.hits   = 0
        self.misses = 0

    def _cosine_similarity(self, a: list, b: list) -> float:
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def get(self, query: str, query_embedding: list) -> Optional[dict]:
        """
        Look up a query in the cache.
        Returns cached result if similar query exists, else None.
        """
        best_score  = 0
        best_result = None

        for entry in self.cache:
            score = self._cosine_similarity(query_embedding, entry["embedding"])
            if score > best_score:
                best_score  = score
                best_result = entry

        if best_score >= self.threshold and best_result:
            self.hits += 1
            print(f"[CACHE HIT] similarity={best_score:.3f} — returning cached answer")
            print(f"  Original query: '{best_result['query']}'")
            return best_result["result"]

        self.misses += 1
        return None

    def set(self, query: str, query_embedding: list, result: dict):
        """Store a query result in the cache."""
        self.cache.append({
            "query":     query,
            "embedding": query_embedding,
            "result":    result,
        })

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits":       self.hits,
            "misses":     self.misses,
            "total":      total,
            "hit_rate":   round(self.hits / max(total, 1), 3),
            "cache_size": len(self.cache),
        }

    def clear(self):
        self.cache = []
        self.hits  = 0
        self.misses = 0


if __name__ == "__main__":
    cache = SemanticCache(similarity_threshold=0.92)

    print("Semantic Cache Test\n" + "="*40)

    # simulate first query
    q1     = "How do I register a sole proprietorship?"
    emb1   = embed_text(q1)
    result1 = {"answer": "You can register by filing a DBA with your county...", "query_type": "factual"}

    cache.set(q1, emb1, result1)
    print(f"Cached: '{q1}'")

    # test similar query — should hit cache
    q2   = "What are the steps to register a sole proprietorship?"
    emb2 = embed_text(q2)
    hit  = cache.get(q2, emb2)
    print(f"\nQuery: '{q2}'")
    print(f"Cache hit: {hit is not None}")
    if hit:
        print(f"Answer: {hit['answer']}")

    # test different query — should miss
    q3   = "What is dollar cost averaging?"
    emb3 = embed_text(q3)
    miss = cache.get(q3, emb3)
    print(f"\nQuery: '{q3}'")
    print(f"Cache hit: {miss is not None}")

    print(f"\nCache stats: {cache.stats()}")