# generation/context_assembler.py

from typing import List


def deduplicate(chunks: List[dict], similarity_threshold: float = 0.85) -> List[dict]:
    """
    Remove near-duplicate chunks by comparing text overlap.
    Simple approach: if two chunks share >85% of words, keep the higher scored one.
    """
    seen_words = []
    unique     = []

    for chunk in chunks:
        words = set(chunk["text"].lower().split())
        is_duplicate = False
        for seen in seen_words:
            overlap = len(words & seen) / max(len(words | seen), 1)
            if overlap >= similarity_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(chunk)
            seen_words.append(words)

    return unique


def trim_to_budget(chunks: List[dict], max_chars: int = 3000) -> List[dict]:
    """
    Trim chunk list so total text fits within character budget.
    Keeps highest scored chunks first.
    """
    total   = 0
    trimmed = []
    for chunk in chunks:
        length = len(chunk["text"])
        if total + length <= max_chars:
            trimmed.append(chunk)
            total += length
        else:
            break
    return trimmed


def assemble_context(chunks: List[dict], query: str, max_chars: int = 3000) -> dict:
    """
    Full assembly pipeline:
    1. Sort by relevance score
    2. Deduplicate
    3. Trim to token budget
    4. Format into a clean context string with source attribution
    """
    # sort by relevance score if available, else retrieval score
    chunks = sorted(
        chunks,
        key=lambda c: c.get("relevance_score", c.get("score", 0)),
        reverse=True,
    )

    # deduplicate
    chunks = deduplicate(chunks)

    # trim to budget
    chunks = trim_to_budget(chunks, max_chars=max_chars)

    # format context string
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "unknown")
        context_parts.append(f"[{i}] (source: {source})\n{chunk['text']}")

    context_str = "\n\n".join(context_parts)

    return {
        "query":        query,
        "context":      context_str,
        "chunks_used":  chunks,
        "num_chunks":   len(chunks),
        "total_chars":  len(context_str),
    }


if __name__ == "__main__":
    # simulate scored chunks coming from relevance critic
    mock_chunks = [
        {
            "chunk_id": "3_0",
            "doc_id":   "3",
            "source":   "beir/fiqa",
            "text":     "Dollar cost averaging is an investment strategy where you invest a fixed amount regularly regardless of price.",
            "relevance_score": 0.85,
            "score": 0.85,
        },
        {
            "chunk_id": "5_0",
            "doc_id":   "5",
            "source":   "beir/fiqa",
            "text":     "Dollar cost averaging is an investment strategy where you invest a fixed amount regularly regardless of price.",
            "relevance_score": 0.80,
            "score": 0.80,
        },  # near duplicate of above
        {
            "chunk_id": "7_0",
            "doc_id":   "7",
            "source":   "beir/fiqa",
            "text":     "By investing consistently over time, investors reduce the impact of volatility on their portfolio.",
            "relevance_score": 0.75,
            "score": 0.75,
        },
        {
            "chunk_id": "9_0",
            "doc_id":   "9",
            "source":   "beir/fiqa",
            "text":     "This strategy removes the emotional aspect of investing and prevents trying to time the market.",
            "relevance_score": 0.70,
            "score": 0.70,
        },
    ]

    query  = "How does dollar cost averaging reduce investment risk?"
    result = assemble_context(mock_chunks, query)

    print(f"Query: {query}")
    print(f"Chunks used: {result['num_chunks']} (1 duplicate removed)")
    print(f"Total chars: {result['total_chars']}")
    print(f"\nAssembled context:\n{'-'*40}")
    print(result["context"])