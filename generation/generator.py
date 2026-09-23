# generation/generator.py

import os
import requests
from typing import List
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3")

SYSTEM_PROMPT = """You are a helpful assistant answering questions based strictly on the provided context.

Rules:
- Answer ONLY from the context provided
- If the context doesn't contain enough information, say "I don't have enough information to answer this"
- Always cite which source(s) you used using [1], [2] etc.
- Be concise and precise
- Never make up information not present in the context
"""


def generate(query: str, context: str) -> dict:
    """
    Generate an answer given a query and assembled context string.
    Returns dict with: answer, query, context_used
    """
    user_message = f"""Context:
{context}

Question: {query}

Answer based strictly on the context above:"""

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model":  OLLAMA_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
        },
    )
    response.raise_for_status()
    answer = response.json()["message"]["content"].strip()

    return {
        "query":        query,
        "answer":       answer,
        "context_used": context,
    }


def generate_from_assembly(assembly: dict) -> dict:
    """
    Generate directly from a context_assembler output dict.
    """
    result = generate(assembly["query"], assembly["context"])
    result["chunks_used"] = assembly.get("chunks_used", [])
    result["num_chunks"]  = assembly.get("num_chunks", 0)
    return result


if __name__ == "__main__":
    from generation.context_assembler import assemble_context

    mock_chunks = [
        {
            "chunk_id": "3_0",
            "doc_id":   "3",
            "source":   "beir/fiqa",
            "text":     "Dollar cost averaging is an investment strategy where you invest a fixed amount regularly regardless of the asset price. This means you buy more shares when prices are low and fewer when prices are high.",
            "relevance_score": 0.85,
            "score": 0.85,
        },
        {
            "chunk_id": "7_0",
            "doc_id":   "7",
            "source":   "beir/fiqa",
            "text":     "By investing consistently over time, investors reduce the impact of market volatility on their overall portfolio. This strategy prevents emotional decision making and market timing mistakes.",
            "relevance_score": 0.75,
            "score": 0.75,
        },
        {
            "chunk_id": "9_0",
            "doc_id":   "9",
            "source":   "beir/fiqa",
            "text":     "A possible disadvantage of dollar cost averaging is that if the market consistently rises, investing a lump sum upfront would have yielded better returns.",
            "relevance_score": 0.65,
            "score": 0.65,
        },
    ]

    query    = "How does dollar cost averaging reduce investment risk?"
    assembly = assemble_context(mock_chunks, query)

    print(f"Query: {query}")
    print(f"Context chunks: {assembly['num_chunks']}")
    print("\nGenerating answer...\n")

    result = generate_from_assembly(assembly)

    print(f"Answer:\n{'-'*40}")
    print(result["answer"])