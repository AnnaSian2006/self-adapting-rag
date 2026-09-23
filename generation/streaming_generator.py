# generation/streaming_generator.py

import os
import requests
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


def stream_generate(query: str, context: str):
    """
    Generator function that streams answer tokens one by one.
    Usage:
        for token in stream_generate(query, context):
            print(token, end="", flush=True)
    """
    user_message = f"""Context:
{context}

Question: {query}

Answer based strictly on the context above:"""

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model":  OLLAMA_MODEL,
            "stream": True,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
        },
        stream=True,
    )
    response.raise_for_status()

    full_answer = ""
    for line in response.iter_lines():
        if line:
            import json
            chunk = json.loads(line.decode("utf-8"))
            token = chunk.get("message", {}).get("content", "")
            if token:
                full_answer += token
                yield token
            if chunk.get("done", False):
                break

    return full_answer


def stream_generate_from_assembly(assembly: dict):
    """
    Stream directly from a context_assembler output dict.
    Yields tokens and returns full answer at the end.
    """
    full_answer = ""
    for token in stream_generate(assembly["query"], assembly["context"]):
        full_answer += token
        yield token, full_answer


if __name__ == "__main__":
    from generation.context_assembler import assemble_context

    mock_chunks = [
        {
            "chunk_id": "3_0",
            "doc_id":   "3",
            "source":   "beir/fiqa",
            "text":     "A sole proprietorship reports business income directly on the owner's personal tax return using Schedule C. Self-employment tax of 15.3% applies to net earnings from self-employment.",
            "relevance_score": 0.85,
            "score": 0.85,
        },
        {
            "chunk_id": "7_0",
            "doc_id":   "7",
            "source":   "beir/fiqa",
            "text":     "As a sole proprietor you pay income tax on all business profits. You can deduct business expenses to reduce your taxable income. Estimated quarterly taxes must be paid to avoid penalties.",
            "relevance_score": 0.80,
            "score": 0.80,
        },
    ]

    query    = "How do taxes work for a sole proprietorship?"
    assembly = assemble_context(mock_chunks, query)

    print(f"Query: {query}")
    print(f"\nStreaming answer:\n{'-'*40}")

    full = ""
    for token in stream_generate(query, assembly["context"]):
        print(token, end="", flush=True)
        full += token

    print(f"\n{'-'*40}")
    print(f"Total tokens: {len(full.split())}")