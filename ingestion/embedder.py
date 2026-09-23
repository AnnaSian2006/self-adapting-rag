# ingestion/embedder.py

import os
import requests
from typing import List
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def embed_text(text: str) -> List[float]:
    """Embed a single string using Ollama."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
    )
    response.raise_for_status()
    return response.json()["embedding"]


def embed_batch(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """Embed a list of texts in batches."""
    embeddings = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i: i + batch_size]
        for j, text in enumerate(batch):
            emb = embed_text(text)
            embeddings.append(emb)
        print(f"  Embedded {min(i + batch_size, total)}/{total}", end="\r")
    print()
    return embeddings


def embed_chunks(chunks: List[dict], batch_size: int = 32) -> List[dict]:
    """Add embedding vectors to chunk dicts in-place."""
    texts = [c["text"] for c in chunks]
    vectors = embed_batch(texts, batch_size=batch_size)
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks


if __name__ == "__main__":
    test_texts = [
        "What is dollar cost averaging?",
        "How do index funds work?",
        "What is compound interest?",
    ]
    print(f"Embedding model: {OLLAMA_EMBED_MODEL}")
    print(f"Ollama URL: {OLLAMA_BASE_URL}")
    print("Testing embedder...")

    embeddings = embed_batch(test_texts)
    print(f"Got {len(embeddings)} embeddings")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print("Embedder working correctly!")