# ingestion/chunker.py

from typing import List
import re


def chunk_fixed(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """Split text into fixed-size character chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


def chunk_sentences(text: str, max_chunk_size: int = 512, overlap_sentences: int = 1) -> List[str]:
    """Split text into sentence-aware chunks."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        if current_len + len(sent) > max_chunk_size and current:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:]
            current_len = sum(len(s) for s in current)
        current.append(sent)
        current_len += len(sent)

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c.strip()]


def chunk_adaptive(text: str, max_chunk_size: int = 512) -> List[str]:
    """
    Adaptive chunking:
    - Short docs (< max_chunk_size): keep as single chunk
    - Medium docs: sentence-aware chunking
    - Long docs: sentence-aware with smaller chunks
    """
    length = len(text)

    if length <= max_chunk_size:
        return [text.strip()]
    elif length <= max_chunk_size * 4:
        return chunk_sentences(text, max_chunk_size=max_chunk_size)
    else:
        return chunk_sentences(text, max_chunk_size=max_chunk_size // 2, overlap_sentences=2)


def chunk_document(doc: dict, strategy: str = "adaptive") -> List[dict]:
    """
    Chunk a single document dict (from loader.py) into chunk dicts.
    Each chunk carries its parent doc metadata.
    """
    text = doc["text"]

    if strategy == "fixed":
        chunks = chunk_fixed(text)
    elif strategy == "sentences":
        chunks = chunk_sentences(text)
    elif strategy == "adaptive":
        chunks = chunk_adaptive(text)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return [
        {
            "chunk_id": f"{doc['doc_id']}_{i}",
            "doc_id":   doc["doc_id"],
            "source":   doc["source"],
            "text":     chunk,
            "chunk_index": i,
            "total_chunks": len(chunks),
        }
        for i, chunk in enumerate(chunks)
    ]


def chunk_corpus(docs: List[dict], strategy: str = "adaptive") -> List[dict]:
    """Chunk an entire corpus of documents."""
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, strategy=strategy))
    return all_chunks


if __name__ == "__main__":
    from evaluation.beir_loader import load_fiqa
    from ingestion.loader import load_from_beir

    corpus, _, _ = load_fiqa()
    docs = list(load_from_beir(corpus))

    # test on first 100 docs
    sample_chunks = chunk_corpus(docs[:100], strategy="adaptive")

    print(f"100 docs → {len(sample_chunks)} chunks")
    print(f"Avg chunks per doc: {len(sample_chunks)/100:.1f}")
    print(f"\nSample chunk:")
    print(sample_chunks[0])