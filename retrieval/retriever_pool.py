# retrieval/retriever_pool.py

import os
from typing import List, Dict
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter
)
from rank_bm25 import BM25Okapi

load_dotenv()

QDRANT_HOST       = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT       = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_chunks")
VECTOR_DIM        = 768  # nomic-embed-text dimension


class RetrieverPool:

    def __init__(self, in_memory: bool = True):
        # Qdrant — vector store (dense retrieval)
        if in_memory:
            self.qdrant = QdrantClient(":memory:")
        else:
            self.qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        self._collection_ready = False

        # BM25 — sparse retrieval (keyword-based)
        self.bm25        = None
        self.bm25_chunks = []  # keeps original chunks for BM25 lookup

    def _ensure_collection(self):
        if not self._collection_ready:
            existing = [c.name for c in self.qdrant.get_collections().collections]
            if QDRANT_COLLECTION not in existing:
                self.qdrant.create_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=VectorParams(
                        size=VECTOR_DIM,
                        distance=Distance.COSINE,
                    ),
                )
            self._collection_ready = True

    def index(self, chunks: List[dict]):
        """Index a list of embedded chunk dicts into Qdrant + BM25."""
        self._ensure_collection()

        # --- Dense index (Qdrant) ---
        points = []
        for i, chunk in enumerate(chunks):
            points.append(PointStruct(
                id=i,
                vector=chunk["embedding"],
                payload={
                    "chunk_id":    chunk["chunk_id"],
                    "doc_id":      chunk["doc_id"],
                    "source":      chunk["source"],
                    "text":        chunk["text"],
                    "chunk_index": chunk["chunk_index"],
                },
            ))

        # upload in batches of 256
        batch_size = 256
        for i in range(0, len(points), batch_size):
            self.qdrant.upsert(
                collection_name=QDRANT_COLLECTION,
                points=points[i: i + batch_size],
            )
        print(f"  Indexed {len(points)} chunks into Qdrant")

        # --- Sparse index (BM25) ---
        self.bm25_chunks = chunks
        tokenized = [c["text"].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        print(f"  Indexed {len(chunks)} chunks into BM25")

    def dense_retrieve(self, query_embedding: List[float], top_k: int = 5) -> List[dict]:
        """Retrieve using dense vector similarity (Qdrant)."""
        results = self.qdrant.query_points(
    collection_name=QDRANT_COLLECTION,
    query=query_embedding,
    limit=top_k,
).points
        return [
            {**hit.payload, "score": hit.score, "retrieval_type": "dense"}
            for hit in results
        ]

    def sparse_retrieve(self, query: str, top_k: int = 5) -> List[dict]:
        """Retrieve using BM25 keyword matching."""
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built yet. Call index() first.")
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {**self.bm25_chunks[i], "score": float(scores[i]), "retrieval_type": "sparse"}
            for i in top_indices
        ]

    def hybrid_retrieve(self, query: str, query_embedding: List[float], top_k: int = 5) -> List[dict]:
        """Combine dense + sparse results, deduplicate by chunk_id."""
        dense  = self.dense_retrieve(query_embedding, top_k=top_k)
        sparse = self.sparse_retrieve(query, top_k=top_k)

        seen, merged = set(), []
        for result in dense + sparse:
            cid = result["chunk_id"]
            if cid not in seen:
                seen.add(cid)
                merged.append(result)

        return merged[:top_k]


if __name__ == "__main__":
    from evaluation.beir_loader import load_fiqa
    from ingestion.loader import load_from_beir
    from ingestion.chunker import chunk_corpus
    from ingestion.embedder import embed_chunks

    print("Loading data...")
    corpus, _, _ = load_fiqa()
    docs   = list(load_from_beir(corpus))[:50]  # small test
    chunks = chunk_corpus(docs, strategy="adaptive")

    print(f"Embedding {len(chunks)} chunks...")
    chunks = embed_chunks(chunks)

    print("Indexing...")
    pool = RetrieverPool(in_memory=True)
    pool.index(chunks)

    print("\nTesting retrieval...")
    from ingestion.embedder import embed_text

    query = "How does dollar cost averaging work?"
    qvec  = embed_text(query)

    print("\n--- Dense ---")
    for r in pool.dense_retrieve(qvec, top_k=3):
        print(f"  [{r['score']:.3f}] {r['text'][:80]}...")

    print("\n--- Sparse ---")
    for r in pool.sparse_retrieve(query, top_k=3):
        print(f"  [{r['score']:.3f}] {r['text'][:80]}...")

    print("\n--- Hybrid ---")
    for r in pool.hybrid_retrieve(query, qvec, top_k=3):
        print(f"  [{r['retrieval_type']}] {r['text'][:80]}...")