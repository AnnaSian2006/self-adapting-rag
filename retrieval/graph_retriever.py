# retrieval/graph_retriever.py

import re
from typing import List, Dict, Set
from collections import defaultdict


def extract_entities(text: str) -> Set[str]:
    """
    Extract key entities from text for graph linking.
    Simple approach: extract capitalized phrases, numbers,
    financial terms and meaningful nouns.
    """
    entities = set()

    # capitalized words/phrases (proper nouns)
    caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    entities.update([c.lower() for c in caps if len(c) > 3])

    # financial/domain keywords
    domain_terms = [
        "tax", "income", "investment", "business", "expense",
        "deduction", "profit", "loss", "capital", "interest",
        "dividend", "equity", "debt", "asset", "liability",
        "revenue", "cash", "fund", "stock", "bond", "ira",
        "llc", "sole proprietor", "dba", "credit", "loan",
        "mortgage", "insurance", "salary", "wage", "budget",
    ]
    text_lower = text.lower()
    for term in domain_terms:
        if term in text_lower:
            entities.add(term)

    # dollar amounts and percentages
    amounts = re.findall(r'\$[\d,]+|\d+%|\d+\.\d+%', text)
    entities.update(amounts)

    return entities


class ChunkGraph:
    """
    Graph where nodes are chunks and edges connect:
    1. Chunks from the same document (same doc_id)
    2. Chunks that share key entities
    """

    def __init__(self):
        self.nodes: Dict[str, dict] = {}        # chunk_id → chunk
        self.edges: Dict[str, Set[str]] = defaultdict(set)  # chunk_id → set of neighbor chunk_ids
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)  # entity → set of chunk_ids

    def add_chunk(self, chunk: dict):
        """Add a chunk as a node in the graph."""
        cid = chunk["chunk_id"]
        self.nodes[cid] = chunk

        # index by entities
        entities = extract_entities(chunk["text"])
        for entity in entities:
            self.entity_index[entity].add(cid)

    def build_edges(self):
        """
        Build edges after all chunks are added.
        Two types of edges:
        1. Same document — chunks from same doc_id are linked
        2. Shared entity — chunks sharing a key entity are linked
        """
        # group by doc_id
        doc_groups: Dict[str, List[str]] = defaultdict(list)
        for cid, chunk in self.nodes.items():
            doc_groups[chunk["doc_id"]].append(cid)

        # edge type 1 — same document
        for doc_id, chunk_ids in doc_groups.items():
            for i, cid1 in enumerate(chunk_ids):
                for cid2 in chunk_ids[i+1:]:
                    self.edges[cid1].add(cid2)
                    self.edges[cid2].add(cid1)

        # edge type 2 — shared entities
        for entity, chunk_ids in self.entity_index.items():
            chunk_list = list(chunk_ids)
            for i, cid1 in enumerate(chunk_list):
                for cid2 in chunk_list[i+1:]:
                    self.edges[cid1].add(cid2)
                    self.edges[cid2].add(cid1)

        print(f"  Graph built: {len(self.nodes)} nodes, {sum(len(v) for v in self.edges.values())//2} edges")

    def get_neighbors(self, chunk_id: str, max_neighbors: int = 3) -> List[dict]:
        """Get neighboring chunks for a given chunk_id."""
        neighbors = []
        for neighbor_id in list(self.edges.get(chunk_id, set()))[:max_neighbors]:
            if neighbor_id in self.nodes:
                neighbors.append(self.nodes[neighbor_id])
        return neighbors

    def multi_hop_retrieve(
        self,
        seed_chunks: List[dict],
        hops: int = 2,
        max_neighbors_per_node: int = 2,
        max_total: int = 8,
    ) -> List[dict]:
        """
        Starting from seed chunks, traverse the graph
        for `hops` steps to find related chunks.

        Returns seed chunks + discovered neighbors,
        deduplicated and ranked by hop distance.
        """
        visited:  Dict[str, int]  = {}   # chunk_id → hop distance
        queue:    List[tuple]     = []   # (chunk_id, hop)
        results:  List[dict]      = []

        # initialize with seed chunks at hop 0
        for chunk in seed_chunks:
            cid = chunk["chunk_id"]
            if cid in self.nodes:
                visited[cid] = 0
                queue.append((cid, 0))
                results.append({**chunk, "hop": 0})

        # BFS traversal
        idx = 0
        while idx < len(queue) and len(results) < max_total:
            cid, hop = queue[idx]
            idx += 1

            if hop >= hops:
                continue

            neighbors = self.get_neighbors(cid, max_neighbors=max_neighbors_per_node)
            for neighbor in neighbors:
                nid = neighbor["chunk_id"]
                if nid not in visited:
                    visited[nid] = hop + 1
                    queue.append((nid, hop + 1))
                    results.append({**neighbor, "hop": hop + 1})

                    if len(results) >= max_total:
                        break

        # sort: seed chunks first (hop=0), then by hop distance
        results.sort(key=lambda x: x.get("hop", 0))
        return results[:max_total]


class GraphRetriever:
    """
    Wraps the ChunkGraph and integrates with existing RetrieverPool.
    Usage:
        graph = GraphRetriever()
        graph.build(all_chunks)
        enriched = graph.enrich(seed_chunks, hops=2)
    """

    def __init__(self):
        self.graph = ChunkGraph()
        self.built = False

    def build(self, chunks: List[dict]):
        """Build graph from all indexed chunks."""
        print(f"Building chunk graph from {len(chunks)} chunks...")
        for chunk in chunks:
            self.graph.add_chunk(chunk)
        self.graph.build_edges()
        self.built = True

    def enrich(
        self,
        seed_chunks: List[dict],
        hops: int = 2,
        max_total: int = 8,
    ) -> List[dict]:
        """
        Enrich seed chunks with graph neighbors.
        Returns combined list of seed + related chunks.
        """
        if not self.built:
            raise RuntimeError("Graph not built yet. Call build() first.")

        enriched = self.graph.multi_hop_retrieve(
            seed_chunks,
            hops=hops,
            max_neighbors_per_node=2,
            max_total=max_total,
        )

        # tag each chunk with how it was found
        for chunk in enriched:
            if chunk.get("hop", 0) == 0:
                chunk["retrieval_source"] = "direct"
            else:
                chunk["retrieval_source"] = f"graph_hop_{chunk['hop']}"

        return enriched


if __name__ == "__main__":
    from evaluation.beir_loader import load_fiqa
    from ingestion.loader import load_from_beir
    from ingestion.chunker import chunk_corpus
    from ingestion.embedder import embed_chunks
    from retrieval.retriever_pool import RetrieverPool
    from ingestion.embedder import embed_text

    print("Loading data...")
    corpus, _, _ = load_fiqa()
    docs   = list(load_from_beir(corpus))[:100]
    chunks = chunk_corpus(docs, strategy="adaptive")
    chunks = embed_chunks(chunks)

    # build retriever pool
    pool = RetrieverPool(in_memory=True)
    pool.index(chunks)

    # build graph
    graph_retriever = GraphRetriever()
    graph_retriever.build(chunks)

    # test
    query = "How do taxes work for a sole proprietorship?"
    qvec  = embed_text(query)

    # step 1 — normal retrieval
    seed_chunks = pool.hybrid_retrieve(query, qvec, top_k=3)
    print(f"\nSeed chunks from retrieval: {len(seed_chunks)}")
    for c in seed_chunks:
        print(f"  [{c.get('score', 0):.3f}] {c['text'][:80]}...")

    # step 2 — graph enrichment
    enriched = graph_retriever.enrich(seed_chunks, hops=2, max_total=8)
    print(f"\nAfter graph enrichment: {len(enriched)} chunks")
    for c in enriched:
        print(f"  [hop={c.get('hop',0)}] [{c['retrieval_source']}] {c['text'][:80]}...")