# pipeline.py
from retrieval.semantic_cache import SemanticCache
from retrieval.graph_retriever import GraphRetriever
from retrieval.strategy_router import route
from retrieval.retriever_pool import RetrieverPool
from retrieval.relevance_critic import score_chunks
from retrieval.query_rewriter import rewrite_if_needed
from generation.context_assembler import assemble_context
from generation.generator import generate_from_assembly
from generation.faithfulness_verifier import verify
from adaptation.feedback_logger import FeedbackLogger
from ingestion.embedder import embed_text


class RAGPipeline:
    def __init__(self, retriever_pool: RetrieverPool, graph_retriever=None):
        self.pool            = retriever_pool
        self.logger          = FeedbackLogger()
        self.graph_retriever = graph_retriever
        self.cache           = SemanticCache(similarity_threshold=0.92)

    def run(self, query: str, user_feedback: int = None) -> dict:
        """
        Full end-to-end pipeline:
        1. Classify query + route to strategy
        2. Retrieve chunks
        3. Score chunks with relevance critic
        4. Rewrite + retry if quality is poor
        5. Assemble context
        6. Generate answer
        7. Verify faithfulness
        8. Log everything
        """
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        
                # step 0 — check semantic cache
        query_embedding_early = embed_text(query)
        cached = self.cache.get(query, query_embedding_early)
        if cached:
            print(f"[0] Cache hit — returning instantly")
            return cached

        # step 1 — classify + route
        routing = route(query)
        print(f"[1] Type: {routing['type']} → Strategy: {routing['strategy']} (top_k={routing['top_k']})")

                       # step 2 — embed query + retrieve
        # use HyDE for reasoning/comparative queries
        from retrieval.hyde import hyde_embed
        if routing["type"] in ("reasoning", "comparative"):
            print(f"[2a] HyDE — generating hypothetical document...")
            hyp_doc, query_embedding = hyde_embed(query)
            print(f"[2a] HyDE doc: {hyp_doc[:80]}...")
        else:
            query_embedding = query_embedding_early
        strategy        = routing["strategy"]
        top_k           = routing["top_k"]

        if strategy == "dense":
            chunks = self.pool.dense_retrieve(query_embedding, top_k=top_k)
        elif strategy == "sparse":
            chunks = self.pool.sparse_retrieve(query, top_k=top_k)
        else:
            chunks = self.pool.hybrid_retrieve(query, query_embedding, top_k=top_k)

        print(f"[2] Retrieved {len(chunks)} chunks")

                # step 3 — graph enrichment (multi-document reasoning)
        if hasattr(self, 'graph_retriever') and self.graph_retriever is not None:
            chunks = self.graph_retriever.enrich(chunks, hops=2, max_total=8)
            print(f"[3a] Graph enriched: {len(chunks)} chunks (multi-hop)")

        # step 3b — score with relevance critic
        scored = score_chunks(query, chunks, threshold=0.4)
        print(f"[3b] Passed relevance critic: {len(scored)}/{len(chunks)}")
        
        # step 4 — rewrite + retry if needed
        rewrite_result  = rewrite_if_needed(query, scored, threshold=0.5)
        was_rewritten   = rewrite_result["should_retry"]
        rewritten_query = rewrite_result["rewritten"]

        if was_rewritten:
            print(f"[4] Rewriting query: '{rewritten_query}'")
            new_embedding = embed_text(rewritten_query)
            chunks2       = self.pool.hybrid_retrieve(rewritten_query, new_embedding, top_k=top_k)
            scored2       = score_chunks(rewritten_query, chunks2, threshold=0.4)
            if len(scored2) > len(scored):
                scored = scored2
                print(f"    Retry got {len(scored)} chunks — using rewritten results")
            else:
                print(f"    Retry didn't improve — keeping original results")
        else:
            print(f"[4] No rewrite needed")

        # step 5 — assemble context
        assembly = assemble_context(scored, query)
        print(f"[5] Context assembled: {assembly['num_chunks']} chunks, {assembly['total_chars']} chars")

        # step 6 — generate answer
        print(f"[6] Generating answer...")
        generation = generate_from_assembly(assembly)

        # step 7 — verify faithfulness
        faith = verify(query, generation["answer"], assembly["context"])
        print(f"[7] Faithfulness: {faith['score']} — {faith['reason']}")

        # print the actual answer
        print(f"\n--- Answer ---")
        print(generation["answer"])
        print(f"--------------\n")

        # step 8 — log
        avg_score = sum(c["relevance_score"] for c in scored) / max(len(scored), 1)
        top_score = max((c["relevance_score"] for c in scored), default=0)

        log_id = self.logger.log(
            query                = query,
            query_type           = routing["type"],
            strategy_used        = strategy,
            top_k                = top_k,
            num_chunks_retrieved = len(chunks),
            num_chunks_passed    = len(scored),
            avg_relevance_score  = avg_score,
            top_relevance_score  = top_score,
            faithfulness_score   = faith["score"],
            was_rewritten        = was_rewritten,
            rewritten_query      = rewritten_query if was_rewritten else None,
            answer               = generation["answer"],
            user_feedback        = user_feedback,
        )
        print(f"[8] Logged with ID: {log_id}")

        result = {
            "query":           query,
            "answer":          generation["answer"],
            "query_type":      routing["type"],
            "strategy":        strategy,
            "chunks_used":     assembly["chunks_used"],
            "num_chunks":      assembly["num_chunks"],
            "faithfulness":    faith["score"],
            "was_rewritten":   was_rewritten,
            "rewritten_query": rewritten_query if was_rewritten else None,
            "log_id":          log_id,
        }
        self.cache.set(query, query_embedding, result)
        return result
    def close(self):
        self.logger.close()


if __name__ == "__main__":
    from evaluation.beir_loader import load_fiqa
    from ingestion.loader import load_from_beir
    from ingestion.chunker import chunk_corpus
    from ingestion.embedder import embed_chunks

    print("Setting up pipeline...")
    corpus, _, _ = load_fiqa()
    docs   = list(load_from_beir(corpus))[:100]
    chunks = chunk_corpus(docs, strategy="adaptive")
    chunks = embed_chunks(chunks)

    pool = RetrieverPool(in_memory=True)
    pool.index(chunks)

    # build graph
    graph = GraphRetriever()
    graph.build(chunks)

    pipeline = RAGPipeline(pool, graph_retriever=graph)

    test_queries = [
        "How do taxes work for a sole proprietorship?",
        "What is the difference between LLC and sole proprietorship?",
        "How do I register a DBA?",
    ]

    for query in test_queries:
        pipeline.run(query)

    pipeline.close()
