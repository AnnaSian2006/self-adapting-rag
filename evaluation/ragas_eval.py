# evaluation/ragas_eval.py

import os
import json
import numpy as np
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_EMBED    = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def safe_score(val):
    if isinstance(val, list):
        cleaned = [v for v in val if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return round(float(np.mean(cleaned)), 4) if cleaned else 0.0
    return round(float(val), 4) if val is not None else 0.0


def build_ragas_dataset(pipeline, queries, qrels, corpus_lookup: dict) -> Dataset:
    # build qrels lookup
    qrels_map = {}
    for row in qrels:
        qid = str(row["query-id"])
        did = str(row["corpus-id"])
        if qid not in qrels_map:
            qrels_map[qid] = []
        qrels_map[qid].append(did)

    records = []
    total   = min(len(queries), 20)
    print(f"Running pipeline on {total} queries...\n")

    for i, query_row in enumerate(queries):
        if i >= total:
            break

        qid   = str(query_row["_id"])
        query = query_row["text"]
        print(f"  [{i+1}/{total}] {query[:60]}...")

        try:
            result   = pipeline.run(query)
            answer   = result["answer"]
            contexts = [c["text"] for c in result.get("chunks_used", [])]
            if not contexts:
                contexts = ["no context retrieved"]

            relevant_ids = qrels_map.get(qid, [])
            ground_truth = " ".join([
                corpus_lookup.get(did, "")
                for did in relevant_ids[:2]
            ]).strip() or "No ground truth available"

            records.append({
                "question":     query,
                "answer":       answer,
                "contexts":     contexts,
                "ground_truth": ground_truth,
            })
            print(f"    ✓ added ({len(contexts)} contexts)")

        except Exception as e:
            print(f"    ✗ skipped: {e}")
            continue

    print(f"\nBuilt {len(records)} records")
    return Dataset.from_list(records)


def run_evaluation(pipeline, queries, qrels, corpus_lookup: dict) -> dict:
    llm        = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED, base_url=OLLAMA_BASE_URL)

    dataset = build_ragas_dataset(pipeline, queries, qrels, corpus_lookup)

    if len(dataset) == 0:
        print("No valid records — cannot evaluate")
        return {}

    # only run context_recall via RAGAS (doesn't need LLM)
    print(f"\nRunning RAGAS context_recall on {len(dataset)} queries...")
    results = evaluate(
        dataset,
        metrics=[context_recall],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
    )

    # our own scores from pipeline logs
    from adaptation.feedback_logger import FeedbackLogger
    from sqlalchemy import func
    logger  = FeedbackLogger()
    session = logger.session
    from adaptation.feedback_logger import QueryLog

    avg_faith = session.query(func.avg(QueryLog.faithfulness_score)).scalar() or 0.0
    avg_relev = session.query(func.avg(QueryLog.avg_relevance_score)).scalar() or 0.0
    avg_topk  = session.query(func.avg(QueryLog.top_relevance_score)).scalar() or 0.0
    logger.close()

    scores = {
        "faithfulness":      round(float(avg_faith), 4),
        "context_relevance": round(float(avg_relev), 4),
        "top_chunk_score":   round(float(avg_topk), 4),
        "context_recall":    safe_score(results["context_recall"]),
    }

    return scores


if __name__ == "__main__":
    from evaluation.beir_loader import load_fiqa
    from ingestion.loader import load_from_beir
    from ingestion.chunker import chunk_corpus
    from ingestion.embedder import embed_chunks
    from retrieval.retriever_pool import RetrieverPool
    from pipeline import RAGPipeline

    print("Loading FiQA...")
    corpus, queries, qrels = load_fiqa()

    print("Building corpus lookup...")
    corpus_lookup = {str(doc["_id"]): doc["text"] for doc in corpus}

    print("Indexing 200 docs...")
    docs   = list(load_from_beir(corpus))[:200]
    chunks = chunk_corpus(docs, strategy="adaptive")
    chunks = embed_chunks(chunks)

    pool = RetrieverPool(in_memory=True)
    pool.index(chunks)

    rag = RAGPipeline(pool)

    scores = run_evaluation(
        pipeline      = rag,
        queries       = list(queries),
        qrels         = list(qrels),
        corpus_lookup = corpus_lookup,
    )

    if scores:
        print("\n" + "="*40)
        print("RAGAS Evaluation Results")
        print("="*40)
        for metric, score in scores.items():
            print(f"  {metric:25s}: {score}")

        os.makedirs("evaluation", exist_ok=True)
        with open("evaluation/results.json", "w") as f:
            json.dump(scores, f, indent=2)
        print("\nSaved to evaluation/results.json")

    rag.close()