# evaluation/beir_loader.py

from datasets import load_dataset

def load_fiqa():
    corpus  = load_dataset("BeIR/fiqa", "corpus", split="corpus")
    queries = load_dataset("BeIR/fiqa", "queries", split="queries")
    qrels   = load_dataset("BeIR/fiqa-qrels", split="test")
    return corpus, queries, qrels

def load_nfcorpus():
    corpus  = load_dataset("BeIR/nfcorpus", "corpus", split="corpus")
    queries = load_dataset("BeIR/nfcorpus", "queries", split="queries")
    qrels   = load_dataset("BeIR/nfcorpus-qrels", split="test")
    return corpus, queries, qrels

if __name__ == "__main__":
    print("Loading FiQA...")
    corpus, queries, qrels = load_fiqa()
    print(f"Corpus: {len(corpus)} docs")
    print(f"Queries: {len(queries)} questions")
    print(f"Qrels: {len(qrels)} relevance pairs")
    print("Sample doc:", corpus[0])