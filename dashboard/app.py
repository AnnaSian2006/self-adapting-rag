# dashboard/app.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import json
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from adaptation.feedback_logger import QueryLog

load_dotenv()

DB_URL  = os.getenv("SQLITE_URL", "sqlite:///ragdb.sqlite")
engine  = create_engine(DB_URL, echo=False)

st.set_page_config(
    page_title = "Self-Adapting RAG Dashboard",
    page_icon  = "🧠",
    layout     = "wide",
)

st.title("🧠 Self-Adapting RAG — Observability Dashboard")
st.markdown("Real-time monitoring of retrieval quality, strategy distribution, and system adaptation.")

# load logs
@st.cache_data(ttl=30)
def load_logs():
    with engine.connect() as conn:
        df = pd.read_sql("SELECT * FROM query_logs ORDER BY timestamp DESC", conn)
    return df

df = load_logs()

if df.empty:
    st.warning("No query logs yet. Run the pipeline first.")
    st.stop()

# ── top metrics ──────────────────────────────────────────────
st.subheader("📊 Overall Metrics")
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Queries",       len(df))
col2.metric("Avg Faithfulness",    f"{df['faithfulness_score'].mean():.2f}")
col3.metric("Avg Relevance",       f"{df['avg_relevance_score'].mean():.2f}")
col4.metric("Queries Rewritten",   f"{df['was_rewritten'].sum()}")
col5.metric("Positive Feedback",   f"{int(df['user_feedback'].sum()) if df['user_feedback'].notna().any() else 'N/A'}")

st.divider()

# ── strategy distribution ─────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("🎯 Strategy Distribution")
    strategy_counts = df["strategy_used"].value_counts().reset_index()
    strategy_counts.columns = ["Strategy", "Count"]
    st.bar_chart(strategy_counts.set_index("Strategy"))

with col_b:
    st.subheader("🔍 Query Type Distribution")
    type_counts = df["query_type"].value_counts().reset_index()
    type_counts.columns = ["Query Type", "Count"]
    st.bar_chart(type_counts.set_index("Query Type"))

st.divider()

# ── performance over time ─────────────────────────────────────
st.subheader("📈 Faithfulness & Relevance Over Time")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df_sorted = df.sort_values("timestamp")

chart_df = df_sorted[["timestamp", "faithfulness_score", "avg_relevance_score"]].set_index("timestamp")
st.line_chart(chart_df)

st.divider()

# ── strategy performance table ────────────────────────────────
st.subheader("⚡ Strategy Performance Breakdown")
strategy_perf = df.groupby(["query_type", "strategy_used"]).agg(
    count              = ("id", "count"),
    avg_faithfulness   = ("faithfulness_score", "mean"),
    avg_relevance      = ("avg_relevance_score", "mean"),
    avg_top_score      = ("top_relevance_score", "mean"),
    rewrite_rate       = ("was_rewritten", "mean"),
).reset_index().round(3)
st.dataframe(strategy_perf, use_container_width=True)

st.divider()

# ── ragas results ─────────────────────────────────────────────
st.subheader("🧪 RAGAS Evaluation Results")
ragas_path = "evaluation/results.json"
if os.path.exists(ragas_path):
    with open(ragas_path) as f:
        ragas = json.load(f)
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Faithfulness",      ragas.get("faithfulness", "N/A"))
    r2.metric("Context Relevance", ragas.get("context_relevance", "N/A"))
    r3.metric("Top Chunk Score",   ragas.get("top_chunk_score", "N/A"))
    r4.metric("Context Recall",    ragas.get("context_recall", "N/A"))
else:
    st.info("Run evaluation/ragas_eval.py to see RAGAS scores here.")

st.divider()

# ── recent query log ──────────────────────────────────────────
st.subheader("📋 Recent Query Log")
display_cols = [
    "timestamp", "query", "query_type", "strategy_used",
    "num_chunks_passed", "avg_relevance_score",
    "faithfulness_score", "was_rewritten"
]
st.dataframe(
    df[display_cols].head(50),
    use_container_width=True,
)

st.divider()

# ── live query tester ─────────────────────────────────────────
st.subheader("🚀 Live Query Tester")
st.markdown("Test the pipeline directly from the dashboard.")

user_query = st.text_input("Enter your query:", placeholder="e.g. How do I register a sole proprietorship?")

if st.button("Run Query") and user_query:
    with st.spinner("Setting up pipeline..."):
        from evaluation.beir_loader import load_fiqa
        from ingestion.loader import load_from_beir
        from ingestion.chunker import chunk_corpus
        from ingestion.embedder import embed_chunks, embed_text
        from retrieval.retriever_pool import RetrieverPool
        from retrieval.graph_retriever import GraphRetriever
        from retrieval.strategy_router import route
        from retrieval.relevance_critic import score_chunks
        from generation.context_assembler import assemble_context
        from generation.streaming_generator import stream_generate
        from generation.faithfulness_verifier import verify

        corpus, _, _ = load_fiqa()
        docs   = list(load_from_beir(corpus))[:500]
        chunks = chunk_corpus(docs, strategy="adaptive")
        chunks = embed_chunks(chunks)

        pool = RetrieverPool(in_memory=True)
        pool.index(chunks)

        graph = GraphRetriever()
        graph.build(chunks)

    # routing
    routing = route(user_query)
    st.markdown(f"**Query type:** `{routing['type']}` | **Strategy:** `{routing['strategy']}`")

    # hyde for reasoning/comparative
    from retrieval.hyde import hyde_embed
    if routing["type"] in ("reasoning", "comparative"):
        st.info("Using HyDE — generating hypothetical document for better retrieval...")
        _, qvec = hyde_embed(user_query)
    else:
        qvec = embed_text(user_query)

    # retrieve + graph enrich
    strategy = routing["strategy"]
    top_k    = routing["top_k"]
    if strategy == "dense":
        seed = pool.dense_retrieve(qvec, top_k=top_k)
    elif strategy == "sparse":
        seed = pool.sparse_retrieve(user_query, top_k=top_k)
    else:
        seed = pool.hybrid_retrieve(user_query, qvec, top_k=top_k)

    enriched = graph.enrich(seed, hops=2, max_total=8)
    scored   = score_chunks(user_query, enriched, threshold=0.4)
    assembly = assemble_context(scored, user_query)

    st.markdown(f"**Chunks used:** `{assembly['num_chunks']}` | **Context chars:** `{assembly['total_chars']}`")

    # stream answer
    st.markdown("**Answer:**")
    answer_box = st.empty()
    full_answer = ""

    with st.spinner("Generating..."):
        for token in stream_generate(user_query, assembly["context"]):
            full_answer += token
            answer_box.markdown(full_answer + "▌")

    answer_box.markdown(full_answer)

    # faithfulness
    faith = verify(user_query, full_answer, assembly["context"])
    st.markdown(f"**Faithfulness:** `{faith['score']}` — {faith['reason']}")