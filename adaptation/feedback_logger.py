# adaptation/feedback_logger.py

import os
import json
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine, Column, String, Float,
    Integer, DateTime, Text, Boolean
)
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

# use SQLite locally (no Docker needed)
# swap to PostgreSQL later via POSTGRES_URL in .env
DB_URL = os.getenv("SQLITE_URL", "sqlite:///ragdb.sqlite")

engine       = create_engine(DB_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base         = declarative_base()


class QueryLog(Base):
    """Logs every query and its retrieval + generation outcome."""
    __tablename__ = "query_logs"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    timestamp         = Column(DateTime, default=datetime.utcnow)
    query             = Column(Text, nullable=False)
    query_type        = Column(String(32))   # factual/reasoning/comparative/ambiguous
    strategy_used     = Column(String(32))   # dense/sparse/hybrid
    top_k             = Column(Integer)
    num_chunks_retrieved = Column(Integer)
    num_chunks_passed    = Column(Integer)   # after relevance critic
    avg_relevance_score  = Column(Float)
    top_relevance_score  = Column(Float)
    faithfulness_score   = Column(Float)
    was_rewritten        = Column(Boolean, default=False)
    rewritten_query      = Column(Text, nullable=True)
    answer               = Column(Text, nullable=True)
    user_feedback        = Column(Integer, nullable=True)  # 1=good, 0=bad, None=no feedback


class FeedbackLogger:

    def __init__(self):
        Base.metadata.create_all(engine)
        self.session = SessionLocal()

    def log(
        self,
        query:                str,
        query_type:           str,
        strategy_used:        str,
        top_k:                int,
        num_chunks_retrieved: int,
        num_chunks_passed:    int,
        avg_relevance_score:  float,
        top_relevance_score:  float,
        faithfulness_score:   float,
        was_rewritten:        bool          = False,
        rewritten_query:      Optional[str] = None,
        answer:               Optional[str] = None,
        user_feedback:        Optional[int] = None,
    ) -> int:
        """Log a query interaction. Returns the log entry ID."""
        entry = QueryLog(
            query                = query,
            query_type           = query_type,
            strategy_used        = strategy_used,
            top_k                = top_k,
            num_chunks_retrieved = num_chunks_retrieved,
            num_chunks_passed    = num_chunks_passed,
            avg_relevance_score  = avg_relevance_score,
            top_relevance_score  = top_relevance_score,
            faithfulness_score   = faithfulness_score,
            was_rewritten        = was_rewritten,
            rewritten_query      = rewritten_query,
            answer               = answer,
            user_feedback        = user_feedback,
        )
        self.session.add(entry)
        self.session.commit()
        self.session.refresh(entry)
        return entry.id

    def add_feedback(self, log_id: int, feedback: int):
        """Add user feedback (1=good, 0=bad) to an existing log entry."""
        entry = self.session.get(QueryLog, log_id)
        if entry:
            entry.user_feedback = feedback
            self.session.commit()

    def get_strategy_stats(self) -> dict:
        """
        Aggregate stats per strategy — used by adaptation engine
        to decide which strategy is performing best.
        """
        from sqlalchemy import func
        rows = (
            self.session.query(
                QueryLog.strategy_used,
                QueryLog.query_type,
                func.count(QueryLog.id).label("count"),
                func.avg(QueryLog.avg_relevance_score).label("avg_relevance"),
                func.avg(QueryLog.faithfulness_score).label("avg_faithfulness"),
                func.avg(QueryLog.user_feedback).label("avg_feedback"),
            )
            .group_by(QueryLog.strategy_used, QueryLog.query_type)
            .all()
        )

        stats = {}
        for row in rows:
            key = f"{row.query_type}:{row.strategy_used}"
            stats[key] = {
                "count":            row.count,
                "avg_relevance":    round(row.avg_relevance or 0, 3),
                "avg_faithfulness": round(row.avg_faithfulness or 0, 3),
                "avg_feedback":     round(row.avg_feedback or 0, 3),
            }
        return stats

    def close(self):
        self.session.close()


if __name__ == "__main__":
    logger = FeedbackLogger()

    print("Logging test queries...\n")

    id1 = logger.log(
        query                = "What is dollar cost averaging?",
        query_type           = "factual",
        strategy_used        = "dense",
        top_k                = 3,
        num_chunks_retrieved = 3,
        num_chunks_passed    = 3,
        avg_relevance_score  = 0.82,
        top_relevance_score  = 0.91,
        faithfulness_score   = 0.88,
        answer               = "Dollar cost averaging is...",
        user_feedback        = 1,
    )

    id2 = logger.log(
        query                = "how invest",
        query_type           = "ambiguous",
        strategy_used        = "sparse",
        top_k                = 3,
        num_chunks_retrieved = 3,
        num_chunks_passed    = 1,
        avg_relevance_score  = 0.31,
        top_relevance_score  = 0.42,
        faithfulness_score   = 0.55,
        was_rewritten        = True,
        rewritten_query      = "What are beginner investment strategies?",
        answer               = "For beginners...",
        user_feedback        = 0,
    )

    id3 = logger.log(
        query                = "Difference between ETF and mutual fund?",
        query_type           = "comparative",
        strategy_used        = "hybrid",
        top_k                = 6,
        num_chunks_retrieved = 6,
        num_chunks_passed    = 5,
        avg_relevance_score  = 0.79,
        top_relevance_score  = 0.94,
        faithfulness_score   = 0.91,
        answer               = "ETFs differ from mutual funds in...",
        user_feedback        = 1,
    )

    print(f"Logged 3 entries with IDs: {id1}, {id2}, {id3}")

    print("\nStrategy stats:")
    stats = logger.get_strategy_stats()
    for key, val in stats.items():
        print(f"  {key}: {val}")

    logger.close()
    print("\nFeedback logger working correctly!")