# adaptation/adaptation_engine.py

import os
import json
from dotenv import load_dotenv
from adaptation.feedback_logger import FeedbackLogger

load_dotenv()

# default strategy map (same as strategy_router.py)
DEFAULT_STRATEGY_MAP = {
    "factual":     {"strategy": "dense",  "top_k": 3},
    "reasoning":   {"strategy": "hybrid", "top_k": 5},
    "comparative": {"strategy": "hybrid", "top_k": 6},
    "ambiguous":   {"strategy": "sparse", "top_k": 3},
}

# minimum number of logs needed before adapting
MIN_SAMPLES = 5

# weights for scoring each strategy
SCORE_WEIGHTS = {
    "avg_relevance":    0.4,
    "avg_faithfulness": 0.4,
    "avg_feedback":     0.2,
}


def compute_composite_score(stats: dict) -> float:
    """Compute a single composite score from strategy stats."""
    return (
        stats["avg_relevance"]    * SCORE_WEIGHTS["avg_relevance"] +
        stats["avg_faithfulness"] * SCORE_WEIGHTS["avg_faithfulness"] +
        stats["avg_feedback"]     * SCORE_WEIGHTS["avg_feedback"]
    )


def adapt_strategy_map(current_map: dict, logger: FeedbackLogger) -> dict:
    """
    Look at feedback logs and update strategy map if a better
    strategy is found for any query type.
    Returns updated strategy map.
    """
    stats    = logger.get_strategy_stats()
    new_map  = {k: v.copy() for k, v in current_map.items()}
    changes  = []

    # group stats by query type
    by_type = {}
    for key, val in stats.items():
        query_type, strategy = key.split(":")
        if query_type not in by_type:
            by_type[query_type] = {}
        by_type[query_type][strategy] = val

    for query_type, strategies in by_type.items():
        if query_type not in new_map:
            continue

        # only adapt if we have enough samples
        best_strategy = None
        best_score    = -1

        for strategy, stat in strategies.items():
            if stat["count"] < MIN_SAMPLES:
                continue
            score = compute_composite_score(stat)
            if score > best_score:
                best_score    = score
                best_strategy = strategy

        if best_strategy and best_strategy != new_map[query_type]["strategy"]:
            old_strategy = new_map[query_type]["strategy"]
            new_map[query_type]["strategy"] = best_strategy
            changes.append({
                "query_type":   query_type,
                "old_strategy": old_strategy,
                "new_strategy": best_strategy,
                "score":        round(best_score, 3),
            })

    return new_map, changes


class AdaptationEngine:

    def __init__(self):
        self.logger       = FeedbackLogger()
        self.strategy_map = {k: v.copy() for k, v in DEFAULT_STRATEGY_MAP.items()}

    def run(self) -> dict:
        """
        Run one adaptation cycle.
        Returns dict with: updated_map, changes, stats
        """
        stats = self.logger.get_strategy_stats()

        if not stats:
            return {
                "updated_map": self.strategy_map,
                "changes":     [],
                "stats":       {},
                "message":     "No data yet — need more query logs",
            }

        new_map, changes = adapt_strategy_map(self.strategy_map, self.logger)
        self.strategy_map = new_map

        return {
            "updated_map": new_map,
            "changes":     changes,
            "stats":       stats,
            "message":     f"{len(changes)} strategy updates applied" if changes else "No changes needed",
        }

    def get_strategy(self, query_type: str) -> dict:
        """Get current strategy for a query type."""
        return self.strategy_map.get(query_type, DEFAULT_STRATEGY_MAP["factual"])

    def close(self):
        self.logger.close()


if __name__ == "__main__":
    from adaptation.feedback_logger import FeedbackLogger

    # seed with enough logs to trigger adaptation
    logger = FeedbackLogger()

    print("Seeding feedback logs...\n")

    # seed: hybrid is performing better than dense for factual queries
    for i in range(6):
        logger.log(
            query                = f"What is investment strategy {i}?",
            query_type           = "factual",
            strategy_used        = "dense",
            top_k                = 3,
            num_chunks_retrieved = 3,
            num_chunks_passed    = 2,
            avg_relevance_score  = 0.55,
            top_relevance_score  = 0.60,
            faithfulness_score   = 0.58,
            user_feedback        = 0,
        )

    for i in range(6):
        logger.log(
            query                = f"What is dollar cost averaging {i}?",
            query_type           = "factual",
            strategy_used        = "hybrid",
            top_k                = 5,
            num_chunks_retrieved = 5,
            num_chunks_passed    = 4,
            avg_relevance_score  = 0.88,
            top_relevance_score  = 0.94,
            faithfulness_score   = 0.91,
            user_feedback        = 1,
        )

    logger.close()

    print("Running adaptation engine...\n")
    engine = AdaptationEngine()
    result = engine.run()

    print(f"Message: {result['message']}")
    print(f"\nUpdated strategy map:")
    for qtype, config in result["updated_map"].items():
        print(f"  {qtype}: {config['strategy']} (top_k={config['top_k']})")

    if result["changes"]:
        print(f"\nChanges made:")
        for change in result["changes"]:
            print(f"  {change['query_type']}: {change['old_strategy']} → {change['new_strategy']} (score: {change['score']})")

    engine.close()