# adaptation/drift_detector.py

import os
import numpy as np
from typing import List
from dotenv import load_dotenv

load_dotenv()

# PSI thresholds (same concept you used in FinScore AI)
PSI_LOW      = 0.5   # no significant drift
PSI_MODERATE = 2.0   # moderate drift, worth monitoring
# above 0.2 = significant drift, trigger re-indexing


def compute_psi(reference: List[float], current: List[float], bins: int = 10) -> float:
    """
    Compute Population Stability Index between two distributions.
    reference: embedding values from original index
    current:   embedding values from recent queries
    Returns PSI score.
    """
    # define bin edges from reference distribution
    min_val = min(min(reference), min(current))
    max_val = max(max(reference), max(current))
    bin_edges = np.linspace(min_val, max_val, bins + 1)

    # compute proportions per bin
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current,   bins=bin_edges)

    # avoid division by zero
    ref_props = (ref_counts + 1e-6) / (len(reference) + 1e-6 * bins)
    cur_props = (cur_counts + 1e-6) / (len(current)   + 1e-6 * bins)

    # PSI formula
    psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
    return float(psi)


def embeddings_to_scalar(embeddings: List[List[float]]) -> List[float]:
    """
    Reduce embedding vectors to scalar values for PSI computation.
    Uses L2 norm of each embedding vector — more stable than mean.
    """
    return [float(np.linalg.norm(emb)) for emb in embeddings]


class DriftDetector:

    def __init__(self):
        self.reference_scalars = None  # from original indexed corpus

    def set_reference(self, embeddings: List[List[float]]):
        """Set the reference distribution from the original corpus embeddings."""
        self.reference_scalars = embeddings_to_scalar(embeddings)
        print(f"Reference set with {len(self.reference_scalars)} embeddings")

    def check_drift(self, recent_embeddings: List[List[float]]) -> dict:
        """
        Compare recent query embeddings against reference corpus.
        Returns drift report.
        """
        if self.reference_scalars is None:
            return {"error": "Reference not set. Call set_reference() first."}

        if len(recent_embeddings) < 10:
            return {
                "psi":          None,
                "drift_level":  "unknown",
                "action":       "need at least 10 recent embeddings",
                "should_reindex": False,
            }

        current_scalars = embeddings_to_scalar(recent_embeddings)
        psi             = compute_psi(self.reference_scalars, current_scalars)

        if psi < PSI_LOW:
            drift_level     = "low"
            action          = "No action needed"
            should_reindex  = False
        elif psi < PSI_MODERATE:
            drift_level     = "moderate"
            action          = "Monitor closely, consider partial re-indexing"
            should_reindex  = False
        else:
            drift_level     = "high"
            action          = "Re-indexing recommended"
            should_reindex  = True

        return {
            "psi":            round(psi, 4),
            "drift_level":    drift_level,
            "action":         action,
            "should_reindex": should_reindex,
            "reference_size": len(self.reference_scalars),
            "current_size":   len(current_scalars),
        }


if __name__ == "__main__":
    print("Drift Detector Test\n" + "="*40)

    detector = DriftDetector()

    # simulate reference corpus embeddings (768-dim)
    np.random.seed(42)
    reference_embeddings = [
        np.random.normal(0.0, 0.1, 768).tolist()
        for _ in range(200)
    ]
    detector.set_reference(reference_embeddings)

    # test 1 — similar distribution (low drift)
    similar_embeddings = [
        np.random.normal(0.01, 0.1, 768).tolist()
        for _ in range(50)
    ]
    result1 = detector.check_drift(similar_embeddings)
    print(f"\nTest 1 — Similar distribution:")
    print(f"  PSI:          {result1['psi']}")
    print(f"  Drift level:  {result1['drift_level']}")
    print(f"  Action:       {result1['action']}")
    print(f"  Should reindex: {result1['should_reindex']}")

    # test 2 — shifted distribution (high drift)
    shifted_embeddings = [
        np.random.normal(0.5, 0.3, 768).tolist()
        for _ in range(50)
    ]
    result2 = detector.check_drift(shifted_embeddings)
    print(f"\nTest 2 — Shifted distribution:")
    print(f"  PSI:          {result2['psi']}")
    print(f"  Drift level:  {result2['drift_level']}")
    print(f"  Action:       {result2['action']}")
    print(f"  Should reindex: {result2['should_reindex']}")