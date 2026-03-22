# app/eval/eval_query_rewriting.py
"""
Query rewriting evaluation.

Measures two things independently:

1. Rewrite quality  (LLM judge)
   - Intent preservation  — does the rewrite mean the same thing?
   - Clarity improvement  — is it better for retrieval?
   - Grounding           — no hallucinated concepts added?

2. Retrieval impact (IR metrics — no reranker, isolates rewrite contribution)
   Comparing raw query vs rewritten query on the same vector store:
   - NDCG@K, MRR, MAP, Hit Rate @K, Precision@K, Recall@K

Uses the same queries.json as eval_retrieval.py.
"""

import json
import logging
from pathlib import Path

from app.eval.base import (
    llm_judge,
    # IR metrics
    mean_ndcg_at_k,
    mean_reciprocal_rank,
    mean_average_precision,
    hit_rate_at_k,
    mean_precision_at_k,
    mean_recall_at_k,
    mean_score,
    score_delta,
    save_report,
)
from app.services.llm_utils import rewrite_query
from app.services.embedding_service import get_embedding_service
from app.db.vector_store import get_vector_store
from app.config import settings

logger = logging.getLogger(__name__)
TEST_DATA_PATH = Path(__file__).parent / "test_data" / "queries.json"

K = settings.FINAL_TOP_K


# ── Helpers ────────────────────────────────────────────────────────────────────

def _search(query: str, user_id: str, file_ids: list | None, top_k: int) -> list[dict]:
    """Embed and search — no reranking so we isolate the rewrite's contribution."""
    embedder     = get_embedding_service()
    vector_store = get_vector_store()
    embedding    = embedder.generate_single_embedding(query)
    raw          = vector_store.search(
        user_id=user_id,
        query_embedding=embedding,
        top_k=top_k,
        file_ids=file_ids or [],
    )
    results = []
    for idx, r in enumerate(raw):
        meta = r.get("metadata", {})
        results.append({
            "rank":             idx + 1,
            "document_id":      meta.get("file_id", ""),
            "similarity_score": round(r.get("similarity_score", 0.0), 4),
        })
    return results


def _judge_rewrite(original: str, rewritten: str) -> dict:
    """Ask the LLM to score rewrite quality on three dimensions."""
    return llm_judge(
        input_text=original,
        output_text=rewritten,
        dimensions=["intent_preservation", "specificity_gain", "clarity"],
    )


# ── Runner ──────────────────────────────────────────────────────────────────────

def run(notes: str = "") -> dict:
    """
    Run the query rewriting eval and write a report.

    For each test query:
      1. Call rewrite_query() to get the rewritten version
      2. LLM judge scores the rewrite on 3 dimensions (1–5 scale)
      3. Compare retrieval metrics — raw query vs rewritten query
         (cosine similarity only, no reranker, to isolate the rewrite)
    """
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"Test data not found at {TEST_DATA_PATH}")

    test_cases = json.loads(TEST_DATA_PATH.read_text())
    if not test_cases:
        raise ValueError("queries.json is empty")

    logger.info(f"Running query rewriting eval on {len(test_cases)} queries...")

    per_query = []

    for i, case in enumerate(test_cases):
        query        = case["query"]
        user_id      = case["user_id"]
        file_ids     = case.get("file_ids")
        relevant_ids = set(case.get("relevant_ids", []))

        logger.info(f"[{i+1}/{len(test_cases)}] {query[:70]}")

        # ── Step 1: rewrite ───────────────────────────────────────────────────
        rewritten = rewrite_query(query)
        changed   = rewritten.strip().lower() != query.strip().lower()
        logger.info(f"  → {rewritten[:70]}")

        # ── Step 2: LLM judge ─────────────────────────────────────────────────
        try:
            judge = _judge_rewrite(query, rewritten)
        except Exception as e:
            logger.warning(f"Judge failed for query {i}: {e}")
            judge = {
                "scores":     {"intent_preservation": 0, "specificity_gain": 0, "clarity": 0},
                "reasoning":  {},
                "mean_score": 0.0,
            }

        # ── Step 3: retrieval impact ──────────────────────────────────────────
        raw_results      = _search(query,     user_id, file_ids, K)
        rewritten_results = _search(rewritten, user_id, file_ids, K)

        raw_pair      = [(raw_results,       relevant_ids)]
        rewritten_pair = [(rewritten_results, relevant_ids)]

        impact = {
            "raw": {
                "ndcg_at_k":     mean_ndcg_at_k(raw_pair, K),
                "mrr":           mean_reciprocal_rank(raw_pair),
                "map":           mean_average_precision(raw_pair),
                "hit_at_k":      hit_rate_at_k(raw_pair, K),
                "precision_at_k": mean_precision_at_k(raw_pair, K),
                "recall_at_k":   mean_recall_at_k(raw_pair, K),
                "mean_score":    mean_score(raw_results),
            },
            "rewritten": {
                "ndcg_at_k":     mean_ndcg_at_k(rewritten_pair, K),
                "mrr":           mean_reciprocal_rank(rewritten_pair),
                "map":           mean_average_precision(rewritten_pair),
                "hit_at_k":      hit_rate_at_k(rewritten_pair, K),
                "precision_at_k": mean_precision_at_k(rewritten_pair, K),
                "recall_at_k":   mean_recall_at_k(rewritten_pair, K),
                "mean_score":    mean_score(rewritten_results),
            },
        }

        impact["deltas"] = {
            metric: score_delta(impact["raw"][metric], impact["rewritten"][metric])
            for metric in impact["raw"]
        }
        impact["rewrite_helped"] = impact["deltas"]["ndcg_at_k"]["delta"] > 0

        per_query.append({
            "original":   query,
            "rewritten":  rewritten,
            "changed":    changed,
            "judge":      judge,
            "impact":     impact,
        })

    # ── Aggregate ──────────────────────────────────────────────────────────────
    def avg(key_path: list) -> float:
        vals = []
        for q in per_query:
            obj = q
            for k in key_path:
                obj = obj.get(k, {}) if isinstance(obj, dict) else {}
            if isinstance(obj, (int, float)):
                vals.append(obj)
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    aggregate = {
        "total_queries":         len(test_cases),
        "queries_rewritten":     sum(1 for q in per_query if q["changed"]),
        "queries_unchanged":     sum(1 for q in per_query if not q["changed"]),
        "rewrite_helped_ndcg":   sum(1 for q in per_query if q["impact"]["rewrite_helped"]),

        "avg_judge_scores": {
            "overall":             avg(["judge", "mean_score"]),
            "intent_preservation": avg(["judge", "scores", "intent_preservation"]),
            "specificity_gain":    avg(["judge", "scores", "specificity_gain"]),
            "clarity":             avg(["judge", "scores", "clarity"]),
        },

        "avg_retrieval_impact": {
            metric: {
                "raw":       avg(["impact", "raw",       metric]),
                "rewritten": avg(["impact", "rewritten", metric]),
            }
            for metric in ["ndcg_at_k", "mrr", "map", "hit_at_k", "precision_at_k", "recall_at_k", "mean_score"]
        },
    }

    results = {"aggregate": aggregate, "per_query": per_query}
    path    = save_report("query_rewriting", results)
    logger.info(f"Report saved: {path}")
    _print_console(aggregate)
    return results


def _print_console(agg: dict) -> None:
    w  = 64
    js = agg["avg_judge_scores"]
    ri = agg["avg_retrieval_impact"]

    def row(label, raw, rew):
        d    = rew - raw
        sign = "+" if d >= 0 else ""
        print(f"  {label:<26} {raw:>8.4f}   {rew:>8.4f}   {sign}{d:.4f}")

    print(f"\n{'═' * w}")
    print(f"  QUERY REWRITING EVAL")
    print(f"{'═' * w}")
    print(f"  Queries: {agg['total_queries']}  |  "
          f"Rewritten: {agg['queries_rewritten']}  |  "
          f"Unchanged: {agg['queries_unchanged']}  |  "
          f"Rewrite helped (NDCG): {agg['rewrite_helped_ndcg']}")

    print(f"\n  ── LLM Judge Scores (1–5 scale)")
    print(f"  {'Overall':<30} {js['overall']:.2f}")
    print(f"  {'Intent Preservation':<30} {js['intent_preservation']:.2f}")
    print(f"  {'Specificity Gain':<30} {js['specificity_gain']:.2f}")
    print(f"  {'Clarity':<30} {js['clarity']:.2f}")

    print(f"\n  ── Retrieval Impact  (raw vs rewritten, no reranker)")
    print(f"  {'Metric':<26} {'Raw':>8}   {'Rewritten':>9}   {'Δ':>7}")
    print(f"  {'-' * 56}")
    row(f"NDCG@{K} (primary)", ri["ndcg_at_k"]["raw"],     ri["ndcg_at_k"]["rewritten"])
    row("MRR",                 ri["mrr"]["raw"],            ri["mrr"]["rewritten"])
    row("MAP",                 ri["map"]["raw"],            ri["map"]["rewritten"])
    row(f"Hit Rate @{K}",      ri["hit_at_k"]["raw"],       ri["hit_at_k"]["rewritten"])
    row(f"Precision@{K}",      ri["precision_at_k"]["raw"], ri["precision_at_k"]["rewritten"])
    row(f"Recall@{K}",         ri["recall_at_k"]["raw"],    ri["recall_at_k"]["rewritten"])
    row("Mean Similarity",     ri["mean_score"]["raw"],     ri["mean_score"]["rewritten"])
    print(f"{'═' * w}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()