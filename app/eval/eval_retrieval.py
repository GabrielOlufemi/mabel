# app/eval/eval_retrieval.py
"""
Retrieval evaluation — Baseline vs Mabel pipeline.

Metrics computed for both pipelines:
  - NDCG@K  (primary — Normalized Discounted Cumulative Gain)
  - MRR     (Mean Reciprocal Rank)
  - MAP     (Mean Average Precision)
  - Hit Rate @1, @3, @K
  - Precision@K
  - Recall@K
  - Mean similarity score

queries.json format:
[
    {
        "query":        "What is the role of *args in Python?",
        "user_id":      "test-user-id",
        "file_ids":     ["doc-file-id-1"],
        "relevant_ids": ["doc-file-id-1"]
    }
]
"""

import json
import logging
from pathlib import Path

from app.eval.baseline_rag import BaselineRAG
from app.eval.base import (
    mean_reciprocal_rank,
    mean_average_precision,
    mean_ndcg_at_k,
    hit_rate_at_k,
    mean_precision_at_k,
    mean_recall_at_k,
    mean_score,
    score_delta,
    save_report,
)
from app.services.embedding_service import get_embedding_service
from app.db.vector_store import get_vector_store
from app.services.llm_utils import rewrite_query
from app.services.reranker import get_reranker
from app.config import settings

logger = logging.getLogger(__name__)
TEST_DATA_PATH = Path(__file__).parent / "test_data" / "queries.json"


# ── Mabel pipeline ─────────────────────────────────────────────────────────────


def _mabel_retrieve(query: str, user_id: str, file_ids: list | None = None) -> dict:
    rewritten = rewrite_query(query)
    embedder = get_embedding_service()
    vector_store = get_vector_store()
    reranker = get_reranker()

    embedding = embedder.generate_single_embedding(rewritten)
    raw = vector_store.search(
        user_id=user_id,
        query_embedding=embedding,
        top_k=settings.INITIAL_RETRIEVAL_K,
        file_ids=file_ids or [],
    )

    if not raw:
        return {
            "rewritten_query": rewritten,
            "results": [],
            "top_score": 0.0,
            "mean_score": 0.0,
        }

    reranked = reranker.rerank(query=rewritten, results=raw, top_k=settings.FINAL_TOP_K)

    results = []
    for idx, r in enumerate(reranked):
        meta = r.get("metadata", {})
        results.append(
            {
                "rank": idx + 1,
                "chunk_text": r.get("chunk_text", ""),
                "filename": meta.get("filename", "unknown"),
                "document_id": meta.get("file_id", ""),
                "similarity_score": round(r.get("similarity_score", 0.0), 4),
                "rerank_score": round(r.get("rerank_score", 0.0), 4),
                "chunk_index": meta.get("chunk_index", 0),
            }
        )

    scores = [r["similarity_score"] for r in results]
    return {
        "rewritten_query": rewritten,
        "results": results,
        "top_score": round(scores[0], 4) if scores else 0.0,
        "mean_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
    }


# ── Runner ──────────────────────────────────────────────────────────────────────


def run(notes: str = "") -> dict:
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"Test data not found at {TEST_DATA_PATH}")

    test_cases = json.loads(TEST_DATA_PATH.read_text())
    if not test_cases:
        raise ValueError("queries.json is empty")

    logger.info(f"Running retrieval eval on {len(test_cases)} queries...")

    baseline = BaselineRAG(top_k=settings.FINAL_TOP_K)
    per_query = []

    for i, case in enumerate(test_cases):
        query = case["query"]
        user_id = case["user_id"]
        file_ids = case.get("file_ids")
        relevant_ids = set(case.get("relevant_ids", []))

        logger.info(f"[{i+1}/{len(test_cases)}] {query[:70]}")

        # Search across ALL user documents — no file_id filter.
        # relevant_ids is the ground truth of which doc(s) should rank top.
        # Passing file_ids here would pre-filter to the answer doc,
        # making every metric trivially perfect and the eval meaningless.
        b = baseline.retrieve_with_scores(query, user_id, file_ids=None)

        # Rate limit guard — Gemini free tier allows 20 RPM.
        # rewrite_query() fires one Gemini call per query so we pace
        # to ~4 seconds between queries (15/min) to stay safely under.
        import time

        time.sleep(4)

        m = _mabel_retrieve(query, user_id, file_ids=None)

        rel_list = list(relevant_ids)  # convert set → list for JSON serialization
        per_query.append(
            {
                "query": query,
                "rewritten_query": m["rewritten_query"],
                "relevant_ids": rel_list,
                "baseline": {**b, "relevant_ids": relevant_ids},
                "mabel": {**m, "relevant_ids": relevant_ids},
                "top_score_delta": score_delta(b["top_score"], m["top_score"]),
            }
        )

    K = settings.FINAL_TOP_K
    b_pairs = [
        (q["baseline"]["results"], q["baseline"]["relevant_ids"]) for q in per_query
    ]
    m_pairs = [(q["mabel"]["results"], q["mabel"]["relevant_ids"]) for q in per_query]

    def pair_metric(fn, *args):
        return {"baseline": fn(b_pairs, *args), "mabel": fn(m_pairs, *args)}

    aggregate = {
        "total_queries": len(test_cases),
        "k": K,
        "ndcg_at_k": pair_metric(mean_ndcg_at_k, K),
        "mrr": pair_metric(mean_reciprocal_rank),
        "map": pair_metric(mean_average_precision),
        "hit_rate": {
            "at_1": pair_metric(hit_rate_at_k, 1),
            "at_3": pair_metric(hit_rate_at_k, 3),
            "at_k": pair_metric(hit_rate_at_k, K),
        },
        "precision_at_k": pair_metric(mean_precision_at_k, K),
        "recall_at_k": pair_metric(mean_recall_at_k, K),
        "mean_similarity": {
            "baseline": mean_score(
                [r for q in per_query for r in q["baseline"]["results"]]
            ),
            "mabel": mean_score([r for q in per_query for r in q["mabel"]["results"]]),
        },
        "mabel_wins": sum(1 for q in per_query if q["top_score_delta"]["delta"] > 0),
        "baseline_wins": sum(
            1 for q in per_query if q["top_score_delta"]["delta"] <= 0
        ),
    }

    # Attach deltas to scalar metric pairs
    for key in ("ndcg_at_k", "mrr", "map", "precision_at_k", "recall_at_k"):
        aggregate[key]["delta"] = score_delta(
            aggregate[key]["baseline"], aggregate[key]["mabel"]
        )

    # Strip the sets from per_query before JSON serialization
    for q in per_query:
        q["baseline"].pop("relevant_ids", None)
        q["mabel"].pop("relevant_ids", None)

    results = {"aggregate": aggregate, "per_query": per_query}
    path = save_report("retrieval", results)
    logger.info(f"Report saved: {path}")
    _print_console(aggregate)
    return results


def _print_console(agg: dict) -> None:
    K = agg["k"]
    w = 64

    def row(label, b, m):
        d = m - b
        sign = "+" if d >= 0 else ""
        print(f"  {label:<28} {b:>8.4f}   {m:>8.4f}   {sign}{d:.4f}")

    print(f"\n{'═' * w}")
    print(f"  RETRIEVAL EVAL — BASELINE vs MABEL   (K={K})")
    print(f"{'═' * w}")
    print(
        f"  Queries: {agg['total_queries']}  |  "
        f"Mabel wins: {agg['mabel_wins']}  |  "
        f"Baseline wins: {agg['baseline_wins']}"
    )
    print(f"\n  {'Metric':<28} {'Baseline':>8}   {'Mabel':>8}   {'Δ':>7}")
    print(f"  {'-' * 58}")
    print(f"  ── Primary")
    row(f"NDCG@{K}", agg["ndcg_at_k"]["baseline"], agg["ndcg_at_k"]["mabel"])
    print(f"  ── Ranking Quality")
    row("MRR", agg["mrr"]["baseline"], agg["mrr"]["mabel"])
    row("MAP", agg["map"]["baseline"], agg["map"]["mabel"])
    print(f"  ── Coverage")
    row(
        "Hit Rate @1",
        agg["hit_rate"]["at_1"]["baseline"],
        agg["hit_rate"]["at_1"]["mabel"],
    )
    row(
        "Hit Rate @3",
        agg["hit_rate"]["at_3"]["baseline"],
        agg["hit_rate"]["at_3"]["mabel"],
    )
    row(
        f"Hit Rate @{K}",
        agg["hit_rate"]["at_k"]["baseline"],
        agg["hit_rate"]["at_k"]["mabel"],
    )
    print(f"  ── Precision / Recall")
    row(
        f"Precision@{K}",
        agg["precision_at_k"]["baseline"],
        agg["precision_at_k"]["mabel"],
    )
    row(f"Recall@{K}", agg["recall_at_k"]["baseline"], agg["recall_at_k"]["mabel"])
    print(f"  ── Score Distributions")
    row(
        "Mean Similarity",
        agg["mean_similarity"]["baseline"],
        agg["mean_similarity"]["mabel"],
    )
    print(f"{'═' * w}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
