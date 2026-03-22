# app/eval/base.py
"""
Shared utilities for the Mabel eval suite.

Three responsibilities:
  1. LLM judge   — asks Gemini to score outputs against a rubric
  2. IR metrics  — MRR, hit@k, mean score helpers
  3. Report writer — formats and saves eval results to reports/
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from app.services.llm_utils import _call_gemini

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
EVAL_DIR    = Path(__file__).parent
REPORTS_DIR = EVAL_DIR / "reports"
TEST_DATA   = EVAL_DIR / "test_data"
REPORTS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  1. LLM JUDGE
# ══════════════════════════════════════════════════════════════════════════════

RUBRIC = {
    "relevance":           "How relevant is the output to the input query? (1 = irrelevant, 5 = perfectly relevant)",
    "faithfulness":        "Does the output stay faithful to the source without hallucinating? (1 = fabricated, 5 = fully grounded)",
    "completeness":        "Does the output cover the key points? (1 = very incomplete, 5 = comprehensive)",
    "clarity":             "Is the output clear and well-structured? (1 = confusing, 5 = very clear)",
    "intent_preservation": "Does the rewritten query preserve the original intent? (1 = intent lost, 5 = fully preserved)",
    "specificity_gain":    "Did rewriting make the query more retrieval-friendly? (1 = no improvement, 5 = significantly better)",
}


def llm_judge(
    input_text: str,
    output_text: str,
    dimensions: list[str],
    context: Optional[str] = None,
) -> dict:
    """
    Ask Gemini to score an output against selected rubric dimensions.

    Args:
        input_text:  The original query or source text.
        output_text: The text being evaluated.
        dimensions:  Rubric dimension names to score — must be keys in RUBRIC.
        context:     Optional source document for faithfulness checks.

    Returns:
        {
            "scores":     {"dimension": int, ...},
            "reasoning":  {"dimension": str, ...},
            "mean_score": float,
            "raw":        str,
        }
    """
    unknown = [d for d in dimensions if d not in RUBRIC]
    if unknown:
        raise ValueError(f"Unknown rubric dimensions: {unknown}. Valid: {list(RUBRIC)}")

    rubric_block  = "\n".join(f"- {d}: {RUBRIC[d]}" for d in dimensions)
    context_block = f"\nSource context:\n{context}\n" if context else ""

    prompt = f"""You are an objective evaluator for an AI study assistant.
Score the following output on each dimension using integers 1 to 5.

Input:
{input_text}
{context_block}
Output to evaluate:
{output_text}

Scoring dimensions:
{rubric_block}

Return ONLY valid JSON in this exact format — no markdown, no preamble:
{{
  "scores": {{{", ".join(f'"{d}": <int>' for d in dimensions)}}},
  "reasoning": {{{", ".join(f'"{d}": "<one sentence>"' for d in dimensions)}}}
}}"""

    system = (
        "You are a strict, consistent evaluator. "
        "Return only valid JSON. Never include markdown fences or extra text."
    )

    try:
        raw     = _call_gemini(system, prompt, temperature=0.0)
        cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        parsed  = json.loads(cleaned)

        scores    = parsed.get("scores", {})
        reasoning = parsed.get("reasoning", {})

        for d in dimensions:
            if d not in scores:
                raise ValueError(f"Missing score for dimension '{d}'")
            if not isinstance(scores[d], (int, float)) or not (1 <= scores[d] <= 5):
                raise ValueError(f"Score for '{d}' out of range: {scores[d]}")

        mean = round(sum(scores[d] for d in dimensions) / len(dimensions), 2)

        return {
            "scores":     {d: int(scores[d]) for d in dimensions},
            "reasoning":  {d: reasoning.get(d, "") for d in dimensions},
            "mean_score": mean,
            "raw":        raw,
        }

    except json.JSONDecodeError as e:
        logger.error(f"LLM judge JSON parse error: {e}\nRaw: {raw[:300]}")
        raise ValueError(f"Judge returned invalid JSON: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  2. IR METRICS
# ══════════════════════════════════════════════════════════════════════════════

def hit_at_k(results: list[dict], relevant_ids: set[str], k: int) -> bool:
    """
    Hit@K — did at least one relevant document appear in the top-k results?

    Args:
        results:      Ranked result dicts, each with a 'document_id' key.
        relevant_ids: Set of document_ids considered relevant for this query.
        k:            Cutoff rank.
    """
    return any(r["document_id"] in relevant_ids for r in results[:k])


def reciprocal_rank(results: list[dict], relevant_ids: set[str]) -> float:
    """
    Reciprocal Rank — 1/rank of the first relevant result, or 0.0 if none found.
    """
    for idx, result in enumerate(results):
        if result["document_id"] in relevant_ids:
            return round(1.0 / (idx + 1), 4)
    return 0.0


def mean_reciprocal_rank(query_results: list[tuple[list[dict], set[str]]]) -> float:
    """
    MRR across multiple queries.

    Args:
        query_results: List of (results, relevant_ids) tuples, one per query.
    """
    if not query_results:
        return 0.0
    rrs = [reciprocal_rank(results, rel) for results, rel in query_results]
    return round(sum(rrs) / len(rrs), 4)


def hit_rate_at_k(query_results: list[tuple[list[dict], set[str]]], k: int) -> float:
    """
    Hit Rate@K — proportion of queries with at least one relevant result in top-k.
    """
    if not query_results:
        return 0.0
    hits = sum(1 for results, rel in query_results if hit_at_k(results, rel, k))
    return round(hits / len(query_results), 4)


def ndcg_at_k(results: list[dict], relevant_ids: set[str], k: int) -> float:
    """
    NDCG@K — Normalized Discounted Cumulative Gain.
    The gold-standard IR metric. Rewards relevant results ranked highly,
    with diminishing credit as rank position increases. Normalized 0.0–1.0.
    """
    import math

    def dcg(ranked: list) -> float:
        return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(ranked))

    top_k  = results[:k]
    gains  = [1 if r["document_id"] in relevant_ids else 0 for r in top_k]
    actual = dcg(gains)
    ideal  = dcg(sorted(gains, reverse=True))
    return round(actual / ideal, 4) if ideal > 0 else 0.0


def precision_at_k(results: list[dict], relevant_ids: set[str], k: int) -> float:
    """Precision@K — fraction of the top-k results that are relevant."""
    top_k = results[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for r in top_k if r["document_id"] in relevant_ids)
    return round(hits / len(top_k), 4)


def recall_at_k(results: list[dict], relevant_ids: set[str], k: int) -> float:
    """Recall@K — fraction of all relevant docs retrieved in the top-k."""
    if not relevant_ids:
        return 0.0
    hits = sum(1 for r in results[:k] if r["document_id"] in relevant_ids)
    return round(hits / len(relevant_ids), 4)


def average_precision(results: list[dict], relevant_ids: set[str]) -> float:
    """
    Average Precision (AP) for a single query.
    Averages Precision@K at each rank where a relevant doc appears.
    Used to compute MAP.
    """
    if not relevant_ids:
        return 0.0
    hits, score_sum = 0, 0.0
    for idx, result in enumerate(results):
        if result["document_id"] in relevant_ids:
            hits      += 1
            score_sum += hits / (idx + 1)
    return round(score_sum / len(relevant_ids), 4)


def mean_average_precision(query_results: list[tuple]) -> float:
    """MAP — Mean Average Precision across all queries."""
    if not query_results:
        return 0.0
    aps = [average_precision(results, rel) for results, rel in query_results]
    return round(sum(aps) / len(aps), 4)


def mean_ndcg_at_k(query_results: list[tuple], k: int) -> float:
    """Mean NDCG@K across all queries."""
    if not query_results:
        return 0.0
    return round(sum(ndcg_at_k(r, rel, k) for r, rel in query_results) / len(query_results), 4)


def mean_precision_at_k(query_results: list[tuple], k: int) -> float:
    """Mean Precision@K across all queries."""
    if not query_results:
        return 0.0
    return round(sum(precision_at_k(r, rel, k) for r, rel in query_results) / len(query_results), 4)


def mean_recall_at_k(query_results: list[tuple], k: int) -> float:
    """Mean Recall@K across all queries."""
    if not query_results:
        return 0.0
    return round(sum(recall_at_k(r, rel, k) for r, rel in query_results) / len(query_results), 4)


def mean_score(results: list[dict], score_key: str = "similarity_score") -> float:
    """
    Average score across a result list.

    Args:
        score_key: 'similarity_score' for baseline, 'rerank_score' for Mabel.
    """
    scores = [r.get(score_key, 0.0) for r in results if score_key in r]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def score_delta(baseline_score: float, mabel_score: float) -> dict:
    """
    Absolute and relative improvement of Mabel over baseline.

    Returns:
        {
            "baseline": float,
            "mabel":    float,
            "delta":    float,
            "pct_gain": float,
        }
    """
    delta    = round(mabel_score - baseline_score, 4)
    pct_gain = round((delta / baseline_score * 100), 2) if baseline_score else 0.0
    return {
        "baseline": baseline_score,
        "mabel":    mabel_score,
        "delta":    delta,
        "pct_gain": pct_gain,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  3. REPORT WRITER
# ══════════════════════════════════════════════════════════════════════════════

def save_report(name: str, data: dict[str, Any]) -> Path:
    """
    Save eval results to reports/ as a timestamped JSON file.

    Args:
        name: Short eval name e.g. 'retrieval', 'query_rewriting'.
        data: Results dict to serialise.

    Returns:
        Path to the written file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = REPORTS_DIR / f"{name}_{timestamp}.json"

    payload = {
        "eval":    name,
        "run_at":  datetime.now().isoformat(),
        "results": data,
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info(f"Report saved: {filename}")
    return filename


def print_summary(name: str, metrics: dict[str, Any]) -> None:
    """
    Print a human-readable summary of eval metrics to stdout.
    """
    width = 60
    print(f"\n{'═' * width}")
    print(f"  {name.upper()} EVAL RESULTS")
    print(f"{'═' * width}")
    for key, val in metrics.items():
        if isinstance(val, dict):
            print(f"\n  {key}:")
            for k, v in val.items():
                print(f"    {k:<30} {v}")
        else:
            print(f"  {key:<32} {val}")
    print(f"{'═' * width}\n")