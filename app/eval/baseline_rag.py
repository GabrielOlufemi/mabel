# app/eval/baseline_rag.py
"""
Baseline RAG pipeline — your typical rag system.

No query rewriting, no classification, no reranking, no history,
no similarity threshold gating. Raw query goes straight to the
vector store and we return the top-k chunks by cosine similarity.

This exists purely as a comparison target for eval_retrieval.py.
"""

import logging
from typing import Optional

from app.services.embedding_service import get_embedding_service
from app.db.vector_store import get_vector_store

logger = logging.getLogger(__name__)


class BaselineRAG:
    """
    Minimal retrieval pipeline: embed → search → return.
    No preprocessing, no postprocessing.
    """

    def __init__(self, top_k: int = 5):
        self.top_k       = top_k
        self.embedder    = get_embedding_service()
        self.vector_store = get_vector_store()

    def retrieve(
        self,
        query: str,
        user_id: str,
        file_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Retrieve top-k chunks for a raw query using cosine similarity only.

        Args:
            query:    Raw user query — no rewriting applied.
            user_id:  Used to scope the ChromaDB collection.
            file_ids: Optional list of document IDs to filter by.
                      Pass None to search across all user documents.

        Returns:
            List of result dicts, sorted by similarity_score descending:
            [
                {
                    "rank":             int,
                    "chunk_text":       str,
                    "filename":         str,
                    "document_id":      str,
                    "similarity_score": float,   # cosine similarity, 0–1
                    "chunk_index":      int,
                },
                ...
            ]
        """
        if not query or not query.strip():
            logger.warning("BaselineRAG.retrieve called with empty query")
            return []

        logger.info(f"[Baseline] Retrieving top-{self.top_k} for: '{query[:80]}...'")

        # Embed the raw query — no rewriting
        query_embedding = self.embedder.generate_single_embedding(query)

        # Search — cosine similarity only, no reranker
        raw_results = self.vector_store.search(
            user_id=user_id,
            query_embedding=query_embedding,
            top_k=self.top_k,
            file_ids=file_ids or [],
        )

        # Normalise into a clean, flat structure
        results = []
        for idx, r in enumerate(raw_results):
            meta = r.get("metadata", {})
            results.append(
                {
                    "rank":             idx + 1,
                    "chunk_text":       r.get("chunk_text", ""),
                    "filename":         meta.get("filename", "unknown"),
                    "document_id":      meta.get("file_id", ""),
                    "similarity_score": round(r.get("similarity_score", 0.0), 4),
                    "chunk_index":      meta.get("chunk_index", 0),
                }
            )

        logger.info(
            f"[Baseline] Retrieved {len(results)} chunks. "
            f"Top score: {results[0]['similarity_score'] if results else 'n/a'}"
        )

        return results

    def retrieve_with_scores(
        self,
        query: str,
        user_id: str,
        file_ids: Optional[list[str]] = None,
    ) -> dict:
        """
        Same as retrieve() but returns a richer dict for eval use.
        Includes the raw query, scores list, and a summary — makes it
        easy to log and diff against the Mabel pipeline in eval_retrieval.py.

        Returns:
            {
                "query":          str,             # the raw query as given
                "results":        list[dict],      # same as retrieve()
                "top_score":      float,
                "mean_score":     float,
                "result_count":   int,
            }
        """
        results = self.retrieve(query, user_id, file_ids)
        scores  = [r["similarity_score"] for r in results]

        return {
            "query":        query,
            "results":      results,
            "top_score":    round(scores[0], 4)                        if scores else 0.0,
            "mean_score":   round(sum(scores) / len(scores), 4)        if scores else 0.0,
            "result_count": len(results),
        }