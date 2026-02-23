from sentence_transformers import CrossEncoder
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class Reranker:
    """
    Rerank search results using a cross-encoder model
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the reranker
        
        Args:
            model_name: Cross-encoder model to use
                - 'cross-encoder/ms-marco-MiniLM-L-6-v2': Fast, good quality
                - 'cross-encoder/ms-marco-TinyBERT-L-2-v2': Faster, decent quality
                - 'cross-encoder/ms-marco-MiniLM-L-12-v2': Slower, better quality
        """
        try:
            logger.info(f"Loading reranker model: {model_name}")
            self.model = CrossEncoder(model_name)
            self.model_name = model_name
            logger.info(f"Reranker loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load reranker: {e}")
            raise
    
    def rerank(
        self, 
        query: str, 
        results: List[Dict], 
        top_k: int = 5
    ) -> List[Dict]:
        """
        Rerank search results based on relevance to query
        
        Args:
            query: User's query
            results: List of search results with 'chunk_text' field
            top_k: Number of top results to return
        
        Returns:
            Reranked list of results (top_k best)
        """
        if not results:
            logger.warning("No results to rerank")
            return []
        
        try:
            # Prepare query-document pairs for the cross-encoder
            pairs = []
            for result in results:
                chunk_text = result.get("chunk_text", "")
                pair = [query, chunk_text]
                pairs.append(pair)
            
            logger.info(f"Reranking {len(results)} results")
            
            # Get relevance scores from cross-encoder
            scores = self.model.predict(pairs)
            
            # Combine results with their rerank scores
            for result, score in zip(results, scores):
                result["rerank_score"] = float(score)
            
            # Sort by rerank score (descending)
            reranked = sorted(results, key = lambda x: x["rerank_score"], reverse=True)   

            # Take top_k
            top_results = reranked[:top_k]
            
            logger.info(
                f"Reranking complete. Top score: {top_results[0]['rerank_score']:.3f}, "
                f"Bottom score: {top_results[-1]['rerank_score']:.3f}"
            )
            
            return top_results
        
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Fallback: return original results if reranking fails
            return results[:top_k]


# Singleton instance
_reranker = None

def get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> Reranker:
    """
    Get or create the singleton reranker instance
    
    Args:
        model_name: Model to use (only used on first call)
    
    Returns:
        Reranker instance
    """
    global _reranker
    
    if _reranker is None:
        _reranker = Reranker(model_name)
    
    return _reranker