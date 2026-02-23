from sentence_transformers import SentenceTransformer
from typing import List
import logging
import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Service for generating text embeddings using Sentence Transformers
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model

        Args: 
            model_name: Name of the sentence transformer model to use
                Options:
                - 'all-MiniLM-L6-v2': Fast, 384 dimensions (default)
                - 'all-mpnet-base-v2': Better quality, 768 dimensions
                - 'multi-qa-mpnet-base-dot-v1': Optimized for Q&A
        """
        try:
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def generate_embeddings(
        self, 
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once
            normalize: Whether to normalize embeddings (recommended for cosine similarity)
      
        Returns:
            List of embedding vectors
        """
        if not texts:
            logger.warning("Empty text list provided for embedding generation")
            return []

        # Filter out empty strings
        valid_texts = [text for text in texts if text and text.strip()]
        if len(valid_texts) != len(texts):
            logger.warning(f"Filtered out {len(texts) - len(valid_texts)} empty texts")

        if not valid_texts:
            return []

        logger.info(f"Generating embeddings for {len(valid_texts)} texts")
        
        try:
            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                show_progress_bar=len(valid_texts) > 10,
                convert_to_numpy=True,
                normalize_embeddings=normalize
            )
            
            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings.tolist()
        
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def generate_single_embedding(
        self, 
        text: str,
        normalize: bool = True
    ) -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: Text string to embed
            normalize: Whether to normalize embedding
      
        Returns:
            Embedding vector  
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for single embedding generation")
            return [0.0] * self.embedding_dim
        
        try:
            embedding = self.model.encode(
                [text],
                normalize_embeddings=normalize
            )[0]
            return embedding.tolist()
        
        except Exception as e:
            logger.error(f"Error generating single embedding: {e}")
            raise

    def compute_similarity(
        self, 
        embedding1: List[float], 
        embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
      
        Returns:
            Similarity score between -1 and 1 (higher = more similar)
        """
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return float(similarity)
        
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            raise

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model
      
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "max_sequence_length": self.model.max_seq_length
        }


# Singleton instance
_embedding_service = None

def get_embedding_service(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingService:
    """
    Get or create the singleton embedding service instance
    
    Args:
        model_name: Model to use (only used on first call)
    
    Returns:
        EmbeddingService instance
    """
    global _embedding_service
    
    if _embedding_service is None:
        _embedding_service = EmbeddingService(model_name)
    
    return _embedding_service