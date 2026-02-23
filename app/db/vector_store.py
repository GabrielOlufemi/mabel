# app/db/vector_store.py
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import logging
import json
import time
from pathlib import Path

from app.config import settings
from app.services.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Service for managing vector storage using ChromaDB with multi-user support.
    Handles both document chunks and conversation history.
    """

    def __init__(self, persist_directory: str = None):
        if persist_directory is None:
            persist_directory = settings.VECTOR_STORE_PATH

        try:
            logger.info(f"Initializing ChromaDB with persist directory: {persist_directory}")
            Path(persist_directory).mkdir(parents=True, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )

            embedding_service = get_embedding_service()
            self.embedding_dim = embedding_service.embedding_dim

            logger.info(f"ChromaDB initialized. Embedding dimension: {self.embedding_dim}")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    # ── Document collections ──────────────────────────────────────────────────

    def get_user_collection(self, user_id: str):
        """Get or create a document collection for a specific user"""
        collection_name = f"user_{user_id}_documents"
        try:
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"user_id": user_id, "embedding_dimension": self.embedding_dim},
            )
            return collection
        except Exception as e:
            logger.error(f"Error getting collection for user {user_id}: {e}")
            raise

    def add_document(
        self,
        user_id: str,
        file_id: str,
        filename: str,
        chunks: List[str],
        embeddings: List[List[float]],
        additional_metadata: Optional[Dict] = None,
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        if not chunks:
            return 0

        logger.info(f"Adding {len(chunks)} chunks for user {user_id}, file {file_id}")

        try:
            collection = self.get_user_collection(user_id)
            ids = [f"{file_id}_chunk_{idx}" for idx in range(len(chunks))]
            metadatas = []

            for idx, chunk in enumerate(chunks):
                metadata = {
                    "file_id": file_id,
                    "filename": filename,
                    "chunk_index": idx,
                    "text_length": len(chunk),
                    "user_id": user_id,
                }
                if additional_metadata:
                    metadata.update(additional_metadata)
                metadatas.append(metadata)

            collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
            logger.info(f"Successfully added {len(chunks)} chunks for file {file_id}")
            return len(chunks)

        except Exception as e:
            logger.error(f"Error adding document chunks: {e}")
            raise

    def search(
        self,
        user_id: str,
        query_embedding: List[float],
        top_k: int = 5,
        file_id: Optional[str] = None,
        file_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Search for similar chunks.
        file_ids: list of active document IDs to filter by.
                  Pass empty list or None to search all documents.
        """
        logger.info(f"Searching for top {top_k} chunks for user {user_id}")
        try:
            collection = self.get_user_collection(user_id)

            # Build where filter
            if file_ids and len(file_ids) == 1:
                where_filter = {"file_id": file_ids[0]}
            elif file_ids and len(file_ids) > 1:
                where_filter = {"file_id": {"$in": file_ids}}
            elif file_id:
                where_filter = {"file_id": file_id}
            else:
                where_filter = None

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            formatted_results = []
            if results["ids"] and results["ids"][0]:
                for idx in range(len(results["ids"][0])):
                    formatted_results.append({
                        "id": results["ids"][0][idx],
                        "chunk_text": results["documents"][0][idx],
                        "metadata": results["metadatas"][0][idx],
                        "distance": results["distances"][0][idx],
                        "similarity_score": 1 - (results["distances"][0][idx] / 2),
                    })

            logger.info(f"Found {len(formatted_results)} matches")
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching: {e}")
            raise

    def delete_document(self, user_id: str, file_id: str) -> bool:
        try:
            collection = self.get_user_collection(user_id)
            collection.delete(where={"file_id": file_id})
            logger.info(f"Successfully deleted document {file_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False

    def delete_user_collection(self, user_id: str) -> bool:
        collection_name = f"user_{user_id}_documents"
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Successfully deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False

    def get_user_stats(self, user_id: str) -> Dict:
        try:
            collection = self.get_user_collection(user_id)
            count = collection.count()
            return {
                "user_id": user_id,
                "total_chunks": count,
                "collection_name": f"user_{user_id}_documents",
            }
        except Exception as e:
            logger.error(f"Error getting user stats: {e}")
            return {"user_id": user_id, "total_chunks": 0, "error": str(e)}

    def list_user_documents(self, user_id: str) -> List[Dict]:
        try:
            collection = self.get_user_collection(user_id)
            results = collection.get(include=["metadatas"])

            file_ids = set()
            documents = {}

            for metadata in results["metadatas"]:
                file_id = metadata.get("file_id")
                if file_id and file_id not in file_ids:
                    file_ids.add(file_id)
                    documents[file_id] = {
                        "file_id": file_id,
                        "filename": metadata.get("filename"),
                        "user_id": user_id,
                    }

            return list(documents.values())

        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []

    def get_all_collections(self) -> List[str]:
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []

    # ── Conversation history ──────────────────────────────────────────────────

    def get_conversation_collection(self, user_id: str):
        """Get or create a conversation history collection for a user"""
        collection_name = f"user_{user_id}_conversations"
        try:
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"user_id": user_id, "type": "conversations"},
            )
            return collection
        except Exception as e:
            logger.error(f"Error getting conversation collection for user {user_id}: {e}")
            raise

    def save_conversation_turn(
        self,
        user_id: str,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
        turn_index: int,
    ) -> bool:
        """
        Save a single conversation turn (user + assistant message pair).

        Args:
            user_id: User identifier
            conversation_id: Unique conversation session ID
            user_message: What the user said
            assistant_message: What the assistant responded
            turn_index: Turn number in the conversation (0, 1, 2...)

        Returns:
            True if saved successfully
        """
        try:
            collection = self.get_conversation_collection(user_id)

            turn_id = f"{conversation_id}_turn_{turn_index}"

            # Store both messages as a JSON blob in the document field
            turn_data = json.dumps({
                "user": user_message,
                "assistant": assistant_message,
            })

            metadata = {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "turn_index": turn_index,
                "timestamp": int(time.time()),
            }

            collection.add(
                ids=[turn_id],
                documents=[turn_data],
                metadatas=[metadata],
            )

            logger.info(f"Saved turn {turn_index} for conversation {conversation_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving conversation turn: {e}")
            return False

    def get_conversation_history(
        self,
        user_id: str,
        conversation_id: str,
        last_n: int = 6,
    ) -> List[Dict]:
        """
        Retrieve the last N turns of a conversation.

        Args:
            user_id: User identifier
            conversation_id: Conversation session ID
            last_n: Number of recent turns to retrieve (default 6 = 3 back-and-forth exchanges)

        Returns:
            List of turn dicts sorted oldest to newest:
            [{"user": "...", "assistant": "...", "turn_index": 0}, ...]
        """
        try:
            collection = self.get_conversation_collection(user_id)

            results = collection.get(
                where={"conversation_id": conversation_id},
                include=["documents", "metadatas"],
            )

            if not results["ids"]:
                return []

            turns = []
            for i in range(len(results["ids"])):
                turn_data = json.loads(results["documents"][i])
                turn_data["turn_index"] = results["metadatas"][i]["turn_index"]
                turns.append(turn_data)

            # sort by turn_index oldest first, take last N
            turns.sort(key=lambda t: t["turn_index"])
            return turns[-last_n:]

        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return []

    def get_conversation_turn_count(self, user_id: str, conversation_id: str) -> int:
        """Get the current number of turns in a conversation"""
        try:
            collection = self.get_conversation_collection(user_id)
            results = collection.get(
                where={"conversation_id": conversation_id},
                include=["metadatas"],
            )
            return len(results["ids"])
        except Exception as e:
            logger.error(f"Error getting turn count: {e}")
            return 0

    def delete_conversation(self, user_id: str, conversation_id: str) -> bool:
        """Delete all turns for a specific conversation"""
        try:
            collection = self.get_conversation_collection(user_id)
            collection.delete(where={"conversation_id": conversation_id})
            logger.info(f"Deleted conversation {conversation_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting conversation: {e}")
            return False

    def delete_user_conversations(self, user_id: str) -> bool:
        """Delete all conversations for a user"""
        collection_name = f"user_{user_id}_conversations"
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Deleted all conversations for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting user conversations: {e}")
            return False


# ── Singleton ─────────────────────────────────────────────────────────────────

_vector_store = None


def get_vector_store(persist_directory: str = None) -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(persist_directory)
    return _vector_store