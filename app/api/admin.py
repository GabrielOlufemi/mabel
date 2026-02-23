# app/api/admin.py
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict
import logging

from app.db.vector_store import get_vector_store
from app.api.auth import verify_token

logger = logging.getLogger(__name__)

admin_router = APIRouter()


# ── Collections ───────────────────────────────────────────────────────────────


@admin_router.get("/collections")
async def list_all_collections(user_id: str = Depends(verify_token)):
    """List all collections in ChromaDB"""
    try:
        vector_store = get_vector_store()
        collections = vector_store.get_all_collections()
        return {"total_collections": len(collections), "collections": collections}
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        raise HTTPException(500, str(e))


@admin_router.get("/collections/{target_user_id}/stats")
async def get_collection_stats(
    target_user_id: str, user_id: str = Depends(verify_token)
):
    """Get stats for a specific user's document collection"""
    try:
        vector_store = get_vector_store()
        stats = vector_store.get_user_stats(target_user_id)
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(500, str(e))


@admin_router.get("/collections/{target_user_id}/documents")
async def list_user_documents_admin(
    target_user_id: str, user_id: str = Depends(verify_token)
):
    """List all documents for a specific user"""
    try:
        vector_store = get_vector_store()
        documents = vector_store.list_user_documents(target_user_id)
        return {
            "user_id": target_user_id,
            "total_documents": len(documents),
            "documents": documents,
        }
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(500, str(e))


@admin_router.get("/collections/{target_user_id}/chunks")
async def view_all_chunks(
    target_user_id: str, limit: int = 50, user_id: str = Depends(verify_token)
):
    """View chunk data for a user (for debugging)"""
    try:
        vector_store = get_vector_store()
        collection = vector_store.get_user_collection(target_user_id)

        results = collection.get(limit=limit, include=["documents", "metadatas"])

        chunks = []
        if results["ids"]:
            for i in range(len(results["ids"])):
                chunks.append(
                    {
                        "id": results["ids"][i],
                        "text_preview": results["documents"][i][:200] + "...",
                        "metadata": results["metadatas"][i],
                    }
                )

        return {
            "user_id": target_user_id,
            "total_chunks": len(chunks),
            "chunks": chunks,
        }
    except Exception as e:
        logger.error(f"Error viewing chunks: {e}")
        raise HTTPException(500, str(e))


@admin_router.get("/search-test")
async def test_search(
    target_user_id: str,
    query: str,
    top_k: int = 3,
    user_id: str = Depends(verify_token),
):
    """Test search functionality"""
    try:
        from app.services.embedding_service import get_embedding_service

        vector_store = get_vector_store()
        embedding_service = get_embedding_service()

        query_embedding = embedding_service.generate_single_embedding(query)
        results = vector_store.search(
            user_id=target_user_id, query_embedding=query_embedding, top_k=top_k
        )

        return {
            "query": query,
            "user_id": target_user_id,
            "results_found": len(results),
            "results": results,
        }
    except Exception as e:
        logger.error(f"Search test error: {e}")
        raise HTTPException(500, str(e))


# ── Conversation history ──────────────────────────────────────────────────────


@admin_router.get("/conversations/{target_user_id}")
async def list_user_conversations(
    target_user_id: str, user_id: str = Depends(verify_token)
):
    """List all conversation IDs for a user"""
    try:
        vector_store = get_vector_store()
        collection = vector_store.get_conversation_collection(target_user_id)
        results = collection.get(include=["metadatas"])

        conversation_ids = set()
        for metadata in results["metadatas"]:
            conversation_ids.add(metadata.get("conversation_id"))

        return {
            "user_id": target_user_id,
            "total_conversations": len(conversation_ids),
            "conversation_ids": list(conversation_ids),
        }
    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        raise HTTPException(500, str(e))


@admin_router.get("/conversations/{target_user_id}/{conversation_id}")
async def get_conversation(
    target_user_id: str, conversation_id: str, user_id: str = Depends(verify_token)
):
    """Get full history of a specific conversation"""
    try:
        vector_store = get_vector_store()
        history = vector_store.get_conversation_history(
            user_id=target_user_id,
            conversation_id=conversation_id,
            last_n=100,  # get all turns for admin view
        )

        return {
            "user_id": target_user_id,
            "conversation_id": conversation_id,
            "total_turns": len(history),
            "history": history,
        }
    except Exception as e:
        logger.error(f"Error getting conversation: {e}")
        raise HTTPException(500, str(e))


@admin_router.delete("/conversations/{target_user_id}/{conversation_id}")
async def delete_conversation(
    target_user_id: str, conversation_id: str, user_id: str = Depends(verify_token)
):
    """Delete a specific conversation"""
    try:
        vector_store = get_vector_store()
        success = vector_store.delete_conversation(target_user_id, conversation_id)

        if success:
            return {
                "status": "success",
                "message": f"Conversation {conversation_id} deleted",
            }
        else:
            raise HTTPException(500, "Failed to delete conversation")
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(500, str(e))
