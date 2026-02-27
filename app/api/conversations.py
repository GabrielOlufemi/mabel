# app/api/conversations.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import logging
import json
import time

from app.db.vector_store import get_vector_store
from app.api.auth import verify_token

logger = logging.getLogger(__name__)
conversations_router = APIRouter()


class ConversationSummary(BaseModel):
    conversation_id: str
    title: str
    last_message: str
    turn_count: int
    last_updated: int


class ConversationsListResponse(BaseModel):
    conversations: List[ConversationSummary]


class ConversationTurn(BaseModel):
    user: str
    assistant: str
    turn_index: int
    sources: Optional[List[Dict[str, Any]]] = None  # ← added


class ConversationDetailResponse(BaseModel):
    conversation_id: str
    turns: List[ConversationTurn]


@conversations_router.get("/", response_model=ConversationsListResponse)
async def list_conversations(user_id: str = Depends(verify_token)):
    """List all conversation sessions for the authenticated user, newest first."""
    try:
        vector_store = get_vector_store()
        collection = vector_store.get_conversation_collection(user_id)
        results = collection.get(include=["documents", "metadatas"])

        if not results["ids"]:
            return ConversationsListResponse(conversations=[])

        conv_map: Dict[str, dict] = {}

        for i, doc in enumerate(results["documents"]):
            meta = results["metadatas"][i]
            cid = meta.get("conversation_id")
            if not cid:
                continue

            turn_data = json.loads(doc)
            turn_index = meta.get("turn_index", 0)
            timestamp = meta.get("timestamp", 0)

            if cid not in conv_map:
                conv_map[cid] = {"conversation_id": cid, "turns": [], "last_updated": 0}

            conv_map[cid]["turns"].append(
                {
                    "turn_index": turn_index,
                    "user": turn_data.get("user", ""),
                    "assistant": turn_data.get("assistant", ""),
                    "timestamp": timestamp,
                }
            )

            if timestamp > conv_map[cid]["last_updated"]:
                conv_map[cid]["last_updated"] = timestamp

        summaries = []
        for cid, data in conv_map.items():
            turns = sorted(data["turns"], key=lambda t: t["turn_index"])
            first_user_msg = turns[0]["user"] if turns else "Untitled session"
            last_user_msg = turns[-1]["user"] if turns else ""
            title = first_user_msg[:60] + ("…" if len(first_user_msg) > 60 else "")
            snippet = last_user_msg[:80] + ("…" if len(last_user_msg) > 80 else "")

            summaries.append(
                ConversationSummary(
                    conversation_id=cid,
                    title=title,
                    last_message=snippet,
                    turn_count=len(turns),
                    last_updated=data["last_updated"],
                )
            )

        summaries.sort(key=lambda s: s.last_updated, reverse=True)
        return ConversationsListResponse(conversations=summaries)

    except Exception as e:
        logger.error(f"Error listing conversations: {e}", exc_info=True)
        raise HTTPException(500, f"Error listing conversations: {str(e)}")


@conversations_router.get(
    "/{conversation_id}", response_model=ConversationDetailResponse
)
async def get_conversation(conversation_id: str, user_id: str = Depends(verify_token)):
    """
    Retrieve all turns for a specific conversation, including sources for each turn,
    so the frontend can restore the full chat session with source cards intact.
    """
    try:
        vector_store = get_vector_store()
        turns = vector_store.get_conversation_history(
            user_id=user_id,
            conversation_id=conversation_id,
            last_n=9999,
        )

        if not turns:
            raise HTTPException(404, "Conversation not found")

        return ConversationDetailResponse(
            conversation_id=conversation_id,
            turns=[
                ConversationTurn(
                    user=t["user"],
                    assistant=t["assistant"],
                    turn_index=t["turn_index"],
                    sources=t.get("sources") or None,  # ← restored from stored metadata
                )
                for t in turns
            ],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error fetching conversation {conversation_id}: {e}", exc_info=True
        )
        raise HTTPException(500, f"Error fetching conversation: {str(e)}")


@conversations_router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str, user_id: str = Depends(verify_token)
):
    """Permanently delete all turns for a conversation."""
    try:
        vector_store = get_vector_store()
        success = vector_store.delete_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if not success:
            raise HTTPException(500, "Failed to delete conversation")
        return {"deleted": True, "conversation_id": conversation_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error deleting conversation {conversation_id}: {e}", exc_info=True
        )
        raise HTTPException(500, f"Error deleting conversation: {str(e)}")
