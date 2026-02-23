# app/api/chat.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from app.services.embedding_service import get_embedding_service
from app.db.vector_store import get_vector_store
from app.services.utils import clean_text
from app.services.llm_utils import (
    rewrite_query,
    classify_query,
    generate_response,
    generate_conversational_response,
    generate_general_response,
)
from app.services.reranker import get_reranker
from app.config import settings
from app.api.auth import verify_token

import logging
import uuid

logger = logging.getLogger(__name__)
chat_router = APIRouter()


class ChatRequest(BaseModel):
    query: str = Field(description="Original user query")
    links: Optional[List[str]] = Field(
        default=None, description="Links provided by user"
    )
    conversation_id: Optional[str] = None
    active_document_ids: Optional[List[str]] = Field(
        default=None,
        description="List of document IDs to search. If empty, searches all user documents.",
    )


class ChatResponse(BaseModel):
    response: str
    rewritten_query: Optional[str] = None
    conversation_id: str
    sources: Optional[List[Dict[str, Any]]] = None


@chat_router.post("/")
async def chat(request: ChatRequest, user_id: str = Depends(verify_token)):
    """
    Main chat endpoint with intelligent routing and conversation memory.

    Flow:
    1. Classify query type
    2. Retrieve conversation history for context
    3. Conversational -> respond directly
    4. Otherwise -> search vector store
    5. Good similarity -> RAG response with history
    6. Weak/no results -> general knowledge with history
    7. Save turn to conversation history
    """
    try:
        if not request.query.strip():
            raise HTTPException(400, "Query cannot be empty")

        logger.info(f"Chat request from user {user_id}: {request.query[:100]}...")

        conversation_id = request.conversation_id or str(uuid.uuid4())
        cleaned_query = clean_text(request.query)
        vector_store = get_vector_store()

        #   Classify query
        query_type = classify_query(cleaned_query)
        logger.info(f"Query type: {query_type}")

        #  Retrieve conversation history
        history = vector_store.get_conversation_history(
            user_id=user_id,
            conversation_id=conversation_id,
            last_n=6,  # last 3 exchanges
        )
        logger.info(
            f"Retrieved {len(history)} previous turns for conversation {conversation_id}"
        )

        # Step 3: Conversational, we skip RAG
        if query_type == "conversational":
            response_text = generate_conversational_response(
                cleaned_query, history=history
            )

            # save turn
            turn_index = vector_store.get_conversation_turn_count(
                user_id, conversation_id
            )
            vector_store.save_conversation_turn(
                user_id=user_id,
                conversation_id=conversation_id,
                user_message=request.query,
                assistant_message=response_text,
                turn_index=turn_index,
            )

            return ChatResponse(
                response=response_text,
                conversation_id=conversation_id,
                sources=[],
            )

        #  Step 4: Rewrite and search 
        rewritten_query = rewrite_query(cleaned_query)
        logger.info(f"Rewritten query: {rewritten_query}")

        embedding_service = get_embedding_service()
        query_embedding = embedding_service.generate_single_embedding(rewritten_query)

        search_results = vector_store.search(
            user_id=user_id,
            query_embedding=query_embedding,
            top_k=settings.INITIAL_RETRIEVAL_K,
            file_ids=request.active_document_ids or [],
        )

        #  If no results, we use general knowledge
        if not search_results or len(search_results) == 0:
            logger.info(f"No documents found, falling back to general knowledge")
            response_text = generate_general_response(request.query, history=history)

            turn_index = vector_store.get_conversation_turn_count(
                user_id, conversation_id
            )
            vector_store.save_conversation_turn(
                user_id=user_id,
                conversation_id=conversation_id,
                user_message=request.query,
                assistant_message=response_text,
                turn_index=turn_index,
            )

            return ChatResponse(
                response=response_text,
                rewritten_query=(
                    rewritten_query if rewritten_query != request.query else None
                ),
                conversation_id=conversation_id,
                sources=[],
            )

        # We Check similarity threshold 
        top_score = search_results[0].get("similarity_score", 0.0)
        logger.info(
            f"Top similarity score: {top_score:.3f} (threshold: {settings.SIMILARITY_THRESHOLD})"
        )

        if top_score < settings.SIMILARITY_THRESHOLD:
            logger.info("Similarity too low, falling back to general knowledge")
            response_text = generate_general_response(request.query, history=history)

            turn_index = vector_store.get_conversation_turn_count(
                user_id, conversation_id
            )
            vector_store.save_conversation_turn(
                user_id=user_id,
                conversation_id=conversation_id,
                user_message=request.query,
                assistant_message=response_text,
                turn_index=turn_index,
            )

            return ChatResponse(
                response=response_text,
                rewritten_query=(
                    rewritten_query if rewritten_query != request.query else None
                ),
                conversation_id=conversation_id,
                sources=[],
            )

        # RAG pipeline 
        reranker = get_reranker()
        reranked_results = reranker.rerank(
            query=rewritten_query, results=search_results, top_k=settings.FINAL_TOP_K
        )

        logger.info(f"Using top {len(reranked_results)} chunks after reranking")

        sources = []
        temp_list = []

        for idx, result in enumerate(reranked_results):
            chunk_text = result.get("chunk_text", "")

            sources.append(
                {
                    "rank": idx + 1,
                    "text": (
                        chunk_text[:300] + "..."
                        if len(chunk_text) > 300
                        else chunk_text
                    ),
                    "filename": result.get("metadata", {}).get("filename", "unknown"),
                    "document_id": result.get("metadata", {}).get("file_id", ""),
                    "similarity_score": result.get("similarity_score", 0.0),
                    "rerank_score": result.get("rerank_score", 0.0),
                    "chunk_index": result.get("metadata", {}).get("chunk_index", 0),
                }
            )

            formatted_string = f"[Source {idx+1}: {result.get('metadata', {}).get('filename', 'Unknown')}]\n{chunk_text}"
            temp_list.append(formatted_string)

        context = "\n\n---\n\n".join(temp_list)
        response_text = generate_response(
            query=request.query,
            context=context,
            history=history,
        )

        # Step 8: Save turn abi chat abi session 
        turn_index = vector_store.get_conversation_turn_count(user_id, conversation_id)
        vector_store.save_conversation_turn(
            user_id=user_id,
            conversation_id=conversation_id,
            user_message=request.query,
            assistant_message=response_text,
            turn_index=turn_index,
        )

        logger.info(
            f"Chat completed for conversation {conversation_id}, turn {turn_index}"
        )

        return ChatResponse(
            response=response_text,
            rewritten_query=(
                rewritten_query if rewritten_query != request.query else None
            ),
            conversation_id=conversation_id,
            sources=sources,
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(500, f"Error processing chat request: {str(e)}")
