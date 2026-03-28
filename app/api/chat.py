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
import json

logger = logging.getLogger(__name__)
chat_router = APIRouter()


class ChatRequest(BaseModel):
    query: str = Field(description="Original user query")
    links: Optional[List[str]] = Field(default=None)
    conversation_id: Optional[str] = None
    active_document_ids: Optional[List[str]] = Field(default=None)


class ChatResponse(BaseModel):
    response: str
    rewritten_query: Optional[str] = None
    conversation_id: str
    sources: Optional[List[Dict[str, Any]]] = None


def _build_sources(reranked_results: list) -> tuple[list, str]:
    """
    Build the sources list and context string from reranked results.
    Sources are already in rerank order — we drop similarity_score since
    rerank_score is the authoritative ranking signal after reranking.
    """
    sources = []
    temp_list = []

    for idx, result in enumerate(reranked_results):
        chunk_text = result.get("chunk_text", "")
        meta = result.get("metadata", {})

        sources.append(
            {
                "rank": idx + 1,
                "text": (
                    chunk_text[:300] + "..." if len(chunk_text) > 300 else chunk_text
                ),
                "filename": meta.get("filename", "unknown"),
                "document_id": meta.get("file_id", ""),
                "rerank_score": result.get("rerank_score", 0.0),
                "chunk_index": meta.get("chunk_index", 0),
            }
        )

        temp_list.append(
            f"[Source {idx+1}: {meta.get('filename', 'Unknown')}]\n{chunk_text}"
        )

    context = "\n\n---\n\n".join(temp_list)
    return sources, context


@chat_router.post("/")
async def chat(request: ChatRequest, user_id: str = Depends(verify_token)):
    """
    Main chat endpoint with intelligent routing and conversation memory.

    Flow:
    1. Classify query type
    2. Retrieve conversation history for context
    3. Conversational → respond directly, else search vector store
    4. Good similarity → RAG response with history, else general knowledge
    5. Save turn to conversation history (with sources_json for RAG turns)
    """
    try:
        if not request.query.strip():
            raise HTTPException(400, "Query cannot be empty")

        logger.info(f"Chat request from user {user_id}: {request.query[:100]}...")

        conversation_id = request.conversation_id or str(uuid.uuid4())
        cleaned_query = clean_text(request.query)
        vector_store = get_vector_store()

        # 1. Classify
        query_type = classify_query(cleaned_query)
        logger.info(f"Query type: {query_type}")

        # 2. Retrieve history
        history = vector_store.get_conversation_history(
            user_id=user_id,
            conversation_id=conversation_id,
            last_n=6,
        )
        logger.info(
            f"Retrieved {len(history)} previous turns for conversation {conversation_id}"
        )

        # 3. Conversational shortcut — ONLY when no documents are active.
        #    If the user has active docs, even a "conversational"-looking query
        #    (e.g. "tell me about this", "explain that part") should still hit
        #    vector search so we don't miss document-scoped questions the
        #    classifier misjudged as casual chat.
        has_active_docs = bool(request.active_document_ids)

        if query_type == "conversational" and not has_active_docs:
            response_text = generate_conversational_response(
                cleaned_query, history=history
            )
            turn_index = vector_store.get_conversation_turn_count(
                user_id, conversation_id
            )
            vector_store.save_conversation_turn(
                user_id=user_id,
                conversation_id=conversation_id,
                user_message=request.query,
                assistant_message=response_text,
                turn_index=turn_index,
                sources=None,  # no sources for conversational turns
                active_document_ids=request.active_document_ids,
            )
            return ChatResponse(
                response=response_text, conversation_id=conversation_id, sources=[]
            )

        if query_type == "conversational" and has_active_docs:
            logger.info(
                "Query classified as conversational but user has active docs — "
                "proceeding to vector search as a safety net"
            )

        # 4. Rewrite + embed + search
        #    Only search if the user has actively selected documents.
        #    When active_document_ids is None/empty the user hasn't scoped to
        #    any document, so there's nothing to RAG against — go straight to
        #    general knowledge instead of searching the entire collection and
        #    pulling back irrelevant chunks from old uploads.
        rewritten_query = rewrite_query(cleaned_query)
        logger.info(f"Rewritten query: {rewritten_query}")

        if not has_active_docs:
            logger.info("No active documents selected — using general knowledge")
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
                sources=None,
                active_document_ids=request.active_document_ids,
            )
            return ChatResponse(
                response=response_text,
                rewritten_query=(
                    rewritten_query if rewritten_query != request.query else None
                ),
                conversation_id=conversation_id,
                sources=[],
            )

        embedding_service = get_embedding_service()
        query_embedding = embedding_service.generate_single_embedding(rewritten_query)

        search_results = vector_store.search(
            user_id=user_id,
            query_embedding=query_embedding,
            top_k=settings.INITIAL_RETRIEVAL_K,
            file_ids=request.active_document_ids,
        )

        # No results -> general knowledge fallback
        if not search_results:
            logger.info("No documents found, falling back to general knowledge")
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
                sources=None,
                active_document_ids=request.active_document_ids,
            )
            return ChatResponse(
                response=response_text,
                rewritten_query=(
                    rewritten_query if rewritten_query != request.query else None
                ),
                conversation_id=conversation_id,
                sources=[],
            )

        # Similarity threshold check — only apply when no specific documents
        # are selected (i.e. a future scenario where we re-enable broad search).
        # When the user has actively pinned documents, they've declared intent:
        # "answer from THESE docs."  A low embedding similarity just means the
        # query phrasing doesn't overlap with chunk wording, not that the doc
        # is irrelevant.  The reranker (cross-encoder) is a much better
        # semantic judge and will filter genuinely useless chunks below.
        top_score = search_results[0].get("similarity_score", 0.0)
        logger.info(
            f"Top similarity score: {top_score:.3f} (threshold: {settings.SIMILARITY_THRESHOLD})"
        )

        if not has_active_docs and top_score < settings.SIMILARITY_THRESHOLD:
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
                sources=None,
                active_document_ids=request.active_document_ids,
            )
            return ChatResponse(
                response=response_text,
                rewritten_query=(
                    rewritten_query if rewritten_query != request.query else None
                ),
                conversation_id=conversation_id,
                sources=[],
            )

        # 5. Rerank -> filter by rerank score -> RAG
        reranker = get_reranker()
        reranked_results = reranker.rerank(
            query=rewritten_query,
            results=search_results,
            top_k=settings.FINAL_TOP_K,
        )

        # Filter out chunks the reranker considers irrelevant.
        # ms-marco cross-encoders produce logit scores where negative values
        # mean the passage is unlikely relevant.  A threshold of -5.0 is
        # generous — in the screenshot, irrelevant chunks scored -7 to -8.
        rerank_threshold = getattr(settings, "RERANK_SCORE_THRESHOLD", -5.0)
        reranked_results = [
            r for r in reranked_results
            if r.get("rerank_score", -999) >= rerank_threshold
        ]

        if not reranked_results:
            logger.info(
                "All chunks filtered out by rerank threshold — falling back to general knowledge"
            )
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
                sources=None,
                active_document_ids=request.active_document_ids,
            )
            return ChatResponse(
                response=response_text,
                rewritten_query=(
                    rewritten_query if rewritten_query != request.query else None
                ),
                conversation_id=conversation_id,
                sources=[],
            )

        logger.info(f"Using top {len(reranked_results)} chunks after reranking")

        sources, context = _build_sources(reranked_results)

        response_text = generate_response(
            query=request.query,
            context=context,
            history=history,
        )

        # 6. Save turn — include sources so they can be restored later
        turn_index = vector_store.get_conversation_turn_count(user_id, conversation_id)
        vector_store.save_conversation_turn(
            user_id=user_id,
            conversation_id=conversation_id,
            user_message=request.query,
            assistant_message=response_text,
            turn_index=turn_index,
            sources=sources,  # <- persisted
            active_document_ids=request.active_document_ids,
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