# app/services/llm_utils.py
from google import genai
from google.genai import errors as genai_errors
from app.config import settings
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

# ── Client init — fail loudly if key is missing ───────────────────
if not settings.GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set. Check your .env file.")

client = genai.Client(api_key=settings.GEMINI_API_KEY)
logger.info(f"Gemini client initialized. Model: {settings.GEMINI_MODEL_NAME}")


def _call_gemini(
    system_instruction: str, contents: str, temperature: float = 0.4
) -> str:
    """
    Central Gemini call — raises on failure so callers can handle it explicitly.
    All API calls go through here so error logging is consistent.
    """
    response = client.models.generate_content(
        model=settings.GEMINI_MODEL_NAME,
        contents=contents,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        ),
    )

    if hasattr(response, "usage_metadata") and response.usage_metadata:
        usage = response.usage_metadata
        logger.debug(
            f"Token usage — in: {usage.prompt_token_count}, "
            f"out: {usage.candidates_token_count}, "
            f"total: {usage.total_token_count}"
        )

    return response.text.strip()


def rewrite_query(query: str) -> str:
    """Rewrite user query to be more explicit and optimized for retrieval."""
    if not query or not query.strip():
        return query

    try:
        prompt = (
            "Rewrite this search query to be more explicit, clear, and optimized for document retrieval. "
            "Expand abbreviations, clarify vague terms, and make the intent explicit. "
            "Return ONLY the rewritten query, nothing else. Do not use emojis."
        )
        rewritten = _call_gemini(prompt, query, temperature=0.2)
        logger.info(f"Query rewritten: '{query}' -> '{rewritten}'")
        return rewritten

    except Exception as e:
        logger.error(
            f"rewrite_query failed — falling back to original. Error: {e}",
            exc_info=True,
        )
        return query


def classify_query(query: str) -> str:
    """
    Classify the user query into one of:
    - 'conversational': greetings, thanks, casual chat
    - 'document_question': questions about uploaded study material
    - 'general_knowledge': factual questions not tied to documents
    """
    if not query or not query.strip():
        return "conversational"

    try:
        prompt = (
            "Classify the following user message into exactly one of these three categories:\n\n"
            "1. conversational — greetings, thanks, casual chat, questions about what you can do\n"
            "2. document_question — questions likely about uploaded study material or documents\n"
            "3. general_knowledge — factual or conceptual questions not tied to any specific document\n\n"
            "Return ONLY one of these exact strings: conversational, document_question, general_knowledge"
        )
        result = _call_gemini(prompt, query, temperature=0.0).lower()

        if result not in ("conversational", "document_question", "general_knowledge"):
            logger.warning(
                f"Unexpected classification '{result}', defaulting to document_question"
            )
            return "document_question"

        logger.info(f"Query classified as: {result}")
        return result

    except Exception as e:
        logger.error(
            f"classify_query failed — defaulting to document_question. Error: {e}",
            exc_info=True,
        )
        return "document_question"


def generate_conversational_response(query: str, history: List[Dict] = None) -> str:
    """Generate a casual conversational response with optional history context."""
    try:
        prompt = (
            f"You are {settings.SERVICE_NAME}, a friendly and helpful AI study assistant. "
            "You help students understand their documents, generate flashcards, quizzes, and summaries. "
            "Respond naturally and warmly to the user's message. "
            "If they ask what you can do, briefly explain your study features. "
            "Keep responses concise and friendly. Do not use emojis."
        )
        history_text = _format_history(history)
        contents = f"{history_text}\nUser: {query}" if history_text else query

        answer = _call_gemini(prompt, contents, temperature=0.7)
        logger.info("Conversational response generated")
        return answer

    except Exception as e:
        logger.error(
            f"generate_conversational_response failed. Error: {e}", exc_info=True
        )
        return "Hey! I'm MABEL, your study assistant. Upload a document and ask me anything about it!"


def generate_general_response(query: str, history: List[Dict] = None) -> str:
    """Generate a general knowledge response when no relevant docs are found."""
    try:
        prompt = (
            f"You are {settings.SERVICE_NAME}, a helpful AI study assistant. "
            "Answer the user's question from your general knowledge. "
            "Be accurate, clear, and concise. Do not use emojis. "
            "At the end of your response, add a short note on a new line: "
            "'Note: I answered from general knowledge - upload a document for answers specific to your study material.'"
        )
        history_text = _format_history(history)
        contents = f"{history_text}\nUser: {query}" if history_text else query

        answer = _call_gemini(prompt, contents, temperature=settings.GEMINI_TEMPERATURE)
        logger.info("General knowledge response generated")
        return answer

    except Exception as e:
        logger.error(f"generate_general_response failed. Error: {e}", exc_info=True)
        raise  # ← re-raise so chat.py returns a proper 500 instead of a fake success


def generate_response(query: str, context: str, history: List[Dict] = None) -> str:
    """
    Generate a RAG response using retrieved document context and conversation history.
    """
    if not query or not query.strip():
        return "I couldn't understand your question. Please try again."
    if not context or not context.strip():
        return "I couldn't find any relevant information to answer your question."

    try:
        prompt = (
            f"You are {settings.SERVICE_NAME}, a helpful study assistant. "
            "Answer the user's question based ONLY on the provided context from their documents. "
            "If the context doesn't contain enough information, say so clearly. "
            "Be concise, accurate, and helpful. Do not use emojis. "
            "Cite sources by mentioning [Source 1], [Source 2], etc."
        )
        history_text = _format_history(history)

        if history_text:
            message = (
                f"Previous conversation:\n{history_text}\n\n"
                f"Context from user documents:\n{context}\n\n----\n\n"
                f"User question: {query}\n\n"
                "Please answer the question based on the context provided."
            )
        else:
            message = (
                f"Context from user documents:\n{context}\n\n----\n\n"
                f"User question: {query}\n\n"
                "Please answer the question based on the context provided."
            )

        answer = _call_gemini(prompt, message, temperature=settings.GEMINI_TEMPERATURE)
        logger.info("RAG response generated successfully")
        return answer

    except Exception as e:
        logger.error(f"generate_response failed. Error: {e}", exc_info=True)
        raise  # ← re-raise so chat.py returns a proper 500 instead of a fake success


def _format_history(history: List[Dict]) -> str:
    """Format conversation history into a readable string for the LLM prompt."""
    if not history:
        return ""
    lines = []
    for turn in history:
        lines.append(f"User: {turn.get('user', '')}")
        lines.append(f"Assistant: {turn.get('assistant', '')}")
    return "\n".join(lines)
