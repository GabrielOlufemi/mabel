# app/services/llm_utils.py
from google import genai
from google.genai import errors as genai_errors
from app.config import settings
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

# Client initialization, should fail loudly if key is missing
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


# query classification function
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
            "You help learners understand their documents, generate flashcards, quizzes, and summaries. "
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
        return f"Hey! I'm {settings.SERVICE_NAME}, your study assistant. Upload a document and ask me anything about it!"


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
        raise


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
        raise


def _format_history(history: List[Dict]) -> str:
    """Format conversation history into a readable string for the LLM prompt."""
    if not history:
        return ""
    lines = []
    for turn in history:
        lines.append(f"User: {turn.get('user', '')}")
        lines.append(f"Assistant: {turn.get('assistant', '')}")
    return "\n".join(lines)


def generate_flashcards(
    chunks: list[str], filename: str = "document", card_count: int = 8
) -> list[dict]:
    """
    Generate flashcards from reconstructed document chunks.

    Returns:
        List of dicts: [{"q": "...", "a": "..."}, ...]
    Raises:
        ValueError: If Gemini returns unparseable or malformed JSON
    """
    import json

    if not chunks:
        raise ValueError("No chunks provided for flashcard generation")

    document_text = "\n\n".join(chunks)

    max_chars = 80_000
    if len(document_text) > max_chars:
        logger.warning(
            f"Document text truncated from {len(document_text)} to {max_chars} chars for flashcard generation"
        )
        document_text = document_text[:max_chars]

    system_instruction = (
        "You are a study assistant that generates high-quality flashcards to help learners learn. "
        "Your flashcards must test understanding of CONCEPTS, FACTS, DEFINITIONS, and IDEAS found in the document. "
        "STRICTLY IGNORE any of the following — they are document metadata, not study content: "
        "file names, file sizes, submission IDs, dates, deadlines, form fields, author names, "
        "page numbers, headers, footers, timestamps, version numbers, and any administrative details. "
        "If the document contains very little actual study content, generate the best cards you can from what exists. "
        "Do not use emojis. Return ONLY valid JSON — no markdown, no backticks, no preamble."
    )

    prompt = (
        f'Generate exactly {card_count} flashcards from the STUDY CONTENT of the following document: "{filename}"\n\n'
        "Each question should test meaningful understanding — not trivia about the document itself.\n"
        "Good question: 'What is the purpose of X?' / 'How does Y work?' / 'Define Z'\n"
        "Bad question: 'What is the submission date?' / 'What is the file name?' / 'What is the document ID?'\n\n"
        "Return ONLY a JSON array in this exact format:\n"
        '[{"q": "question text", "a": "answer text"}, ...]\n\n'
        "Document content:\n"
        f"{document_text}"
    )

    try:
        raw = _call_gemini(system_instruction, prompt, temperature=0.2)
        cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        cards = json.loads(cleaned)

        if not isinstance(cards, list):
            raise ValueError(f"Expected JSON array, got {type(cards)}")

        validated = []
        for i, card in enumerate(cards):
            if not isinstance(card, dict) or "q" not in card or "a" not in card:
                logger.warning(f"Skipping malformed card at index {i}: {card}")
                continue
            validated.append({"q": str(card["q"]).strip(), "a": str(card["a"]).strip()})

        if not validated:
            raise ValueError("No valid cards parsed from Gemini response")

        if len(validated) != card_count:
            logger.warning(
                f"Expected {card_count} flashcards, got {len(validated)} for file '{filename}'"
            )

        logger.info(f"Generated {len(validated)} flashcards for '{filename}'")
        return validated

    except json.JSONDecodeError as e:
        logger.error(
            f"Failed to parse flashcard JSON for '{filename}': {e}\nRaw response: {raw[:500]}"
        )
        raise ValueError(f"Gemini returned invalid JSON: {e}")


def generate_quiz(
    chunks: list[str],
    filename: str = "document",
    question_count: int = 6,
) -> list[dict]:
    """
    Generate multiple-choice quiz questions from reconstructed document chunks.

    Returns:
        List of dicts:
        [{"q": "...", "options": ["A","B","C","D"], "answer": 0, "explanation": "..."}, ...]
    Raises:
        ValueError: If Gemini returns unparseable or malformed JSON
    """
    import json

    if not chunks:
        raise ValueError("No chunks provided for quiz generation")

    document_text = "\n\n".join(chunks)

    max_chars = 80_000
    if len(document_text) > max_chars:
        logger.warning(
            f"Document text truncated from {len(document_text)} to {max_chars} chars for quiz generation"
        )
        document_text = document_text[:max_chars]

    system_instruction = (
        f"You are {settings.SERVICE_NAME}, a study assistant that generates high-quality multiple-choice quiz questions. "
        "Questions must test genuine understanding of CONCEPTS, FACTS, DEFINITIONS, and IDEAS. "
        "STRICTLY IGNORE document metadata: file names, submission IDs, dates, form fields, "
        "author names, page numbers, headers, footers, and administrative details. "
        "Distribute correct answers randomly across positions A, B, C, D — do not always put "
        "the correct answer first. Make all four options plausible to prevent easy guessing. "
        "Do not use emojis. Return ONLY valid JSON — no markdown, no backticks, no preamble."
    )

    prompt = (
        f"Generate exactly {question_count} multiple-choice questions from the STUDY CONTENT "
        f'of the following document: "{filename}"\n\n'
        "Each question must have exactly 4 options and one correct answer.\n"
        "Include a brief explanation (1-2 sentences) for why the correct answer is right.\n\n"
        "Return ONLY a JSON array in this exact format:\n"
        '[{"q": "Question?", "options": ["Option A", "Option B", "Option C", "Option D"], '
        '"answer": 0, "explanation": "Reason why Option A is correct."}, ...]\n\n'
        "The 'answer' field is the 0-based index of the correct option.\n\n"
        "Document content:\n"
        f"{document_text}"
    )

    try:
        raw = _call_gemini(system_instruction, prompt, temperature=0.3)
        cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        questions = json.loads(cleaned)

        if not isinstance(questions, list):
            raise ValueError(f"Expected JSON array, got {type(questions)}")

        validated = []
        for i, q in enumerate(questions):
            if not isinstance(q, dict):
                logger.warning(f"Skipping non-dict question at index {i}")
                continue
            if not all(k in q for k in ("q", "options", "answer")):
                logger.warning(
                    f"Skipping question missing required fields at index {i}: {q}"
                )
                continue
            if not isinstance(q["options"], list) or len(q["options"]) != 4:
                logger.warning(
                    f"Skipping question {i} with wrong option count: {len(q.get('options', []))}"
                )
                continue
            if not isinstance(q["answer"], int) or not (0 <= q["answer"] <= 3):
                logger.warning(
                    f"Skipping question {i} with invalid answer index: {q['answer']}"
                )
                continue
            validated.append(
                {
                    "q": str(q["q"]).strip(),
                    "options": [str(o).strip() for o in q["options"]],
                    "answer": int(q["answer"]),
                    "explanation": str(q.get("explanation", "")).strip(),
                }
            )

        if not validated:
            raise ValueError("No valid questions parsed from Gemini response")

        if len(validated) != question_count:
            logger.warning(
                f"Expected {question_count} questions, got {len(validated)} for file '{filename}'"
            )

        logger.info(f"Generated {len(validated)} quiz questions for '{filename}'")
        return validated

    except json.JSONDecodeError as e:
        logger.error(
            f"Failed to parse quiz JSON for '{filename}': {e}\nRaw response: {raw[:500]}"
        )
        raise ValueError(f"Gemini returned invalid JSON: {e}")


def generate_summary(
    chunks: list[str],
    filename: str = "document",
    style: str = "bullets",
) -> str:
    """
    Generate a summary from reconstructed document chunks.

    Args:
        chunks:   Ordered list of text chunks from ChromaDB
        filename: Original filename for context
        style:    'bullets' | 'key_terms'

    Returns:
        Summary as a plain text string (markdown-flavoured for the frontend to render)
    Raises:
        ValueError: on empty input or Gemini failure
    """
    if not chunks:
        raise ValueError("No chunks provided for summary generation")

    document_text = "\n\n".join(chunks)

    max_chars = 80_000
    if len(document_text) > max_chars:
        logger.warning(
            f"Document truncated from {len(document_text)} to {max_chars} chars for summary"
        )
        document_text = document_text[:max_chars]

    if style == "key_terms":
        system_instruction = (
            "You are a study assistant that extracts key terms and definitions from documents. "
            "For each important term, concept, or idea, provide a clear, concise definition. "
            "Ignore metadata, file info, dates, and administrative content. "
            "Do not use emojis. Format your response as a clean list."
        )
        prompt = (
            f'Extract the key terms and definitions from this document: "{filename}"\n\n'
            "Return a list of terms in this exact format — one per line, no extra spacing:\n"
            "**Term**: Definition in one or two clear sentences.\n\n"
            "Focus on concepts that a learner would need to understand and remember.\n\n"
            f"Document:\n{document_text}"
        )
    else:  # bullets
        system_instruction = (
            "You are a study assistant that creates concise, accurate bullet-point summaries. "
            "Each bullet should capture one key idea, finding, or concept from the document. "
            "Group related bullets under short bold headings where appropriate. "
            "Ignore metadata, file info, dates, and administrative content. "
            "Do not use emojis."
        )
        prompt = (
            f'Summarize the key content of this document in bullet points: "{filename}"\n\n'
            "Use this format:\n"
            "**Section or Theme**\n"
            "- Key point\n"
            "- Key point\n\n"
            "Be thorough but concise. Cover all major ideas a learner should know.\n\n"
            f"Document:\n{document_text}"
        )

    try:
        result = _call_gemini(system_instruction, prompt, temperature=0.2)
        if not result or not result.strip():
            raise ValueError("Gemini returned an empty summary")
        logger.info(f"Generated {style} summary for '{filename}'")
        return result.strip()
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"generate_summary failed for '{filename}': {e}", exc_info=True)
        raise
