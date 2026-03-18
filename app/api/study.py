# app/api/study.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from typing import List, Optional
import logging
import json

from app.db.database import get_db
from app.db.models import FlashcardDeck, SummaryResult
from app.db.vector_store import get_vector_store
from app.services.llm_utils import generate_flashcards, generate_quiz, generate_summary
from app.api.auth import verify_token

logger = logging.getLogger(__name__)
study_router = APIRouter()


# Shared schemas

class FlashcardCard(BaseModel):
    q: str
    a: str


# Flashcard schemas

class GenerateRequest(BaseModel):
    file_id: str
    filename: Optional[str] = "document"
    card_count: int = Field(default=8, ge=4, le=20)
    save: bool = True


class GenerateResponse(BaseModel):
    deck_id: Optional[str] = None
    file_id: str
    title: str
    cards: List[FlashcardCard]
    card_count: int


class DeckSummary(BaseModel):
    deck_id: str
    title: str
    file_id: Optional[str]
    card_count: int
    created_at: int


class DecksListResponse(BaseModel):
    decks: List[DeckSummary]


class DeckDetailResponse(BaseModel):
    deck_id: str
    title: str
    file_id: Optional[str]
    cards: List[FlashcardCard]
    card_count: int
    created_at: int


# Quiz schemas

class QuizOption(BaseModel):
    """One multiple-choice option."""

    text: str


class QuizQuestion(BaseModel):
    q: str
    options: List[str]  # exactly 4 option strings
    answer: int  # 0-based index of the correct option
    explanation: Optional[str] = None


class QuizGenerateRequest(BaseModel):
    file_id: str
    filename: Optional[str] = "document"
    question_count: int = Field(default=6, ge=3, le=20)


class QuizGenerateResponse(BaseModel):
    file_id: str
    title: str
    questions: List[QuizQuestion]
    question_count: int


# Shared helper

def _reconstruct_chunks_from_store(user_id: str, file_id: str) -> tuple[list[str], str]:
    """
    Pull all stored chunks for a file_id back out of ChromaDB,
    sort by chunk_index, and return (ordered_chunks, filename).
    Raises HTTPException 404 if no chunks found.
    """
    vector_store = get_vector_store()
    collection = vector_store.get_user_collection(user_id)

    results = collection.get(
        where={"file_id": file_id},
        include=["documents", "metadatas"],
    )

    if not results["ids"]:
        raise HTTPException(
            404, f"No document found with file_id '{file_id}' for this user"
        )

    paired = []
    filename = "document"
    for i, doc in enumerate(results["documents"]):
        meta = results["metadatas"][i]
        chunk_index = meta.get("chunk_index", i)
        filename = meta.get("filename", filename)
        paired.append((chunk_index, doc))

    paired.sort(key=lambda x: x[0])
    ordered_chunks = [text for _, text in paired]

    logger.info(
        f"Reconstructed {len(ordered_chunks)} chunks for file_id '{file_id}' ('{filename}')"
    )
    return ordered_chunks, filename


# Flashcard routes


@study_router.post("/flashcards/generate", response_model=GenerateResponse)
async def generate_deck(
    request: GenerateRequest,
    user_id: str = Depends(verify_token),
    db: Session = Depends(get_db),
):
    """
    Generate flashcards from a previously uploaded document.
    Reconstructs text from ChromaDB chunks, sends to Gemini,
    and optionally persists the deck to SQLite.
    """
    try:
        chunks, filename = _reconstruct_chunks_from_store(user_id, request.file_id)
        display_name = request.filename if request.filename != "document" else filename

        cards = generate_flashcards(
            chunks, filename=display_name, card_count=request.card_count
        )
        title = display_name

        deck_id = None
        if request.save:
            deck = FlashcardDeck(
                user_id=user_id,
                file_id=request.file_id,
                title=title,
                cards_json=json.dumps(cards),
                card_count=len(cards),
            )
            db.add(deck)
            db.commit()
            db.refresh(deck)
            deck_id = deck.id
            logger.info(f"Saved deck '{deck_id}' for user '{user_id}'")

        return GenerateResponse(
            deck_id=deck_id,
            file_id=request.file_id,
            title=title,
            cards=[FlashcardCard(**c) for c in cards],
            card_count=len(cards),
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Flashcard generation value error: {e}")
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.error(f"Error generating flashcards: {e}", exc_info=True)
        raise HTTPException(500, f"Error generating flashcards: {str(e)}")


@study_router.get("/flashcards", response_model=DecksListResponse)
async def list_decks(
    user_id: str = Depends(verify_token),
    db: Session = Depends(get_db),
):
    """List all saved flashcard decks for the authenticated user, newest first."""
    try:
        decks = (
            db.query(FlashcardDeck)
            .filter(FlashcardDeck.user_id == user_id)
            .order_by(FlashcardDeck.created_at.desc())
            .all()
        )
        return DecksListResponse(
            decks=[
                DeckSummary(
                    deck_id=d.id,
                    title=d.title,
                    file_id=d.file_id,
                    card_count=d.card_count,
                    created_at=d.created_at,
                )
                for d in decks
            ]
        )
    except Exception as e:
        logger.error(f"Error listing decks for user {user_id}: {e}", exc_info=True)
        raise HTTPException(500, f"Error listing decks: {str(e)}")


@study_router.get("/flashcards/{deck_id}", response_model=DeckDetailResponse)
async def get_deck(
    deck_id: str,
    user_id: str = Depends(verify_token),
    db: Session = Depends(get_db),
):
    """Retrieve a saved deck by ID."""
    try:
        deck = (
            db.query(FlashcardDeck)
            .filter(FlashcardDeck.id == deck_id, FlashcardDeck.user_id == user_id)
            .first()
        )
        if not deck:
            raise HTTPException(404, "Deck not found")

        cards = json.loads(deck.cards_json)
        return DeckDetailResponse(
            deck_id=deck.id,
            title=deck.title,
            file_id=deck.file_id,
            cards=[FlashcardCard(**c) for c in cards],
            card_count=deck.card_count,
            created_at=deck.created_at,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching deck {deck_id}: {e}", exc_info=True)
        raise HTTPException(500, f"Error fetching deck: {str(e)}")


@study_router.delete("/flashcards/{deck_id}")
async def delete_deck(
    deck_id: str,
    user_id: str = Depends(verify_token),
    db: Session = Depends(get_db),
):
    """Permanently delete a saved deck."""
    try:
        deck = (
            db.query(FlashcardDeck)
            .filter(FlashcardDeck.id == deck_id, FlashcardDeck.user_id == user_id)
            .first()
        )
        if not deck:
            raise HTTPException(404, "Deck not found")

        db.delete(deck)
        db.commit()
        logger.info(f"Deleted deck '{deck_id}' for user '{user_id}'")
        return {"deleted": True, "deck_id": deck_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting deck {deck_id}: {e}", exc_info=True)
        raise HTTPException(500, f"Error deleting deck: {str(e)}")


# Quiz routes


@study_router.post("/quiz/generate", response_model=QuizGenerateResponse)
async def generate_quiz_endpoint(
    request: QuizGenerateRequest,
    user_id: str = Depends(verify_token),
):
    """
    Generate a multiple-choice quiz from a previously uploaded document.
    Reconstructs text from ChromaDB chunks and sends to Gemini.
    Quizzes are stateless — they are not persisted to the database.
    """
    try:
        chunks, filename = _reconstruct_chunks_from_store(user_id, request.file_id)
        display_name = request.filename if request.filename != "document" else filename

        questions = generate_quiz(
            chunks,
            filename=display_name,
            question_count=request.question_count,
        )

        return QuizGenerateResponse(
            file_id=request.file_id,
            title=display_name,
            questions=[QuizQuestion(**q) for q in questions],
            question_count=len(questions),
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Quiz generation value error: {e}")
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.error(f"Error generating quiz: {e}", exc_info=True)
        raise HTTPException(500, f"Error generating quiz: {str(e)}")


# Summarize schemas


class SummaryStyle(str):
    bullets = "bullets"
    key_terms = "key_terms"


class SummarizeRequest(BaseModel):
    file_id: str
    filename: Optional[str] = "document"
    style: str = Field(default="bullets", pattern="^(bullets|key_terms)$")


class SummarizeResponse(BaseModel):
    summary_id: str
    file_id: str
    title: str
    style: str
    content: str
    created_at: int


class SummaryListResponse(BaseModel):
    summaries: List[SummarizeResponse]


# Summarize routes


@study_router.post("/summarize", response_model=SummarizeResponse)
async def create_summary(
    request: SummarizeRequest,
    user_id: str = Depends(verify_token),
    db: Session = Depends(get_db),
):
    """
    Generate a summary from a previously uploaded document and persist it.
    Style must be 'bullets' or 'key_terms'.
    """
    try:
        chunks, filename = _reconstruct_chunks_from_store(user_id, request.file_id)
        display_name = request.filename if request.filename != "document" else filename

        content = generate_summary(chunks, filename=display_name, style=request.style)

        record = SummaryResult(
            user_id=user_id,
            file_id=request.file_id,
            title=display_name,
            style=request.style,
            content=content,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        logger.info(f"Saved summary '{record.id}' for user '{user_id}'")

        return SummarizeResponse(
            summary_id=record.id,
            file_id=record.file_id,
            title=record.title,
            style=record.style,
            content=record.content,
            created_at=record.created_at,
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Summary generation value error: {e}")
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.error(f"Error generating summary: {e}", exc_info=True)
        raise HTTPException(500, f"Error generating summary: {str(e)}")


@study_router.get("/summaries", response_model=SummaryListResponse)
async def list_summaries(
    user_id: str = Depends(verify_token),
    db: Session = Depends(get_db),
):
    """List all saved summaries for the authenticated user, newest first."""
    try:
        rows = (
            db.query(SummaryResult)
            .filter(SummaryResult.user_id == user_id)
            .order_by(SummaryResult.created_at.desc())
            .all()
        )
        return SummaryListResponse(
            summaries=[
                SummarizeResponse(
                    summary_id=r.id,
                    file_id=r.file_id,
                    title=r.title,
                    style=r.style,
                    content=r.content,
                    created_at=r.created_at,
                )
                for r in rows
            ]
        )
    except Exception as e:
        logger.error(f"Error listing summaries for user {user_id}: {e}", exc_info=True)
        raise HTTPException(500, f"Error listing summaries: {str(e)}")


@study_router.get("/summaries/{summary_id}", response_model=SummarizeResponse)
async def get_summary(
    summary_id: str,
    user_id: str = Depends(verify_token),
    db: Session = Depends(get_db),
):
    """Retrieve a saved summary by ID."""
    try:
        row = (
            db.query(SummaryResult)
            .filter(SummaryResult.id == summary_id, SummaryResult.user_id == user_id)
            .first()
        )
        if not row:
            raise HTTPException(404, "Summary not found")

        return SummarizeResponse(
            summary_id=row.id,
            file_id=row.file_id,
            title=row.title,
            style=row.style,
            content=row.content,
            created_at=row.created_at,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching summary {summary_id}: {e}", exc_info=True)
        raise HTTPException(500, f"Error fetching summary: {str(e)}")


@study_router.delete("/summaries/{summary_id}")
async def delete_summary(
    summary_id: str,
    user_id: str = Depends(verify_token),
    db: Session = Depends(get_db),
):
    """Permanently delete a saved summary."""
    try:
        row = (
            db.query(SummaryResult)
            .filter(SummaryResult.id == summary_id, SummaryResult.user_id == user_id)
            .first()
        )
        if not row:
            raise HTTPException(404, "Summary not found")

        db.delete(row)
        db.commit()
        logger.info(f"Deleted summary '{summary_id}' for user '{user_id}'")
        return {"deleted": True, "summary_id": summary_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting summary {summary_id}: {e}", exc_info=True)
        raise HTTPException(500, f"Error deleting summary: {str(e)}")
