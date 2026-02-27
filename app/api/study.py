# app/api/study.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from typing import List, Optional
import logging
import json

from app.db.database import get_db
from app.db.models import FlashcardDeck
from app.db.vector_store import get_vector_store
from app.services.llm_utils import generate_flashcards
from app.api.auth import verify_token

logger = logging.getLogger(__name__)
study_router = APIRouter()


# schema shit
class FlashcardCard(BaseModel):
    q: str
    a: str


class GenerateRequest(BaseModel):
    file_id: str
    filename: Optional[str] = "document"
    card_count: int = Field(default=8, ge=4, le=20)
    save: bool = True


class GenerateResponse(BaseModel):
    deck_id: Optional[str] = None           # None if save=False
    file_id: str
    title: str
    cards: List[FlashcardCard]
    card_count: int


class DeckSummary(BaseModel):
    deck_id: str
    title: str
    file_id: Optional[str]
    card_count: int
    created_at: int # unix type integer


class DecksListResponse(BaseModel):
    decks: List[DeckSummary]


class DeckDetailResponse(BaseModel):
    deck_id: str
    title: str
    file_id: Optional[str]
    cards: List[FlashcardCard]
    card_count: int
    created_at: int


# helper functions -- too lazy to createa new file 
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
        raise HTTPException(404, f"No document found with file_id '{file_id}' for this user")

    # Pair each chunk with its index so we can sort
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


# relevant routes
@study_router.post("/flashcards/generate", response_model=GenerateResponse)
async def generate_deck(
    request: GenerateRequest,
    user_id: str = Depends(verify_token),
    db: Session = Depends(get_db),
):
    """
    Generate x flashcards from a previously uploaded document.

    Reconstructs the document text from its stored ChromaDB chunks,
    sends it to Gemini and optionally persists the deck to SQLite.
    """
    try:
        # Rebuild text from stored chunks
        chunks, filename = _reconstruct_chunks_from_store(user_id, request.file_id)

        # Use frontend-supplied filename if provided, fallback to what's in metadata
        display_name = request.filename if request.filename != "document" else filename

        # Generate via Gemini
        cards = generate_flashcards(chunks, filename=display_name, card_count=request.card_count)

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
        # Gemini JSON parse failures etc.
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
    """Retrieve a saved deck by ID — used to restore a session without regenerating."""
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