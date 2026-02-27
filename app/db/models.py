# app/db/models.py
from sqlalchemy import Column, String, DateTime, Boolean
from datetime import datetime
import uuid

# added for flashcards
import time
from sqlalchemy import Column, String, Integer, Text, ForeignKey

from app.db.database import Base



class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False, index=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class FlashcardDeck(Base):
    __tablename__ = "flashcard_decks"

    id         = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id    = Column(String, nullable=False, index=True)
    file_id    = Column(String, nullable=True)   # FK-ish ref to ChromaDB file_id, nullable for manually created decks
    title      = Column(String, nullable=False)  # derived from filename
    cards_json = Column(Text, nullable=False)    # JSON string: [{"q": "...", "a": "..."}]
    card_count = Column(Integer, nullable=False, default=8)
    created_at = Column(Integer, nullable=False, default=lambda: int(time.time()))  # unix timestamp
