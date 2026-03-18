# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.config import settings

from app.services.embedding_service import get_embedding_service
from app.db.vector_store import get_vector_store
from app.services.reranker import get_reranker

# db stuff
from app.db.database import init_db
from app.api.auth import auth_router

# routers
from app.api.upload import upload_router
from app.api.chat import chat_router
from app.api.admin import admin_router
from app.api.conversations import conversations_router
from app.api.study import study_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting up {settings.SERVICE_NAME}... warming up models")

    logger.info("Initializing SQL database")
    init_db()
    logger.info("SQL database ready.")

    logger.info("Loading embedding model")
    get_embedding_service()
    logger.info("Embedding model ready.")

    logger.info("Initializing vector store")
    get_vector_store()
    logger.info("Vector store ready.")

    logger.info("Loading reranker model")
    get_reranker()
    logger.info("Reranker model ready.")

    logger.info(f"All models loaded. {settings.SERVICE_NAME} is ready.")

    yield

    logger.info(f"Shutting down {settings.SERVICE_NAME}")


app = FastAPI(
    title="Mabel",
    description="AI Powered RAG centric Study Support System.",
    version="1.0",
    lifespan=lifespan,
)

allowed_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8000",
    "null",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Mabel API is running", "version": "1.0", "status": "healthy"}


app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
app.include_router(upload_router, prefix="/api/upload", tags=["uploads"])
app.include_router(admin_router, prefix="/api/admin", tags=["admin"])
app.include_router(chat_router, prefix="/api/chat", tags=["chat"])
app.include_router(
    conversations_router, prefix="/api/conversations", tags=["conversations"]
)
app.include_router(study_router, prefix="/api/study", tags="study")
