# app/config.py
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # api Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")

    # Model Settings
    GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
    GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.3"))

    # Embedding Settings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    RERANKER_MODEL: str = os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    INITIAL_RETRIEVAL_K: int = int(os.getenv("INITIAL_RETRIEVAL_K", "15"))
    FINAL_TOP_K: int = int(os.getenv("FINAL_TOP_K", "5"))

    # Vector store settings
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "./chroma_db")

    # Service name
    SERVICE_NAME: str = os.getenv("SERVICE_NAME")

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./mabel.db")

    # Auth
    SECRET_KEY: str = os.getenv("SECRET_KEY")
    JWT_ALGORITHM: str = "HS256"
    TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    # chat routing stuff
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))


settings = Settings()
