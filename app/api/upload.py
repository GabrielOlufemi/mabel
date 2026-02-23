# app/api/upload.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pathlib import Path
import shutil
from typing import List
import logging
import uuid

from app.services.document_processor import process_document
from app.services.utils import chunk_text
from app.services.embedding_service import get_embedding_service
from app.db.vector_store import get_vector_store
from app.api.auth import verify_token

logger = logging.getLogger(__name__)

upload_router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@upload_router.post("/upload")
async def upload_document(
    file: UploadFile = File(...), user_id: str = Depends(verify_token)
):
    """Upload and process a single document"""
    allowed_extensions = [".pdf", ".docx", ".txt"]
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_extension} not supported. Allowed: {', '.join(allowed_extensions)}",
        )

    document_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{document_id}_{file.filename}"

    try:
        logger.info(f"Saving file: {file.filename} for user: {user_id}")
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Extracting text from {file.filename}")
        text = process_document(file_path)

        if not text or not text.strip():
            raise HTTPException(400, "No text could be extracted from the document")

        logger.info(f"Chunking text from {file.filename}")
        chunks = chunk_text(text)

        if not chunks:
            raise HTTPException(400, "No valid chunks created from document")

        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        embedding_service = get_embedding_service()
        embeddings = embedding_service.generate_embeddings(chunks)

        logger.info(
            f"Storing {len(chunks)} chunks in vector database for user {user_id}"
        )
        vector_store = get_vector_store()
        chunks_stored = vector_store.add_document(
            user_id=user_id,
            file_id=document_id,
            filename=file.filename,
            chunks=chunks,
            embeddings=embeddings,
            additional_metadata={
                "file_extension": file_extension,
                "original_text_length": len(text),
            },
        )

        logger.info(f"Successfully processed {file.filename} for user {user_id}")

        return {
            "status": "success",
            "message": "File processed and stored successfully",
            "document_id": document_id,
            "filename": file.filename,
            "user_id": user_id,
            "stats": {
                "original_text_length": len(text),
                "chunks_created": len(chunks),
                "chunks_stored": chunks_stored,
                "embedding_dimension": embedding_service.embedding_dim,
            },
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {e}")
        raise HTTPException(500, f"Error processing file: {str(e)}")

    finally:
        if file_path.exists():
            file_path.unlink()


@upload_router.post("/upload/batch")
async def upload_multiple_documents(
    files: List[UploadFile] = File(...), user_id: str = Depends(verify_token)
):
    """Upload and process multiple documents"""
    results = []
    successful = 0
    failed = 0

    embedding_service = get_embedding_service()
    vector_store = get_vector_store()

    for file in files:
        try:
            file_extension = Path(file.filename).suffix.lower()
            if file_extension not in [".pdf", ".docx", ".txt"]:
                results.append(
                    {
                        "filename": file.filename,
                        "status": "skipped",
                        "reason": f"Unsupported file type: {file_extension}",
                    }
                )
                failed += 1
                continue

            document_id = str(uuid.uuid4())
            file_path = UPLOAD_DIR / f"{document_id}_{file.filename}"

            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            text = process_document(file_path)
            chunks = chunk_text(text)
            embeddings = embedding_service.generate_embeddings(chunks)

            chunks_stored = vector_store.add_document(
                user_id=user_id,
                file_id=document_id,
                filename=file.filename,
                chunks=chunks,
                embeddings=embeddings,
            )

            results.append(
                {
                    "filename": file.filename,
                    "document_id": document_id,
                    "status": "success",
                    "chunks_stored": chunks_stored,
                }
            )
            successful += 1

            if file_path.exists():
                file_path.unlink()

        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            results.append(
                {"filename": file.filename, "status": "failed", "error": str(e)}
            )
            failed += 1

    return {
        "message": "Batch upload completed",
        "user_id": user_id,
        "total_files": len(files),
        "successful": successful,
        "failed": failed,
        "results": results,
    }


@upload_router.get("/documents")
async def list_user_documents(user_id: str = Depends(verify_token)):
    """List all documents for a user"""
    try:
        vector_store = get_vector_store()
        documents = vector_store.list_user_documents(user_id)
        stats = vector_store.get_user_stats(user_id)

        return {
            "user_id": user_id,
            "total_documents": len(documents),
            "total_chunks": stats.get("total_chunks", 0),
            "documents": documents,
        }

    except Exception as e:
        logger.error(f"Error listing documents for user {user_id}: {e}")
        raise HTTPException(500, str(e))


@upload_router.delete("/documents/all")
async def clear_all_documents(user_id: str = Depends(verify_token)):
    """Delete all documents for a user"""
    try:
        vector_store = get_vector_store()
        success = vector_store.delete_user_collection(user_id)

        if success:
            return {
                "status": "success",
                "message": "All documents cleared successfully",
            }
        else:
            raise HTTPException(500, "Failed to clear documents")

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error clearing documents for user {user_id}: {e}")
        raise HTTPException(500, str(e))


@upload_router.delete("/documents/{document_id}")
async def delete_document(document_id: str, user_id: str = Depends(verify_token)):
    """Delete a specific document"""
    try:
        vector_store = get_vector_store()
        success = vector_store.delete_document(user_id, document_id)

        if success:
            return {
                "status": "success",
                "message": f"Document {document_id} deleted successfully",
            }
        else:
            raise HTTPException(404, f"Document {document_id} not found")

    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(500, str(e))


@upload_router.get("/health")
async def upload_health():
    """Health check — no auth needed"""
    try:
        embedding_service = get_embedding_service()
        vector_store = get_vector_store()

        return {
            "status": "healthy",
            "embedding_model": embedding_service.get_model_info(),
            "vector_store": "chromadb",
            "upload_dir_exists": UPLOAD_DIR.exists(),
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
