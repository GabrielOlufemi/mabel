# app/api/auth.py
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
import os

from app.db.database import get_db
from app.db.models import User
from app.config import settings

auth_router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.JWT_ALGORITHM
TOKEN_EXPIRE_MINUTES = settings.TOKEN_EXPIRE_MINUTES


# ── Schemas ──────────────────────────────────────────────
class RegisterRequest(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str
    first_name: str
    last_name: str

class UpdateProfileRequest(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr


# ── Helpers ──────────────────────────────────────────────
def create_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "exp": datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(401, "Invalid token")
        return user_id
    except JWTError:
        raise HTTPException(401, "Invalid or expired token")


# ── Routes ───────────────────────────────────────────────
@auth_router.post("/register", response_model=AuthResponse)
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == request.email).first():
        raise HTTPException(400, "Email already registered")

    if len(request.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")

    user = User(
        first_name=request.first_name,
        last_name=request.last_name,
        email=request.email,
        hashed_password=pwd_context.hash(request.password)
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return AuthResponse(
        access_token=create_token(user.id),
        user_id=user.id,
        email=user.email,
        first_name=user.first_name,
        last_name=user.last_name
    )


@auth_router.post("/login", response_model=AuthResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == request.email).first()

    if not user or not pwd_context.verify(request.password, user.hashed_password):
        raise HTTPException(401, "Invalid email or password")

    if not user.is_active:
        raise HTTPException(401, "Invalid email or password")

    return AuthResponse(
        access_token=create_token(user.id),
        user_id=user.id,
        email=user.email,
        first_name=user.first_name,
        last_name=user.last_name
    )


@auth_router.get("/me")
async def get_me(user_id: str = Depends(verify_token), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found")
    return {
        "user_id": user.id,
        "email": user.email,
        "first_name": user.first_name,
        "last_name": user.last_name
    }


@auth_router.patch("/update-profile")
async def update_profile(
    request: UpdateProfileRequest,
    user_id: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found")

    # check email isn't already taken by someone else
    existing = db.query(User).filter(User.email == request.email, User.id != user_id).first()
    if existing:
        raise HTTPException(400, "Email already in use")

    user.first_name = request.first_name
    user.last_name  = request.last_name
    user.email      = request.email
    db.commit()
    db.refresh(user)

    return {
        "status": "success",
        "first_name": user.first_name,
        "last_name": user.last_name,
        "email": user.email
    }


@auth_router.delete("/delete-account")
async def delete_account(user_id: str = Depends(verify_token), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found")
    user.is_active = False
    db.commit()
    return {"status": "success"}