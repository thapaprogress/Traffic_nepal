# -*- coding: utf-8 -*-
"""
api/routers/auth.py
JWT-based authentication for the Traffic Eye police portal.
Roles: admin, operator, viewer
"""

import os
import time
import hashlib
import secrets
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

router = APIRouter(prefix="/auth", tags=["auth"])

# Simple secret key — in production use env variable
SECRET_KEY = os.environ.get("JWT_SECRET", "traffic-eye-nepal-secret-2026")

# In-memory user store for MVP (replace with DB in production)
USERS = {
    "admin": {
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "admin",
        "name": "System Admin",
    },
    "operator": {
        "password_hash": hashlib.sha256("operator123".encode()).hexdigest(),
        "role": "operator",
        "name": "Traffic Operator",
    },
    "viewer": {
        "password_hash": hashlib.sha256("viewer123".encode()).hexdigest(),
        "role": "viewer",
        "name": "Dashboard Viewer",
    },
}

# Active tokens (in-memory for MVP, use Redis in production)
_active_tokens: dict = {}


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    role: str
    name: str
    expires_in: int


class UserInfo(BaseModel):
    username: str
    role: str
    name: str


@router.post("/login", response_model=LoginResponse)
def login(req: LoginRequest):
    user = USERS.get(req.username)
    if not user:
        raise HTTPException(401, "Invalid credentials")

    pw_hash = hashlib.sha256(req.password.encode()).hexdigest()
    if pw_hash != user["password_hash"]:
        raise HTTPException(401, "Invalid credentials")

    # Generate token
    token = secrets.token_hex(32)
    expires_in = 86400  # 24 hours
    _active_tokens[token] = {
        "username": req.username,
        "role": user["role"],
        "name": user["name"],
        "expires_at": time.time() + expires_in,
    }

    return LoginResponse(
        token=token,
        role=user["role"],
        name=user["name"],
        expires_in=expires_in,
    )


@router.post("/logout")
def logout(authorization: str = Header(None)):
    token = _extract_token(authorization)
    _active_tokens.pop(token, None)
    return {"status": "logged out"}


@router.get("/me", response_model=UserInfo)
def get_current_user(authorization: str = Header(None)):
    token = _extract_token(authorization)
    session = _validate_token(token)
    return UserInfo(
        username=session["username"],
        role=session["role"],
        name=session["name"],
    )


def _extract_token(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(401, "No authorization header")
    if authorization.startswith("Bearer "):
        return authorization[7:]
    return authorization


def _validate_token(token: str) -> dict:
    session = _active_tokens.get(token)
    if not session:
        raise HTTPException(401, "Invalid or expired token")
    if time.time() > session["expires_at"]:
        _active_tokens.pop(token, None)
        raise HTTPException(401, "Token expired")
    return session


def require_role(*roles):
    """Dependency factory for role-based access control."""
    def dependency(authorization: str = Header(None)):
        token = _extract_token(authorization)
        session = _validate_token(token)
        if session["role"] not in roles:
            raise HTTPException(403, f"Requires role: {roles}")
        return session
    return Depends(dependency)
