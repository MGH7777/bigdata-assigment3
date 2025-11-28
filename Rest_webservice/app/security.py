# app/security.py
import os
from datetime import datetime, timedelta
from typing import Dict, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import jwt
from passlib.hash import pbkdf2_sha256 as hasher  # portable, no C wheels

JWT_SECRET: str = os.getenv("JWT_SECRET", "dev-secret")
JWT_ALGO: str = "HS256"
JWT_EXPIRE_MINUTES: int = 60  # default token lifetime

security = HTTPBearer(auto_error=False)


# ---------------- Password hashing ----------------
def hash_password(password: str) -> str:
    """
    Hash a plaintext password using PBKDF2-SHA256.
    """
    return hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """
    Verify a plaintext password against the stored hash.
    """
    return hasher.verify(password, password_hash)


# ---------------- JWT helpers ----------------
def create_access_token(
    claims: Dict,
    expires_minutes: int = JWT_EXPIRE_MINUTES,
) -> str:
    """
    Create a signed JWT with the provided claims and expiration.
    """
    to_encode = claims.copy()
    expire = datetime.utcnow() + timedelta(minutes=expires_minutes)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGO)


def get_current_claims(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[Dict]:
    """
    Read and decode JWT from the Authorization: Bearer <token> header.
    Returns the decoded claims dict, or None if no credentials were provided.
    """
    if not creds:
        return None
    token = creds.credentials
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )


def require_roles(roles: list[str]):
    """
    Dependency that enforces role-based access.
    Usage: Depends(require_roles(["admin"]))
    """
    def _dep(claims: Optional[Dict] = Depends(get_current_claims)):
        role = (claims or {}).get("role", "anon")
        if role not in roles:
            raise HTTPException(status_code=403, detail="Insufficient role")
    return _dep

