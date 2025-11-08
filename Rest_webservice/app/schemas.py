# app/schemas.py
from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, EmailStr, Field
from pydantic import ConfigDict  # Pydantic v2 config

# ---------- Auth ----------
class Login(BaseModel):
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

# ---------- Users ----------
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    role: str = "user"  # keep as str to match your current code

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    password: Optional[str] = None
    role: Optional[str] = None

class UserOut(BaseModel):
    # Pydantic v2 style: allow from ORM objects
    model_config = ConfigDict(from_attributes=True)
    id: int
    email: EmailStr
    role: str

# Optional token shape 
class TokenPair(BaseModel):
    access_token: str
    token_type: str = "bearer"

# ---------- Symptoms ----------
class SymptomIn(BaseModel):
    code: str
    name: str

class SymptomOut(SymptomIn):
    model_config = ConfigDict(from_attributes=True)
    id: int

# ---------- Events ----------
class EventIn(BaseModel):
    user_id: int
    payload: Dict[str, Any] = Field(default_factory=dict)
    role: Optional[str] = None
    geo: Optional[str] = None
    age: Optional[int] = None


class EventOut(BaseModel):
    id: int
    segment: str
