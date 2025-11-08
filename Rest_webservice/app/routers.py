# app/routers.py
from __future__ import annotations

from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import get_db
from app import models, schemas
from app.security import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_claims,
    require_roles,
)
from app.services import make_segment, frequent_pairs, extract_symptom_features
from app.rate_limit import rate_limiter

router = APIRouter()


# HATEOAS link helpers
def user_links(uid: int) -> Dict[str, Any]:
    return {
        "self": {"href": f"/users/{uid}"},
        "events": {"href": f"/users/{uid}/events"},
        "update": {"href": f"/users/{uid}", "method": "PUT"},
        "delete": {"href": f"/users/{uid}", "method": "DELETE"},
    }


def event_links(eid: int) -> Dict[str, Any]:
    return {"self": {"href": f"/events/{eid}"}}


def collection_links(path: str) -> Dict[str, Any]:
    return {"self": {"href": path}}


# Health
@router.get("/health", response_model=dict)
def health() -> Dict[str, str]:
    return {"status": "ok"}



@router.post("/auth/login", response_model=dict)
def login(data: schemas.Login, db: Session = Depends(get_db)) -> Dict[str, Any]:
    user = db.query(models.User).filter(models.User.email == data.email).first()
    if not user or not verify_password(data.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    token = create_access_token({"sub": user.email, "uid": user.id, "role": user.role})
    return {"access_token": token, "token_type": "bearer"}


# Users (Admin only)
@router.get(
    "/users",
    response_model=dict,
    dependencies=[Depends(require_roles(["admin"])), rate_limiter("users:get", per=1, burst=10)],
)
def list_users(db: Session = Depends(get_db)) -> Dict[str, Any]:
    users = db.query(models.User).all()
    items = [
        schemas.UserOut.model_validate(u).model_dump() | {"_links": user_links(u.id)}
        for u in users
    ]
    return {"data": items, "_links": collection_links("/users")}


@router.post(
    "/users",
    response_model=dict,
    dependencies=[Depends(require_roles(["admin"])), rate_limiter("users:post", per=1, burst=5)],
)
def create_user(data: schemas.UserCreate, db: Session = Depends(get_db)) -> Dict[str, Any]:
    user = models.User(
        email=data.email,
        password_hash=hash_password(data.password),
        role=data.role or "user",
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Email already exists")
    db.refresh(user)
    return {
        "data": schemas.UserOut.model_validate(user).model_dump(),
        "_links": user_links(user.id),
    }


@router.get(
    "/users/{uid}",
    response_model=dict,
    dependencies=[Depends(require_roles(["admin"])), rate_limiter("users:get_one", per=1, burst=10)],
)
def get_user(uid: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    u = db.query(models.User).get(uid)
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "data": schemas.UserOut.model_validate(u).model_dump(),
        "_links": user_links(u.id),
    }


@router.put(
    "/users/{uid}",
    response_model=dict,
    dependencies=[Depends(require_roles(["admin"])), rate_limiter("users:put", per=1, burst=5)],
)
def update_user(uid: int, data: schemas.UserUpdate, db: Session = Depends(get_db)) -> Dict[str, Any]:
    u = db.query(models.User).get(uid)
    if not u:
        raise HTTPException(status_code=404, detail="User not found")

    if data.email is not None:
        u.email = data.email
    if data.password is not None:
        u.password_hash = hash_password(data.password)
    if data.role is not None:
        u.role = data.role

    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Email may already exist")
    db.refresh(u)
    return {
        "data": schemas.UserOut.model_validate(u).model_dump(),
        "_links": user_links(u.id),
    }


@router.delete(
    "/users/{uid}",
    response_model=dict,
    dependencies=[Depends(require_roles(["admin"])), rate_limiter("users:delete", per=1, burst=5)],
)
def delete_user(uid: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    u = db.query(models.User).get(uid)
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(u)
    db.commit()
    return {"data": {"deleted": uid}}


# Symptoms (admin create; adjust if you want public)
@router.post(
    "/symptoms",
    response_model=dict,
    dependencies=[Depends(require_roles(["admin"])), rate_limiter("symptoms:post", per=1, burst=10)],
)
def create_symptom(data: schemas.SymptomIn, db: Session = Depends(get_db)) -> Dict[str, Any]:
    s = models.Symptom(code=data.code, name=data.name)
    db.add(s)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Code already exists")
    db.refresh(s)
    return {"data": {"id": s.id, "code": s.code, "name": s.name}}


# Events (anyone with/without token; segment uses provided or token claims)
@router.post(
    "/events",
    response_model=dict,
    dependencies=[rate_limiter("events:post", per=1, burst=15)],
)
def create_event(
    evt: schemas.EventIn,
    db: Session = Depends(get_db),
    claims=Depends(get_current_claims),
) -> Dict[str, Any]:
    role = evt.role or (claims or {}).get("role") or "user"
    seg = make_segment(role=role, geo=evt.geo, age=evt.age)
    e = models.Event(user_id=evt.user_id, payload=evt.payload, segment=seg)
    db.add(e)
    db.commit()
    db.refresh(e)
    return {"data": {"id": e.id, "segment": e.segment}, "_links": event_links(e.id)}


# Analytics (admin only)
@router.get(
    "/analytics/frequent-pairs",
    response_model=dict,
    dependencies=[Depends(require_roles(["admin"])), rate_limiter("analytics:pairs", per=1, burst=8)],
)
def get_frequent_pairs(min_support: int = 5, db: Session = Depends(get_db)) -> Dict[str, Any]:
    events = db.query(models.Event).all()
    event_dicts = [{"payload": e.payload} for e in events]
    pairs = frequent_pairs(event_dicts, min_support)
    return {
        "data": pairs,
        "_links": {"self": {"href": f"/analytics/frequent-pairs?min_support={min_support}"}},
    }


@router.get(
    "/analytics/features",
    response_model=dict,
    dependencies=[Depends(require_roles(["admin"])), rate_limiter("analytics:features", per=1, burst=8)],
)
def get_features(db: Session = Depends(get_db)) -> Dict[str, Any]:
    events = db.query(models.Event).all()
    event_dicts = [{"payload": e.payload} for e in events]
    feats = extract_symptom_features(event_dicts)
    return {"data": feats, "_links": collection_links("/analytics/features")}
