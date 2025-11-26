# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from sqlalchemy import text

from app.db import Base, engine, SessionLocal
from app.middleware import TrackingMiddleware
from app.routers import router

app = FastAPI(title="Registration API", version="1.0.0")

# CORS: allow the client on port 8001
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8001", "http://localhost:8001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional metrics + your tracking middleware
Instrumentator().instrument(app).expose(app)
app.add_middleware(TrackingMiddleware)

# DB tables (idempotent)
Base.metadata.create_all(bind=engine)

# Routes
app.include_router(router)

@app.get("/health")
def health():
    with SessionLocal() as db:
        db.execute(text("SELECT 1"))
    return {"status": "ok"}

@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
