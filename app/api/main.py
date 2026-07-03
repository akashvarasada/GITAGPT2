"""FastAPI entrypoint.  Run:  uvicorn app.api.main:app --reload"""
from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(title="GITAGPT", description="RAG over the Bhagavad-gita As It Is")
app.include_router(router)
