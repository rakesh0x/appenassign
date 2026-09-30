"""FastAPI app: /chat, /health and the built React UI."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import validate
from app.rag import answer_question

DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"

app = FastAPI(title="Agentic AI Ebook RAG Chatbot", version="1.0.0")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question about the ebook")


class Source(BaseModel):
    page: int
    score: float
    text: str


class ChatResponse(BaseModel):
    answer: str
    confidence: float
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        validate()
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    try:
        result = answer_question(question)
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Upstream call failed: {error}") from error

    return ChatResponse(**result)


if DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="ui")