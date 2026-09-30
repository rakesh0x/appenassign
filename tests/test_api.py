"""Health, input validation, and one answerable + one unanswerable question.

The last two are skipped when API keys are not configured.
"""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

has_keys = bool(os.getenv("PINECONE_API_KEY") and os.getenv("GROQ_API_KEY"))
needs_live_api = pytest.mark.skipif(
    not has_keys, reason="Requires PINECONE_API_KEY and GROQ_API_KEY plus a finished ingestion."
)

ui_built = (Path(__file__).resolve().parent.parent / "frontend" / "dist").exists()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.skipif(not ui_built, reason="Run `npm run build` in frontend/ first.")
def test_built_react_ui_is_served_at_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Agentic AI Ebook" in response.text


def test_empty_question_is_rejected():
    response = client.post("/chat", json={"question": "   "})
    assert response.status_code == 400


def test_missing_question_field_is_rejected():
    assert client.post("/chat", json={}).status_code == 422


@needs_live_api
def test_question_answerable_from_the_ebook():
    response = client.post("/chat", json={"question": "What is a multi-agent system?"})
    body = response.json()

    assert response.status_code == 200
    assert body["sources"], "expected retrieved chunks"
    assert body["confidence"] > 0.3
    assert "couldn't find" not in body["answer"].lower()


@needs_live_api
def test_question_outside_the_ebook_is_refused():
    response = client.post("/chat", json={"question": "What is the capital of France?"})
    body = response.json()

    assert response.status_code == 200
    assert body["answer"] == "I couldn't find this information in the provided ebook."
