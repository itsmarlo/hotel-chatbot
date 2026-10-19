from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from hotel_assistant.api.main import create_app
from hotel_assistant.config import Settings


def test_health_reports_database_readiness(tmp_path: Path) -> None:
    settings = Settings(database_path=tmp_path / "api.db")

    with TestClient(create_app(settings)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "hotel-ai-assistant",
        "version": "0.1.0",
        "database": "ready",
    }


def test_chat_returns_explicit_phase_a_placeholder(tmp_path: Path) -> None:
    settings = Settings(database_path=tmp_path / "api.db")

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/chat",
            json={"message": "I need a room", "session_id": "demo-session"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "demo-session"
    assert "Phase A" in payload["message"]
    assert payload["debug"] == {
        "detected_intent": None,
        "confidence": None,
        "extracted_slots": {},
        "selected_route": "phase_a_placeholder",
        "sources": [],
    }


def test_chat_validates_empty_messages(tmp_path: Path) -> None:
    settings = Settings(database_path=tmp_path / "api.db")

    with TestClient(create_app(settings)) as client:
        response = client.post("/chat", json={"message": "   "})

    assert response.status_code == 422
