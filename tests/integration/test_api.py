from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from hotel_assistant.api.main import create_app
from hotel_assistant.config import Settings
from hotel_assistant.database.repository import HotelRepository


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
        "retrieval": "lexical",
    }


def test_chat_answers_policy_with_source(tmp_path: Path) -> None:
    settings = Settings(database_path=tmp_path / "api.db")

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/chat",
            json={"message": "When is check-in?"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"]
    assert "15:00" in payload["message"]
    assert payload["debug"]["selected_route"] == "policy"
    assert payload["debug"]["sources"] == ["check-in"]


def test_chat_booking_requires_confirmation(tmp_path: Path) -> None:
    settings = Settings(database_path=tmp_path / "api.db")
    with TestClient(create_app(settings)) as client:
        session_id = None
        for message in (
            "I want to book a double room",
            "my name is Alex Kim",
            "2027-06-12 to 2027-06-14",
            "2 guests",
        ):
            response = client.post("/chat", json={"message": message, "session_id": session_id})
            assert response.status_code == 200
            payload = response.json()
            session_id = payload["session_id"]
        assert payload["debug"]["selected_route"] == "quote"
        assert "confirm booking" in payload["message"]
        repository = HotelRepository(settings.database_path)
        assert repository.count_rows("reservations") == 3
        response = client.post(
            "/chat", json={"message": "confirm booking", "session_id": session_id}
        )
        assert response.json()["debug"]["selected_route"] == "booking_created"
        assert "HTL-" in response.json()["message"]
        assert repository.count_rows("reservations") == 4


def test_chat_declines_unknown_policy_without_source(tmp_path: Path) -> None:
    with TestClient(create_app(Settings(database_path=tmp_path / "api.db"))) as client:
        response = client.post("/chat", json={"message": "Do you offer scuba lessons?"})
    assert response.status_code == 200
    assert response.json()["debug"]["selected_route"] == "unsupported_policy"
    assert response.json()["debug"]["sources"] == []


def test_unknown_bathroom_question_is_not_treated_as_booking(tmp_path: Path) -> None:
    with TestClient(create_app(Settings(database_path=tmp_path / "api.db"))) as client:
        response = client.post("/chat", json={"message": "Is there a bathroom?"})
    assert response.json()["debug"]["selected_route"] == "unsupported_policy"


def test_french_booking_keeps_language_across_date_turns(tmp_path: Path) -> None:
    with TestClient(create_app(Settings(database_path=tmp_path / "api.db"))) as client:
        session_id = None
        for message in (
            "Je veux réserver une chambre double",
            "je m'appelle Alex Kim",
            "2027-06-12 au 2027-06-14",
            "2 personnes",
        ):
            payload = client.post(
                "/chat", json={"message": message, "session_id": session_id}
            ).json()
            session_id = payload["session_id"]
        assert payload["debug"]["selected_route"] == "quote"
        assert "confirmer la réservation" in payload["message"]
        confirmed = client.post(
            "/chat", json={"message": "confirmer la réservation", "session_id": session_id}
        ).json()
        assert "Réservation confirmée" in confirmed["message"]


def test_chat_validates_empty_messages(tmp_path: Path) -> None:
    settings = Settings(database_path=tmp_path / "api.db")

    with TestClient(create_app(settings)) as client:
        response = client.post("/chat", json={"message": "   "})

    assert response.status_code == 422
