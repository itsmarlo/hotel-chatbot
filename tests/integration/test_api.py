from __future__ import annotations

from pathlib import Path

import pytest
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


def test_german_booking_accepts_and_displays_german_dates(tmp_path: Path) -> None:
    settings = Settings(database_path=tmp_path / "german.db")
    with TestClient(create_app(settings)) as client:
        session_id = None
        for message in (
            "Ich möchte ein Doppelzimmer buchen",
            "Mein Name ist Alex Kim",
            "Anreise 12.06.2027",
            "Abreise 14.06.2027",
            "2 Gäste",
        ):
            payload = client.post(
                "/chat", json={"message": message, "session_id": session_id}
            ).json()
            session_id = payload["session_id"]
            if message == "Mein Name ist Alex Kim":
                assert "DD.MM.YYYY" in payload["message"]
        assert payload["debug"]["selected_route"] == "quote"
        assert "12.06.2027" in payload["message"]
        assert "14.06.2027" in payload["message"]
        repository = HotelRepository(settings.database_path)
        assert repository.count_rows("reservations") == 3
        confirmed = client.post(
            "/chat", json={"message": "Buchung bestätigen", "session_id": session_id}
        ).json()
        assert confirmed["debug"]["selected_route"] == "booking_created"
        assert repository.count_rows("reservations") == 4


def test_german_booking_explains_invalid_date_in_german(tmp_path: Path) -> None:
    with TestClient(create_app(Settings(database_path=tmp_path / "german_invalid.db"))) as client:
        first = client.post("/chat", json={"message": "Ich möchte ein Zimmer buchen"}).json()
        invalid = client.post(
            "/chat",
            json={"message": "Anreise 31.02.2027", "session_id": first["session_id"]},
        ).json()
        assert "DD.MM.YYYY" in invalid["message"]
        assert invalid["debug"]["selected_route"] == "clarification"


def test_booking_accepts_bare_name_and_contextual_checkout_date(tmp_path: Path) -> None:
    with TestClient(create_app(Settings(database_path=tmp_path / "contextual_slots.db"))) as client:
        first = client.post("/chat", json={"message": "I want to book a room"}).json()
        assert "name" in first["message"].casefold()

        named = client.post(
            "/chat",
            json={"message": "Carlo Escobar", "session_id": first["session_id"]},
        ).json()
        assert "check-in" in named["message"].casefold()

        checkin = client.post(
            "/chat",
            json={"message": "12.12.2027", "session_id": first["session_id"]},
        ).json()
        assert "check-out" in checkin["message"].casefold()

        checkout = client.post(
            "/chat",
            json={"message": "14.12.2027", "session_id": first["session_id"]},
        ).json()
        assert "number of guests" in checkout["message"].casefold()

        guests = client.post(
            "/chat",
            json={"message": "2", "session_id": first["session_id"]},
        ).json()

    assert guests["debug"]["selected_route"] == "quote"
    assert "12.12.2027" in guests["message"]
    assert "14.12.2027" in guests["message"]


def test_german_fallback_uses_german_date_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOTEL_NLP_MODE", "off")
    with TestClient(create_app(Settings(database_path=tmp_path / "german_guidance.db"))) as client:
        payload = client.post("/chat", json={"message": "Ich brauche Hilfe"}).json()

    assert payload["debug"]["selected_route"] == "guidance"
    assert "DD.MM.YYYY" in payload["message"]
    assert "YYYY-MM-DD" not in payload["message"]


def test_english_fallback_also_requests_display_date_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOTEL_NLP_MODE", "off")
    with TestClient(create_app(Settings(database_path=tmp_path / "english_guidance.db"))) as client:
        payload = client.post("/chat", json={"message": "Help"}).json()

    assert payload["debug"]["selected_route"] == "guidance"
    assert "DD.MM.YYYY" in payload["message"]
    assert "YYYY-MM-DD" not in payload["message"]


def test_declining_quote_discards_booking_without_reservation(tmp_path: Path) -> None:
    settings = Settings(database_path=tmp_path / "api.db")
    with TestClient(create_app(settings)) as client:
        first = client.post(
            "/chat",
            json={
                "message": "Book a double room 2027-06-12 to 2027-06-14 for 2 guests. "
                "My name is Alex Kim"
            },
        ).json()
        assert first["debug"]["selected_route"] == "quote"
        declined = client.post(
            "/chat", json={"message": "don't book", "session_id": first["session_id"]}
        ).json()
    assert declined["debug"]["selected_route"] == "booking_discarded"
    assert HotelRepository(settings.database_path).count_rows("reservations") == 3


def test_chat_validates_empty_messages(tmp_path: Path) -> None:
    settings = Settings(database_path=tmp_path / "api.db")

    with TestClient(create_app(settings)) as client:
        response = client.post("/chat", json={"message": "   "})

    assert response.status_code == 422


def test_trained_intent_model_routes_without_creating_reservations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The multilingual model can identify actions; it must not write bookings.
    monkeypatch.setenv("HOTEL_RAG_MODE", "semantic")
    monkeypatch.setenv("HOTEL_RAG_LOCAL_ONLY", "1")
    settings = Settings(database_path=tmp_path / "semantic.db")
    with TestClient(create_app(settings)) as client:
        cancel = client.post("/chat", json={"message": "Cancel my reservation"}).json()
        book = client.post("/chat", json={"message": "Je veux réserver une chambre"}).json()
    assert cancel["debug"]["selected_route"] == "unsupported_reservation_change"
    assert cancel["debug"]["confidence"] >= 0.7
    assert book["debug"]["selected_route"] == "clarification"
    assert book["debug"]["detected_intent"] == "BOOKING"
    assert HotelRepository(settings.database_path).count_rows("reservations") == 3
