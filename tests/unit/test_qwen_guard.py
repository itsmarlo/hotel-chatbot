from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hotel_assistant.api.schemas import PipelineDebug
from hotel_assistant.database.repository import HotelRepository
from hotel_assistant.ml.qwen import BookingProposal
from hotel_assistant.services.chat import ChatService


class StubQwen:
    def __init__(self, proposal: BookingProposal) -> None:
        self.proposal = proposal

    def interpret(self, _message: str, _state: dict[str, Any]) -> BookingProposal:
        return self.proposal


def proposal(**changes: object) -> BookingProposal:
    data: dict[str, object] = {
        "intent": "BOOKING",
        "guest_name": "",
        "check_in": "",
        "check_out": "",
        "guests": 0,
        "room_type": "",
        "special_request": "",
        "proposed_tool": "create_booking",
    }
    data.update(changes)
    return BookingProposal.model_validate(data)


def test_qwen_tool_proposal_cannot_skip_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOTEL_NLP_MODE", "qwen")
    repository = HotelRepository(tmp_path / "qwen.db")
    service = ChatService(repository=repository, qwen=StubQwen(proposal(guests=2)))  # type: ignore[arg-type]
    result = service.respond(
        "Book a double room 2030-10-01 to 2030-10-02 for 2 guests. My name is Alex Kim."
    )
    assert result.debug["proposed_tool"] == "create_booking"
    assert result.debug["selected_route"] == "quote"
    PipelineDebug.model_validate(result.debug)
    assert repository.count_rows("reservations") == 3


def test_qwen_date_disagreement_requires_clarification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOTEL_NLP_MODE", "qwen")
    repository = HotelRepository(tmp_path / "qwen.db")
    service = ChatService(
        repository=repository,
        qwen=StubQwen(proposal(check_in="2030-10-03")),  # type: ignore[arg-type]
    )
    result = service.respond(
        "Book a room 2030-10-01 to 2030-10-02 for 2 guests. My name is Alex Kim."
    )
    assert result.debug["selected_route"] == "clarification"
    assert result.booking_state is not None
    assert result.booking_state.check_in is None
    assert repository.count_rows("reservations") == 3


def test_qwen_agreed_german_dates_are_admitted_as_normalized_iso(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOTEL_NLP_MODE", "qwen")
    repository = HotelRepository(tmp_path / "qwen_german.db")
    service = ChatService(
        repository=repository,
        qwen=StubQwen(  # type: ignore[arg-type]
            proposal(check_in="2030-10-01", check_out="2030-10-02", guests=2)
        ),
    )
    result = service.respond(
        "Ich möchte ein Zimmer vom 01.10.2030 bis 02.10.2030 für 2 Gäste buchen. "
        "Mein Name ist Alex Kim."
    )
    assert result.debug["selected_route"] == "quote"
    assert "01.10.2030" in result.message
    assert repository.count_rows("reservations") == 3


def test_policy_question_bypasses_booking_proposal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOTEL_NLP_MODE", "qwen")
    repository = HotelRepository(tmp_path / "qwen.db")
    service = ChatService(repository=repository, qwen=StubQwen(proposal()))  # type: ignore[arg-type]
    result = service.respond("When is check-in?")
    assert result.debug["selected_route"] == "policy"
    assert result.debug["sources"] == ["check-in"]
    assert repository.count_rows("reservations") == 3


def test_unsupported_question_cannot_become_booking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOTEL_NLP_MODE", "qwen")
    repository = HotelRepository(tmp_path / "qwen.db")
    service = ChatService(repository=repository, qwen=StubQwen(proposal()))  # type: ignore[arg-type]
    result = service.respond("Do you offer scuba lessons?")
    assert result.debug["selected_route"] == "unsupported_policy"
    assert repository.count_rows("reservations") == 3


def test_llm_rephrases_supported_policy_and_keeps_source(tmp_path: Path) -> None:
    seen: list[tuple[str, str, str, str]] = []

    def generate(message: str, fact: str, language: str, route: str) -> str:
        seen.append((message, fact, language, route))
        return (
            "You can check in from 15:00 and check out by 11:00. "
            "Please let reception know if you'll arrive late."
        )

    service = ChatService(
        repository=HotelRepository(tmp_path / "llm_policy.db"),
        answer_generator=generate,
    )
    result = service.respond("When is check-in?")

    assert result.message.startswith("You can check in from 15:00 and check out by 11:00.")
    assert result.message.endswith("Source: Hotel policy [check-in]")
    assert result.debug["response_mode"] == "llm"
    assert seen[0][2:] == ("en", "policy")


def test_llm_cannot_change_quote_or_claim_confirmation(tmp_path: Path) -> None:
    def generate(*_args: str) -> str:
        return "Your booking is confirmed."

    repository = HotelRepository(tmp_path / "llm_booking.db")
    service = ChatService(repository=repository, answer_generator=generate)
    result = service.respond(
        "Book a double room 2030-10-01 to 2030-10-02 for 2 guests. My name is Alex Kim."
    )

    assert result.debug["selected_route"] == "quote"
    assert result.debug["response_mode"] == "template"
    assert "confirm booking" in result.message
    assert repository.count_rows("reservations") == 3


def test_llm_bad_reply_falls_back_to_safe_clarification(tmp_path: Path) -> None:
    calls: list[str] = []

    def generate(*_args: str) -> str:
        calls.append("called")
        return "Your reservation is confirmed."

    service = ChatService(
        repository=HotelRepository(tmp_path / "llm_clarification.db"),
        answer_generator=generate,
    )
    result = service.respond("I want to book a room")

    assert result.debug["response_mode"] == "template"
    assert "full name" in result.message
    assert calls == []


def test_llm_cannot_merge_checkin_and_checkout_facts(tmp_path: Path) -> None:
    service = ChatService(
        repository=HotelRepository(tmp_path / "llm_times.db"),
        answer_generator=lambda *_: "Check-in begins at 15:00 and ends by 11:00.",
    )
    result = service.respond("When is check-in?")

    assert result.debug["response_mode"] == "template"
    assert "check-out is by 11:00" in result.message


def test_llm_cannot_request_extra_booking_fields(tmp_path: Path) -> None:
    service = ChatService(
        repository=HotelRepository(tmp_path / "llm_extra_slot.db"),
        answer_generator=lambda *_: "Bitte nennen Sie Ihren Namen und Ihr Reisedatum.",
    )
    result = service.respond("Ich möchte ein Zimmer buchen")

    assert result.debug["response_mode"] == "template"
    assert "vollständiger Name" in result.message
    assert "Datum" not in result.message
