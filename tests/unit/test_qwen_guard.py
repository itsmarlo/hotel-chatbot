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
