from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from hotel_assistant.dialogue.state import BookingState, TurnLimitExceeded


def test_booking_state_tracks_required_fields_in_order() -> None:
    state = BookingState()

    assert state.missing_required_fields() == (
        "guest_name",
        "check_in",
        "check_out",
        "guests",
    )
    assert state.is_complete is False


def test_booking_state_updates_and_corrects_values() -> None:
    state = BookingState()
    state.update(
        guest_name="Sarah Miller",
        check_in=date(2027, 10, 12),
        check_out=date(2027, 10, 15),
        guests=2,
    )

    assert state.is_complete is True

    state.update(check_out=date(2027, 10, 16), guests=3)

    assert state.check_out == date(2027, 10, 16)
    assert state.guests == 3
    assert state.to_dict()["check_out"] == "2027-10-16"


def test_booking_state_rejects_invalid_values() -> None:
    state = BookingState()

    with pytest.raises(ValueError, match="guests must be positive"):
        state.update(guests=0)

    with pytest.raises(ValueError, match="check-out must be after check-in"):
        state.update(
            check_in=date(2027, 10, 15),
            check_out=date(2027, 10, 12),
        )


def test_booking_state_enforces_ten_turn_limit() -> None:
    state = BookingState(max_turns=10)

    for _ in range(10):
        state.record_user_turn()

    assert state.has_reached_turn_limit is True
    with pytest.raises(TurnLimitExceeded, match="10"):
        state.record_user_turn()


def test_booking_state_reset_clears_all_session_data() -> None:
    state = BookingState()
    state.update(guest_name="Alex Kim", guests=2, special_request="Late arrival")
    state.record_user_turn()

    state.reset()

    assert state.to_dict() == {
        "guest_name": None,
        "check_in": None,
        "check_out": None,
        "guests": None,
        "room_type": None,
        "special_request": None,
    }
    assert state.turn_count == 0


def test_booking_state_loads_required_slots_and_limit_from_config() -> None:
    config_path = Path(__file__).parents[2] / "configs" / "booking.json"

    state = BookingState.from_config(config_path)

    assert state.required_slots == ("guest_name", "check_in", "check_out", "guests")
    assert state.max_turns == 10


def test_booking_state_rejects_invalid_direct_configuration() -> None:
    with pytest.raises(ValueError, match="max_turns"):
        BookingState(max_turns=0)

    with pytest.raises(ValueError, match="Unknown required"):
        BookingState(required_slots=("guest_name", "credit_card"))
