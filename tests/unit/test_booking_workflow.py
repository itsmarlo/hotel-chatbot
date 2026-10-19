from datetime import date
from pathlib import Path

import pytest

from hotel_assistant.database.repository import HotelRepository
from hotel_assistant.dialogue.state import BookingState
from hotel_assistant.services.booking import BookingWorkflow


@pytest.fixture
def workflow(tmp_path: Path) -> BookingWorkflow:
    repository = HotelRepository(tmp_path / "hotel.db")
    repository.initialize()
    repository.seed()
    state = BookingState()
    state.update(
        guest_name="Alex Kim",
        check_in=date(2027, 6, 12),
        check_out=date(2027, 6, 14),
        guests=2,
        room_type="Double",
    )
    return BookingWorkflow(state, repository, today=date(2026, 10, 19))


def test_booking_requires_search_price_and_explicit_confirmation(workflow: BookingWorkflow) -> None:
    with pytest.raises(ValueError, match="search"):
        workflow.calculate_price(5)
    with pytest.raises(ValueError, match="search"):
        workflow.create_booking()

    rooms = workflow.search_rooms()
    assert [room.room_number for room in rooms] == ["202", "203"]
    quote = workflow.calculate_price(rooms[0].room_id)
    assert quote.total_cents == 35800
    assert quote.nights == 2

    with pytest.raises(ValueError, match="confirm"):
        workflow.create_booking()
    with pytest.raises(ValueError, match="explicit"):
        workflow.confirm("sure")

    workflow.confirm("confirm booking")
    reservation = workflow.create_booking()
    assert reservation.status == "CONFIRMED"
    assert reservation.room_id == rooms[0].room_id
    assert workflow.repository.get_reservation(reservation.reference) == reservation
    with pytest.raises(ValueError, match="search"):
        workflow.create_booking()


def test_correction_invalidates_price_and_confirmation(workflow: BookingWorkflow) -> None:
    room = workflow.search_rooms()[0]
    workflow.calculate_price(room.room_id)
    workflow.confirm("confirm booking")
    workflow.state.update(check_out=date(2027, 6, 15))

    with pytest.raises(ValueError, match="changed"):
        workflow.create_booking()
    with pytest.raises(ValueError, match="search"):
        workflow.calculate_price(room.room_id)
    assert workflow.repository.count_rows("reservations") == 3


def test_incomplete_and_invalid_requests_have_no_side_effects(workflow: BookingWorkflow) -> None:
    workflow.state.update(guests=None)
    with pytest.raises(ValueError, match="missing"):
        workflow.search_rooms()
    with pytest.raises(ValueError, match="positive"):
        workflow.state.update(guests=0)
    with pytest.raises(ValueError, match="after"):
        workflow.state.update(check_out=date(2027, 6, 11))
    assert workflow.repository.count_rows("reservations") == 3


def test_second_booking_cannot_take_room_after_quote(workflow: BookingWorkflow) -> None:
    room = workflow.search_rooms()[0]
    quote = workflow.calculate_price(room.room_id)
    workflow.confirm("confirm booking")
    workflow.repository.create_reservation(
        guest_name="Other Guest",
        room_id=room.room_id,
        check_in=workflow.state.check_in,
        check_out=workflow.state.check_out,
        guest_count=2,
        expected_nightly_cents=quote.nightly_cents,
    )
    with pytest.raises(ValueError, match="available"):
        workflow.create_booking()
    with pytest.raises(ValueError, match="search"):
        workflow.create_booking()
    assert workflow.repository.count_rows("reservations") == 4


def test_price_change_requires_a_new_quote(workflow: BookingWorkflow) -> None:
    room = workflow.search_rooms()[0]
    workflow.calculate_price(room.room_id)
    workflow.confirm("confirm booking")
    with workflow.repository.connection() as connection:
        connection.execute(
            "UPDATE room_types SET price_per_night_cents = 19900 WHERE name = 'Double'"
        )

    with pytest.raises(ValueError, match="price changed"):
        workflow.create_booking()
    with pytest.raises(ValueError, match="search"):
        workflow.create_booking()
    assert workflow.repository.count_rows("reservations") == 3


def test_past_check_in_is_rejected(workflow: BookingWorkflow) -> None:
    workflow.state.update(check_in=date(2026, 10, 18), check_out=date(2026, 10, 20))
    with pytest.raises(ValueError, match="past"):
        workflow.search_rooms()
    assert workflow.repository.count_rows("reservations") == 3
