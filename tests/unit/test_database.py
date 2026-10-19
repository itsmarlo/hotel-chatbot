from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from hotel_assistant.database.repository import HotelRepository


@pytest.fixture
def repository(tmp_path: Path) -> HotelRepository:
    repository = HotelRepository(tmp_path / "hotel.db")
    repository.initialize()
    repository.seed()
    return repository


def test_initialization_and_seed_are_idempotent(repository: HotelRepository) -> None:
    repository.initialize()
    repository.seed()

    assert repository.count_rows("room_types") == 4
    assert repository.count_rows("rooms") == 12
    assert repository.count_rows("guests") == 3
    assert repository.count_rows("reservations") == 3


def test_availability_excludes_overlapping_active_reservation(
    repository: HotelRepository,
) -> None:
    available = repository.find_available_rooms(
        check_in=date(2027, 6, 12),
        check_out=date(2027, 6, 14),
        guest_count=2,
        room_type="Double",
    )

    assert [room.room_number for room in available] == ["202", "203"]
    assert all(room.max_occupancy >= 2 for room in available)


def test_cancelled_reservations_do_not_block_availability(repository: HotelRepository) -> None:
    available = repository.find_available_rooms(
        check_in=date(2027, 7, 2),
        check_out=date(2027, 7, 4),
        guest_count=1,
        room_type="Single",
    )

    assert [room.room_number for room in available] == ["101", "102", "103"]


def test_availability_rejects_invalid_date_range(repository: HotelRepository) -> None:
    with pytest.raises(ValueError, match="check_out must be after check_in"):
        repository.find_available_rooms(
            check_in=date(2027, 6, 14),
            check_out=date(2027, 6, 14),
            guest_count=2,
        )


def test_cancel_reservation_changes_status(repository: HotelRepository) -> None:
    assert repository.cancel_reservation("HTL-2027-0001") is True
    assert repository.get_reservation("HTL-2027-0001").status == "CANCELLED"
    assert repository.cancel_reservation("missing-reference") is False


def test_database_enforces_foreign_keys(repository: HotelRepository) -> None:
    with pytest.raises(sqlite3.IntegrityError), repository.connection() as connection:
        connection.execute(
            "INSERT INTO rooms (room_number, room_type_id, floor, is_active) "
            "VALUES (?, ?, ?, ?)",
            ("999", 999, 9, 1),
        )


def test_count_rows_rejects_unknown_table(repository: HotelRepository) -> None:
    with pytest.raises(ValueError, match="Unsupported table"):
        repository.count_rows("rooms; DROP TABLE rooms")
