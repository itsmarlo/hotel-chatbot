from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum


class ReservationStatus(StrEnum):
    """Lifecycle states persisted for a reservation."""

    CONFIRMED = "CONFIRMED"
    CHECKED_IN = "CHECKED_IN"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True, slots=True)
class RoomAvailability:
    """A room matching a guest's availability search."""

    room_id: int
    room_number: str
    room_type: str
    max_occupancy: int
    price_per_night_cents: int


@dataclass(frozen=True, slots=True)
class Reservation:
    """Reservation data returned by the persistence boundary."""

    reference: str
    guest_id: int
    room_id: int
    check_in: date
    check_out: date
    guest_count: int
    status: str


