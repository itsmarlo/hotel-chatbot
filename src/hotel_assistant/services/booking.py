from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

from hotel_assistant.database.repository import HotelRepository
from hotel_assistant.dialogue.state import BookingState
from hotel_assistant.domain.models import Reservation, RoomAvailability

_CONFIRMATIONS = frozenset({"confirm booking", "buchung bestätigen", "confirmer la réservation"})
_HOTEL_TIMEZONE = ZoneInfo("Asia/Beirut")


def _hotel_today() -> date:
    return datetime.now(_HOTEL_TIMEZONE).date()


@dataclass(frozen=True, slots=True)
class PriceQuote:
    room_id: int
    nights: int
    nightly_cents: int
    total_cents: int


class BookingWorkflow:
    """Admit booking tools in order; never trust a proposed tool call alone."""

    def __init__(
        self, state: BookingState, repository: HotelRepository, *, today: date | None = None
    ) -> None:
        self.state = state
        self.repository = repository
        self._fixed_today = today
        self._snapshot: dict[str, object] | None = None
        self._rooms: list[RoomAvailability] = []
        self._quote: PriceQuote | None = None
        self._confirmed = False

    def search_rooms(self) -> list[RoomAvailability]:
        self._quote = None
        self._confirmed = False
        self._snapshot = None
        self._rooms = []
        missing = self.state.missing_required_fields()
        if missing:
            raise ValueError(f"booking is missing: {', '.join(missing)}")
        assert self.state.check_in is not None
        assert self.state.check_out is not None
        assert self.state.guests is not None
        if self.state.check_in < (self._fixed_today or _hotel_today()):
            raise ValueError("check-in must not be in the past")
        self._rooms = self.repository.find_available_rooms(
            check_in=self.state.check_in,
            check_out=self.state.check_out,
            guest_count=self.state.guests,
            room_type=self.state.room_type,
        )
        self._snapshot = self.state.to_raw_dict()
        return list(self._rooms)

    def _require_current_search(self) -> None:
        if self._snapshot is None:
            raise ValueError("search for rooms first")
        if self._snapshot != self.state.to_raw_dict():
            self._snapshot = None
            self._quote = None
            self._confirmed = False
            raise ValueError("booking details changed; search for rooms again")

    def calculate_price(self, room_id: int) -> PriceQuote:
        self._require_current_search()
        room = next((item for item in self._rooms if item.room_id == room_id), None)
        if room is None:
            raise ValueError("select a room from the latest search")
        assert self.state.check_in is not None
        assert self.state.check_out is not None
        nights = (self.state.check_out - self.state.check_in).days
        self._quote = PriceQuote(room_id, nights, room.price_per_night_cents,
                                 nights * room.price_per_night_cents)
        self._confirmed = False
        return self._quote

    def confirm(self, user_message: str) -> None:
        self._require_current_search()
        if self._quote is None:
            raise ValueError("calculate a price before confirmation")
        if user_message.strip().casefold() not in _CONFIRMATIONS:
            raise ValueError("explicit confirmation is required")
        self._confirmed = True

    def create_booking(self) -> Reservation:
        self._require_current_search()
        if self._quote is None or not self._confirmed:
            raise ValueError("explicitly confirm the quoted booking first")
        assert self.state.guest_name is not None
        assert self.state.check_in is not None
        assert self.state.check_out is not None
        assert self.state.guests is not None
        if self.state.check_in < (self._fixed_today or _hotel_today()):
            raise ValueError("check-in must not be in the past")
        try:
            reservation = self.repository.create_reservation(
                guest_name=self.state.guest_name,
                room_id=self._quote.room_id,
                check_in=self.state.check_in,
                check_out=self.state.check_out,
                guest_count=self.state.guests,
                expected_nightly_cents=self._quote.nightly_cents,
            )
        except ValueError:
            self._confirmed = False
            self._quote = None
            self._snapshot = None
            raise
        self._confirmed = False
        self._quote = None
        self._snapshot = None
        return reservation
