from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

_STATE_FIELDS = frozenset(
    {"guest_name", "check_in", "check_out", "guests", "room_type", "special_request"}
)
_DEFAULT_REQUIRED_SLOTS = ("guest_name", "check_in", "check_out", "guests")


class TurnLimitExceeded(RuntimeError):
    """Raised when a booking conversation exceeds its configured user-turn limit."""


@dataclass(slots=True)
class BookingState:
    """Deterministic, per-session source of truth for a prototype booking request."""

    guest_name: str | None = None
    check_in: date | None = None
    check_out: date | None = None
    guests: int | None = None
    room_type: str | None = None
    special_request: str | None = None
    turn_count: int = 0
    max_turns: int = 30
    required_slots: tuple[str, ...] = field(default=_DEFAULT_REQUIRED_SLOTS)

    def __post_init__(self) -> None:
        unknown_slots = set(self.required_slots) - _STATE_FIELDS
        if unknown_slots:
            raise ValueError(f"Unknown required booking slots: {sorted(unknown_slots)}")
        if len(self.required_slots) != len(set(self.required_slots)):
            raise ValueError("required_slots must not contain duplicates")
        if (
            not isinstance(self.max_turns, int)
            or isinstance(self.max_turns, bool)
            or self.max_turns < 1
        ):
            raise ValueError("max_turns must be a positive integer")

    @classmethod
    def from_config(cls, path: Path) -> BookingState:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        required_slots = tuple(payload["required_slots"])
        max_turns = payload.get("max_turns", 30)
        return cls(required_slots=required_slots, max_turns=max_turns)

    def update(self, **changes: object) -> None:
        """Atomically apply extracted and normalized slot values, including corrections."""
        unknown_slots = set(changes) - _STATE_FIELDS
        if unknown_slots:
            raise ValueError(f"Unknown booking slots: {sorted(unknown_slots)}")

        candidate = self.to_raw_dict()
        candidate.update(changes)
        self._validate(candidate)
        for slot, value in changes.items():
            setattr(self, slot, value)

    def record_user_turn(self) -> None:
        if self.turn_count >= self.max_turns:
            raise TurnLimitExceeded(
                f"Booking conversations are limited to {self.max_turns} user turns"
            )
        self.turn_count += 1

    @property
    def has_reached_turn_limit(self) -> bool:
        return self.turn_count >= self.max_turns

    def missing_required_fields(self) -> tuple[str, ...]:
        return tuple(slot for slot in self.required_slots if getattr(self, slot) is None)

    @property
    def is_complete(self) -> bool:
        return not self.missing_required_fields()

    def reset(self) -> None:
        for slot in _STATE_FIELDS:
            setattr(self, slot, None)
        self.turn_count = 0

    def to_raw_dict(self) -> dict[str, object]:
        return {
            "guest_name": self.guest_name,
            "check_in": self.check_in,
            "check_out": self.check_out,
            "guests": self.guests,
            "room_type": self.room_type,
            "special_request": self.special_request,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            key: value.isoformat() if isinstance(value, date) else value
            for key, value in self.to_raw_dict().items()
        }

    @staticmethod
    def _validate(candidate: dict[str, object]) -> None:
        for slot in ("guest_name", "room_type", "special_request"):
            value = candidate[slot]
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{slot} must be a non-empty string")

        guests = candidate["guests"]
        if guests is not None and (
            not isinstance(guests, int) or isinstance(guests, bool) or guests < 1
        ):
            raise ValueError("guests must be positive")

        for slot in ("check_in", "check_out"):
            value = candidate[slot]
            if value is not None and not isinstance(value, date):
                raise ValueError(f"{slot} must be a date")

        check_in = candidate["check_in"]
        check_out = candidate["check_out"]
        if isinstance(check_in, date) and isinstance(check_out, date) and check_out <= check_in:
            raise ValueError("check-out must be after check-in")
