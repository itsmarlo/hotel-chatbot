from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from uuid import uuid4

from hotel_assistant.database.seed import GUESTS, RESERVATIONS, ROOM_TYPES, ROOMS
from hotel_assistant.domain import Reservation, RoomAvailability

_COUNTABLE_TABLES = frozenset({"room_types", "rooms", "guests", "reservations"})


class HotelRepository:
    """Parameterized SQLite access for hotel inventory and reservations."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
        with self.connection() as connection:
            connection.executescript(schema)

    def seed(self) -> None:
        """Insert deterministic synthetic data without duplicating rows."""
        with self.connection() as connection:
            connection.executemany(
                """INSERT OR IGNORE INTO room_types
                   (id, name, description, max_occupancy, price_per_night_cents)
                   VALUES (?, ?, ?, ?, ?)""",
                ROOM_TYPES,
            )
            connection.executemany(
                """INSERT OR IGNORE INTO rooms
                   (id, room_number, room_type_id, floor, is_active)
                   VALUES (?, ?, ?, ?, ?)""",
                ROOMS,
            )
            connection.executemany(
                """INSERT OR IGNORE INTO guests (id, full_name, email, phone)
                   VALUES (?, ?, ?, ?)""",
                GUESTS,
            )
            connection.executemany(
                """INSERT OR IGNORE INTO reservations
                   (id, reference, guest_id, room_id, check_in, check_out, guest_count, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                RESERVATIONS,
            )

    def find_available_rooms(
        self,
        *,
        check_in: date,
        check_out: date,
        guest_count: int,
        room_type: str | None = None,
    ) -> list[RoomAvailability]:
        if check_out <= check_in:
            raise ValueError("check_out must be after check_in")
        if guest_count < 1:
            raise ValueError("guest_count must be positive")

        query = """
            SELECT r.id, r.room_number, rt.name AS room_type,
                   rt.max_occupancy, rt.price_per_night_cents
            FROM rooms AS r
            JOIN room_types AS rt ON rt.id = r.room_type_id
            WHERE r.is_active = 1
              AND rt.max_occupancy >= ?
              AND (? IS NULL OR rt.name = ? COLLATE NOCASE)
              AND NOT EXISTS (
                  SELECT 1
                  FROM reservations AS reservation
                  WHERE reservation.room_id = r.id
                    AND reservation.status IN ('CONFIRMED', 'CHECKED_IN')
                    AND reservation.check_in < ?
                    AND reservation.check_out > ?
              )
            ORDER BY rt.price_per_night_cents, r.room_number
        """
        with self.connection() as connection:
            rows = connection.execute(
                query,
                (guest_count, room_type, room_type, check_out.isoformat(), check_in.isoformat()),
            ).fetchall()
        return [
            RoomAvailability(
                room_id=row["id"],
                room_number=row["room_number"],
                room_type=row["room_type"],
                max_occupancy=row["max_occupancy"],
                price_per_night_cents=row["price_per_night_cents"],
            )
            for row in rows
        ]

    def get_reservation(self, reference: str) -> Reservation:
        with self.connection() as connection:
            row = connection.execute(
                """SELECT reference, guest_id, room_id, check_in, check_out,
                          guest_count, status
                   FROM reservations WHERE reference = ?""",
                (reference,),
            ).fetchone()
        if row is None:
            raise LookupError(f"Reservation not found: {reference}")
        return Reservation(
            reference=row["reference"],
            guest_id=row["guest_id"],
            room_id=row["room_id"],
            check_in=date.fromisoformat(row["check_in"]),
            check_out=date.fromisoformat(row["check_out"]),
            guest_count=row["guest_count"],
            status=row["status"],
        )

    def create_reservation(
        self,
        *,
        guest_name: str,
        room_id: int,
        check_in: date,
        check_out: date,
        guest_count: int,
        expected_nightly_cents: int,
    ) -> Reservation:
        """Recheck availability and quoted price inside a write transaction."""
        if not guest_name.strip():
            raise ValueError("guest_name is required")
        if check_out <= check_in:
            raise ValueError("check_out must be after check_in")
        if guest_count < 1:
            raise ValueError("guest_count must be positive")

        reference = f"HTL-{uuid4().hex[:12].upper()}"
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            room = connection.execute(
                """SELECT rt.max_occupancy, rt.price_per_night_cents
                   FROM rooms AS r JOIN room_types AS rt ON rt.id = r.room_type_id
                   WHERE r.id = ? AND r.is_active = 1""",
                (room_id,),
            ).fetchone()
            if room is None or room["max_occupancy"] < guest_count:
                raise ValueError("room is not available for this guest count")
            if room["price_per_night_cents"] != expected_nightly_cents:
                raise ValueError("room price changed; request a new quote")
            overlap = connection.execute(
                """SELECT 1 FROM reservations
                   WHERE room_id = ? AND status IN ('CONFIRMED', 'CHECKED_IN')
                     AND check_in < ? AND check_out > ? LIMIT 1""",
                (room_id, check_out.isoformat(), check_in.isoformat()),
            ).fetchone()
            if overlap is not None:
                raise ValueError("room is no longer available")
            # No real email is collected in this academic prototype.
            guest = connection.execute(
                "INSERT INTO guests (full_name, email) VALUES (?, ?)",
                (guest_name.strip(), f"{uuid4().hex}@example.invalid"),
            )
            connection.execute(
                """INSERT INTO reservations
                   (reference, guest_id, room_id, check_in, check_out, guest_count, status)
                   VALUES (?, ?, ?, ?, ?, ?, 'CONFIRMED')""",
                (
                    reference,
                    guest.lastrowid,
                    room_id,
                    check_in.isoformat(),
                    check_out.isoformat(),
                    guest_count,
                ),
            )
        return self.get_reservation(reference)

    def cancel_reservation(self, reference: str) -> bool:
        with self.connection() as connection:
            cursor = connection.execute(
                """UPDATE reservations
                   SET status = 'CANCELLED', updated_at = CURRENT_TIMESTAMP
                   WHERE reference = ? AND status IN ('CONFIRMED', 'CHECKED_IN')""",
                (reference,),
            )
        return cursor.rowcount == 1

    def count_rows(self, table: str) -> int:
        """Count seed rows; the whitelist prevents identifier injection."""
        if table not in _COUNTABLE_TABLES:
            raise ValueError(f"Unsupported table: {table}")
        with self.connection() as connection:
            row = connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
        assert row is not None
        return int(row["count"])
