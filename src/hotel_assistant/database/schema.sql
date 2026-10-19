PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS room_types (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
    description TEXT NOT NULL,
    max_occupancy INTEGER NOT NULL CHECK (max_occupancy > 0),
    price_per_night_cents INTEGER NOT NULL CHECK (price_per_night_cents > 0)
);

CREATE TABLE IF NOT EXISTS rooms (
    id INTEGER PRIMARY KEY,
    room_number TEXT NOT NULL UNIQUE,
    room_type_id INTEGER NOT NULL REFERENCES room_types(id) ON DELETE RESTRICT,
    floor INTEGER NOT NULL CHECK (floor >= 0),
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1))
);

CREATE TABLE IF NOT EXISTS guests (
    id INTEGER PRIMARY KEY,
    full_name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
    phone TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS reservations (
    id INTEGER PRIMARY KEY,
    reference TEXT NOT NULL UNIQUE,
    guest_id INTEGER NOT NULL REFERENCES guests(id) ON DELETE RESTRICT,
    room_id INTEGER NOT NULL REFERENCES rooms(id) ON DELETE RESTRICT,
    check_in TEXT NOT NULL,
    check_out TEXT NOT NULL,
    guest_count INTEGER NOT NULL CHECK (guest_count > 0),
    status TEXT NOT NULL CHECK (
        status IN ('CONFIRMED', 'CHECKED_IN', 'COMPLETED', 'CANCELLED')
    ),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (check_out > check_in)
);

CREATE INDEX IF NOT EXISTS idx_rooms_room_type ON rooms(room_type_id);
CREATE INDEX IF NOT EXISTS idx_reservations_room_dates
    ON reservations(room_id, check_in, check_out);
CREATE INDEX IF NOT EXISTS idx_reservations_guest ON reservations(guest_id);
CREATE INDEX IF NOT EXISTS idx_reservations_status ON reservations(status);

