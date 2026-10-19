from __future__ import annotations

ROOM_TYPES = (
    (1, "Single", "Quiet room with one queen bed.", 1, 12900),
    (2, "Double", "Comfortable room with one king bed or two twins.", 2, 17900),
    (3, "Family", "Two sleeping areas designed for families.", 4, 24900),
    (4, "Suite", "Separate living area with premium amenities.", 3, 32900),
)

ROOMS = (
    (1, "101", 1, 1, 1),
    (2, "102", 1, 1, 1),
    (3, "103", 1, 1, 1),
    (4, "201", 2, 2, 1),
    (5, "202", 2, 2, 1),
    (6, "203", 2, 2, 1),
    (7, "301", 3, 3, 1),
    (8, "302", 3, 3, 1),
    (9, "303", 3, 3, 1),
    (10, "401", 4, 4, 1),
    (11, "402", 4, 4, 1),
    (12, "403", 4, 4, 1),
)

GUESTS = (
    (1, "Amelia Hart", "amelia.hart@example.test", "+49 30 5550101"),
    (2, "Jonas Weber", "jonas.weber@example.test", "+49 30 5550102"),
    (3, "Noor Haddad", "noor.haddad@example.test", "+49 30 5550103"),
)

RESERVATIONS = (
    (1, "HTL-2027-0001", 1, 4, "2027-06-10", "2027-06-15", 2, "CONFIRMED"),
    (2, "HTL-2027-0002", 2, 10, "2027-06-20", "2027-06-25", 2, "CONFIRMED"),
    (3, "HTL-2027-0003", 3, 1, "2027-07-01", "2027-07-05", 1, "CANCELLED"),
)

