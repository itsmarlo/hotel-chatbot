from __future__ import annotations

import re
from datetime import date

from hotel_assistant.dialogue.state import BookingState

_DATE = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{2}\.\d{2}\.\d{4})\b")
_GUESTS = re.compile(r"\b(\d+)\s*(?:guests?|people|persons?|personen|gäste|personnes?)\b", re.I)
_NAME = re.compile(
    r"(?:my name is|name is|name:|ich heiße|mein name ist|je m'appelle|mon nom est)\s+"
    r"([\wÀ-ÿ]+(?:[ '-][\wÀ-ÿ]+){0,3})",
    re.I,
)
_ROOMS = {
    "single": ("single", "einzelzimmer", "individuelle"),
    "double": ("double", "doppelzimmer", "chambre double"),
    "family": ("family", "familienzimmer", "familiale"),
    "suite": ("suite",),
}


def _parse_explicit_date(value: str) -> date:
    if "." in value:
        day, month, year = map(int, value.split("."))
        return date(year, month, day)
    return date.fromisoformat(value)


def language_of(message: str) -> str:
    lower = message.casefold()
    if any(
        word in lower
        for word in (
            "bonjour",
            "je m'appelle",
            "papiers",
            "puis-je",
            "quels",
            "quelle",
            "facturez",
            "réserv",
            "chambre",
            "combien",
            "arrivée",
            "départ",
            "passeport",
            "générateur",
        )
    ):
        return "fr"
    if any(
        word in lower
        for word in (
            "hallo", "ich heisse", "mein name", "zimmer", "buchen", "gäste",
            "ausweis", "strom", "ankunft"
        )
    ):
        return "de"
    if re.search(r"\b(?:ich|möchte|bitte|datum|daten|anreise|abreise|für|buchung)\b", lower):
        return "de"
    return "en"


def booking_changes(message: str) -> dict[str, object]:
    """Extract explicit fields, normalizing ISO and German dates to date values."""
    lower = message.casefold()
    changes: dict[str, object] = {}
    dates = _DATE.findall(message)
    if len(dates) > 2:
        raise ValueError("Please provide at most two dates.")
    try:
        parsed = [_parse_explicit_date(value) for value in dates]
    except ValueError as exc:
        raise ValueError("Please use valid dates in DD.MM.YYYY format.") from exc
    if len(parsed) == 2:
        changes["check_in"], changes["check_out"] = parsed
    elif parsed:
        departure_words = ("check-out", "checkout", "check out", "departure", "abreise", "départ")
        changes["check_out" if any(word in lower for word in departure_words) else "check_in"] = (
            parsed[0]
        )
    guests = _GUESTS.search(message)
    if guests:
        changes["guests"] = int(guests.group(1))
    name = _NAME.search(message)
    if name:
        changes["guest_name"] = re.split(r"\b(?:and|und|et)\b|[,.;!?]", name.group(1))[0].strip()
    for room_type, aliases in _ROOMS.items():
        if any(re.search(rf"\b{re.escape(alias)}\b", lower) for alias in aliases):
            changes["room_type"] = room_type.capitalize()
            break
    return changes


def is_booking_request(message: str, state: BookingState, changes: dict[str, object]) -> bool:
    lower = message.casefold()
    verbs = (
        "book",
        "booking",
        "reserve",
        "reservation",
        "room",
        "rooms",
        "zimmer",
        "buchen",
        "réserver",
        "réservation",
        "chambre",
    )
    return (
        bool(changes)
        or any(re.search(rf"\b{re.escape(word)}\b", lower) for word in verbs)
        or any(value is not None for value in state.to_raw_dict().values())
    )
