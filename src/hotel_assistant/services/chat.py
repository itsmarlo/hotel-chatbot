from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from hotel_assistant.config import Settings
from hotel_assistant.database.repository import HotelRepository
from hotel_assistant.dialogue.state import BookingState, TurnLimitExceeded
from hotel_assistant.rag.policies import PolicyRetriever
from hotel_assistant.services.booking import BookingWorkflow
from hotel_assistant.services.interpret import booking_changes, is_booking_request, language_of

_ROOT = Path(__file__).resolve().parents[3]
_CONFIRM = {"confirm booking", "buchung bestätigen", "confirmer la réservation"}
_PROMPTS = {
    "en": {
        "guest_name": "your name (for example, 'my name is Alex Kim')",
        "check_in": "a check-in date in YYYY-MM-DD format",
        "check_out": "a check-out date in YYYY-MM-DD format",
        "guests": "the number of guests (for example, '2 guests')",
    },
    "de": {
        "guest_name": "Ihren Namen (zum Beispiel 'Ich heiße Alex Kim')",
        "check_in": "ein Anreisedatum im Format YYYY-MM-DD",
        "check_out": "ein Abreisedatum im Format YYYY-MM-DD",
        "guests": "die Gästezahl (zum Beispiel '2 Gäste')",
    },
    "fr": {
        "guest_name": "votre nom (par exemple « je m'appelle Alex Kim »)",
        "check_in": "une date d'arrivée au format YYYY-MM-DD",
        "check_out": "une date de départ au format YYYY-MM-DD",
        "guests": "le nombre de personnes (par exemple « 2 personnes »)",
    },
}
_CONFIRM_BY_LANGUAGE = {
    "en": "confirm booking",
    "de": "Buchung bestätigen",
    "fr": "confirmer la réservation",
}


@dataclass(frozen=True, slots=True)
class ChatResult:
    session_id: str
    message: str
    debug: dict[str, Any] = field(default_factory=dict)
    booking_state: BookingState | None = None


@dataclass(slots=True)
class _Session:
    state: BookingState
    workflow: BookingWorkflow
    touched: float
    booked: bool = False
    language: str = "en"


class ChatService:
    """Single-process conversation controller with bounded, expiring sessions."""

    def __init__(
        self,
        repository: HotelRepository | None = None,
        retriever: PolicyRetriever | None = None,
        *,
        max_sessions: int = 1000,
        session_ttl_seconds: int = 1800,
    ) -> None:
        self.repository = repository or HotelRepository(Settings.from_env().database_path)
        self.repository.initialize()
        self.repository.seed()
        self.retrieval_mode = os.getenv("HOTEL_RAG_MODE", "lexical").casefold()
        if self.retrieval_mode not in {"lexical", "semantic"}:
            raise ValueError("HOTEL_RAG_MODE must be lexical or semantic")
        self.retriever = retriever or PolicyRetriever(
            _ROOT / "data" / "knowledge_base" / "policies.json",
            semantic=self.retrieval_mode == "semantic",
        )
        self.max_sessions = max_sessions
        self.session_ttl_seconds = session_ttl_seconds
        self._sessions: dict[str, _Session] = {}
        self._lock = RLock()

    def reset(self, session_id: str | None) -> None:
        if session_id:
            with self._lock:
                self._sessions.pop(session_id, None)

    def _session(self, session_id: str | None) -> tuple[str, _Session]:
        now = time.monotonic()
        self._sessions = {
            key: value
            for key, value in self._sessions.items()
            if now - value.touched < self.session_ttl_seconds
        }
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            session.touched = now
            return session_id, session
        if len(self._sessions) >= self.max_sessions:
            oldest = min(self._sessions, key=lambda key: self._sessions[key].touched)
            self._sessions.pop(oldest)
        state = BookingState.from_config(_ROOT / "configs" / "booking.json")
        identifier = str(uuid4())
        session = _Session(state, BookingWorkflow(state, self.repository), now)
        self._sessions[identifier] = session
        return identifier, session

    def respond(self, message: str, session_id: str | None = None) -> ChatResult:
        if not message.strip():
            raise ValueError("message must not be empty")
        if len(message) > 4_000:
            raise ValueError("message must be at most 4000 characters")
        with self._lock:
            identifier, session = self._session(session_id)
            return self._respond(message.strip(), identifier, session)

    def _respond(self, message: str, identifier: str, session: _Session) -> ChatResult:
        state = session.state
        detected_language = language_of(message)
        if detected_language != "en" or any(
            word in message.casefold()
            for word in ("book", "room", "my name", "guests", "hello", "when", "what")
        ):
            session.language = detected_language
        language = session.language
        debug: dict[str, Any] = {
            "detected_intent": None,
            "confidence": None,
            "extracted_slots": {},
            "selected_route": "clarification",
            "sources": [],
        }

        def answer(text: str, route: str, intent: str) -> ChatResult:
            debug["selected_route"] = route
            debug["detected_intent"] = intent
            return ChatResult(identifier, text, debug, state)

        try:
            state.record_user_turn()
        except TurnLimitExceeded:
            return answer(
                "This conversation has reached its 10-turn limit. Start a new conversation.",
                "turn_limit",
                "OTHER",
            )

        if session.booked:
            return answer(
                "This booking is complete. Start a new conversation for another request.",
                "complete",
                "OTHER",
            )

        if message.casefold() in _CONFIRM:
            try:
                session.workflow.confirm(message)
                reservation = session.workflow.create_booking()
            except ValueError as exc:
                return answer(str(exc), "clarification", "BOOKING")
            session.booked = True
            confirmation_text = {
                "en": "Booking confirmed. Reference",
                "de": "Buchung bestätigt. Referenz",
                "fr": "Réservation confirmée. Référence",
            }[language]
            return answer(f"{confirmation_text}: {reservation.reference}.",
                          "booking_created", "BOOKING")

        try:
            changes = booking_changes(message)
        except ValueError as exc:
            return answer(str(exc), "clarification", "BOOKING")
        debug["extracted_slots"] = {
            key: value.isoformat() if hasattr(value, "isoformat") else value
            for key, value in changes.items()
        }

        policy = self.retriever.retrieve(message, language)
        question = "?" in message or any(
            word in message.casefold()
            for word in ("what", "when", "how", "wann", "wie", "quel", "quand", "comment")
        )
        if policy and question and not changes:
            debug["sources"] = [policy.source]
            return answer(
                f"{policy.answer}\n\nSource: Hotel policy [{policy.source}]",
                "policy",
                "HOTEL_POLICY",
            )

        if is_booking_request(message, state, changes):
            if changes:
                try:
                    state.update(**changes)
                except ValueError as exc:
                    return answer(
                        f"Please correct the booking details: {exc}", "clarification", "BOOKING"
                    )
            missing = state.missing_required_fields()
            if missing:
                prompt = _PROMPTS[language][missing[0]]
                lead = {"en": "To continue, please provide", "de": "Bitte geben Sie an:",
                        "fr": "Pour continuer, indiquez"}[language]
                return answer(f"{lead} {prompt}.", "clarification", "BOOKING")
            try:
                rooms = session.workflow.search_rooms()
                if not rooms:
                    return answer(
                        "No room is available for those dates and guests. "
                        "Try different dates or a room type.",
                        "availability",
                        "BOOKING",
                    )
                room = rooms[0]
                quote = session.workflow.calculate_price(room.room_id)
            except ValueError as exc:
                return answer(
                    f"Please correct the booking details: {exc}", "clarification", "BOOKING"
                )
            confirmation = _CONFIRM_BY_LANGUAGE[language]
            if language == "de":
                quote_text = (
                    f"Zimmer {room.room_number} ({room.room_type}) ist für {quote.nights} "
                    f"Nacht/Nächte verfügbar. Demo-Gesamtpreis: {quote.total_cents} Cent. "
                    f"Es wird keine Zahlung eingezogen. Antworten Sie zur Bestätigung genau: "
                    f"{confirmation}"
                )
            elif language == "fr":
                quote_text = (
                    f"La chambre {room.room_number} ({room.room_type}) est disponible pour "
                    f"{quote.nights} nuit(s). "
                    f"Total de démonstration : {quote.total_cents} centimes. "
                    f"Aucun paiement n'est prélevé. Pour confirmer, répondez exactement : "
                    f"{confirmation}"
                )
            else:
                quote_text = (
                    f"Room {room.room_number} ({room.room_type}) is available for "
                    f"{quote.nights} night(s). Demo total: {quote.total_cents} cents. "
                    f"No payment is taken. To confirm these details, reply exactly: {confirmation}"
                )
            return answer(
                quote_text,
                "quote",
                "BOOKING",
            )

        if policy:
            debug["sources"] = [policy.source]
            return answer(
                f"{policy.answer}\n\nSource: Hotel policy [{policy.source}]",
                "policy",
                "HOTEL_POLICY",
            )
        if question:
            return answer(
                "I don't have a verified hotel policy for that question.",
                "unsupported_policy",
                "HOTEL_POLICY",
            )
        return answer(
            "I can help with a room request or hotel policies. For a booking, give dates as "
            "YYYY-MM-DD, a guest count, and your name. I will ask before creating it.",
            "guidance",
            "OTHER",
        )
