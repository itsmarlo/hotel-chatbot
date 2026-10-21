from __future__ import annotations

import logging
import os
import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from hotel_assistant.config import Settings
from hotel_assistant.database.repository import HotelRepository
from hotel_assistant.dialogue.state import BookingState, TurnLimitExceeded
from hotel_assistant.ml.inference import TrainedNLP
from hotel_assistant.ml.qwen import BookingProposal, QwenModel
from hotel_assistant.rag.policies import PolicyRetriever
from hotel_assistant.services.booking import BookingWorkflow
from hotel_assistant.services.interpret import booking_changes, is_booking_request, language_of

_ROOT = Path(__file__).resolve().parents[3]
_CONFIRM = {"confirm booking", "buchung bestätigen", "confirmer la réservation"}
_DECLINE = {"no", "don't book", "do not book", "cancel request", "nicht buchen", "ne pas réserver"}
_GENERATIVE_ROUTES = {"policy", "greeting", "guidance", "clarification"}
_UNSAFE_REPLY = re.compile(
    r"\b(?:booked|reserviert|réservée?|confirmed)\b|"
    r"\b(?:booking|reservation|buchung|réservation)\s+(?:is\s+)?"
    r"(?:created|complete|bestätigt|confirmée)\b",
    re.IGNORECASE,
)
_FACT_NUMBERS = re.compile(r"\d+(?::\d+)?(?:\.\d+)*")
_BOOKING_SLOT_TERMS = {
    "guest_name": re.compile(r"\b(?:name[n]?|nom)\b", re.IGNORECASE),
    "check_in": re.compile(r"check[- ]?in|anreise|arriv", re.IGNORECASE),
    "check_out": re.compile(r"check[- ]?out|abreise|départ", re.IGNORECASE),
    "date": re.compile(r"DD\.MM\.YYYY|datum|date", re.IGNORECASE),
    "guests": re.compile(r"guest|gäste|people|personne", re.IGNORECASE),
}
_PROMPTS = {
    "en": {
        "guest_name": "your name (for example, 'my name is Alex Kim')",
        "check_in": "a check-in date in DD.MM.YYYY format (for example, 12.06.2027)",
        "check_out": "a check-out date in DD.MM.YYYY format (for example, 14.06.2027)",
        "guests": "the number of guests (for example, '2 guests')",
    },
    "de": {
        "guest_name": "Ihren Namen (zum Beispiel 'Ich heiße Alex Kim')",
        "check_in": "ein Anreisedatum im Format DD.MM.YYYY (z. B. 12.06.2027)",
        "check_out": "ein Abreisedatum im Format DD.MM.YYYY (z. B. 14.06.2027)",
        "guests": "die Gästezahl (zum Beispiel '2 Gäste')",
    },
    "fr": {
        "guest_name": "votre nom (par exemple « je m'appelle Alex Kim »)",
        "check_in": "une date d'arrivée au format DD.MM.YYYY (par exemple 12.06.2027)",
        "check_out": "une date de départ au format DD.MM.YYYY (par exemple 14.06.2027)",
        "guests": "le nombre de personnes (par exemple « 2 personnes »)",
    },
}
_CONFIRM_BY_LANGUAGE = {
    "en": "confirm booking",
    "de": "Buchung bestätigen",
    "fr": "confirmer la réservation",
}
_GUIDANCE_BY_LANGUAGE = {
    "en": (
        "I can help with a room request or hotel policies. For a booking, give dates as "
        "DD.MM.YYYY, a guest count, and your name. I will ask before creating it."
    ),
    "de": (
        "Ich helfe Ihnen bei Zimmeranfragen und Fragen zum Hotel. Für eine Buchung "
        "nennen Sie bitte An- und Abreisedatum im Format DD.MM.YYYY, die Anzahl der Gäste "
        "und Ihren Namen. Vor der Buchung bitte ich um Ihre Bestätigung."
    ),
    "fr": (
        "Je peux vous aider à réserver une chambre ou répondre aux questions sur l’hôtel. "
        "Pour réserver, indiquez les dates au format DD.MM.YYYY, le nombre de personnes et "
        "votre nom. Je demanderai votre confirmation avant de créer la réservation."
    ),
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
        qwen: QwenModel | None = None,
        answer_generator: Callable[[str, str, str, str], str] | None = None,
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
        artifact_dir = _ROOT / "models" / "trained"
        encoder = getattr(self.retriever, "_model", None)
        self.nlp_mode = os.getenv("HOTEL_NLP_MODE", "trained")
        if self.nlp_mode not in {"trained", "qwen", "off"}:
            raise ValueError("HOTEL_NLP_MODE must be trained, qwen, or off")
        self.nlp = (
            TrainedNLP(artifact_dir, encoder)
            if encoder is not None
            and (artifact_dir / "intent.joblib").is_file()
            and self.nlp_mode == "trained"
            else None
        )
        self.qwen = (qwen or QwenModel()) if self.nlp_mode == "qwen" else None
        self.answer_generator = answer_generator
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
            "model_entities": {},
            "proposed_tool": None,
            "response_mode": "template",
        }
        def answer(text: str, route: str, intent: str) -> ChatResult:
            debug["selected_route"] = route
            debug["detected_intent"] = intent
            friendly_clarification = route != "clarification" or text.startswith(
                ("To continue", "Bitte geben Sie an:", "Pour continuer")
            )
            if (
                self.answer_generator is not None
                and route in _GENERATIVE_ROUTES
                and friendly_clarification
            ):
                # The source citation is application-owned and never entrusted to the LLM.
                body, separator, citation = text.partition("\n\nSource: Hotel policy ")
                try:
                    generated = self.answer_generator(message, body, language, route).strip()
                except Exception as exc:
                    logging.exception("Grounded reply generation failed; using verified text")
                    debug["generation_error"] = type(exc).__name__
                    generated = ""
                if self._valid_generated_reply(body, generated, route):
                    text = generated + (separator + citation if separator else "")
                    debug["response_mode"] = "llm"
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

        if message.casefold() in _DECLINE:
            state.reset()
            session.workflow = BookingWorkflow(state, self.repository)
            return answer(
                "The unconfirmed booking request was discarded.", "booking_discarded", "BOOKING"
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
            return answer(
                f"{confirmation_text}: {reservation.reference}.", "booking_created", "BOOKING"
            )

        # Policy questions with no explicit booking details can be answered from the
        # verified knowledge base before a generative proposal can redirect them.
        question = "?" in message or any(
            word in message.casefold()
            for word in ("what", "when", "how", "wann", "wie", "quel", "quand", "comment")
        )
        try:
            initial_changes = booking_changes(message)
        except ValueError as exc:
            if language == "de":
                return answer(
                    "Bitte verwenden Sie gültige Daten im Format DD.MM.YYYY "
                    "(z. B. 12.06.2027) und geben Sie höchstens zwei Daten an.",
                    "clarification",
                    "BOOKING",
                )
            return answer(str(exc), "clarification", "BOOKING")
        if (
            question
            and not initial_changes
            and not is_booking_request(message, state, initial_changes)
        ):
            policy = self.retriever.retrieve(message, language)
            if policy:
                debug["sources"] = [policy.source]
                return answer(
                    f"{policy.answer}\n\nSource: Hotel policy [{policy.source}]",
                    "policy",
                    "HOTEL_POLICY",
                )
            return answer(
                "I don't have a verified hotel policy for that question.",
                "unsupported_policy",
                "HOTEL_POLICY",
            )

        if self.nlp is not None:
            proposal = self.nlp.predict(message)
            debug["detected_intent"] = proposal.intent
            debug["confidence"] = round(proposal.confidence, 3)
            debug["model_entities"] = proposal.entities
        qwen_proposal: BookingProposal | None = None
        if self.qwen is not None:
            try:
                qwen_proposal = self.qwen.interpret(message, state.to_dict())
            except (ValueError, RuntimeError) as exc:
                debug["model_error"] = type(exc).__name__
            else:
                debug["detected_intent"] = qwen_proposal.intent
                debug["model_entities"] = {
                    key: value
                    for key, value in qwen_proposal.model_dump().items()
                    if key not in {"intent", "proposed_tool"} and value
                }
                debug["proposed_tool"] = qwen_proposal.proposed_tool

        # Only clear intent predictions influence routing. Slot tags stay advisory because
        # the tiny synthetic token dataset cannot safely authorize booking state updates.
        if (
            (self.nlp is not None and debug["confidence"] >= 0.7) or qwen_proposal is not None
        ) and not any(value is not None for value in state.to_raw_dict().values()):
            action_words = (
                "cancel",
                "call off",
                "delete my booking",
                "stornier",
                "absagen",
                "annul",
                "modify",
                "change my",
                "update my",
                "ändern",
                "verschieben",
                "modifier",
                "changer",
            )
            if debug["detected_intent"] in {"CANCEL_BOOKING", "MODIFY_BOOKING"} and any(
                word in message.casefold() for word in action_words
            ):
                return answer(
                    "Existing reservations cannot be changed in this demo. "
                    "Please contact the hotel.",
                    "unsupported_reservation_change",
                    str(debug["detected_intent"]),
                )
            if debug["detected_intent"] == "GREETING":
                return answer(
                    "Hello! I can help with room requests and hotel policies.",
                    "greeting",
                    "GREETING",
                )

        changes = initial_changes
        if qwen_proposal is not None:
            # Proposed ISO dates must agree with dates parsed from the user's
            # explicit text, including dates written in German notation.
            for field_name in ("check_in", "check_out"):
                suggested = getattr(qwen_proposal, field_name)
                parsed = changes.get(field_name)
                if suggested and (
                    not isinstance(parsed, date)
                    or suggested != parsed.isoformat()
                ):
                    return answer(
                        (
                            "Bitte klären Sie die Buchungsdaten im Format DD.MM.YYYY."
                            if language == "de"
                            else "Please clarify the booking dates in DD.MM.YYYY format."
                        ),
                        "clarification",
                        "BOOKING",
                    )
            parsed_guests = changes.get("guests")
            if (
                qwen_proposal.guests
                and isinstance(parsed_guests, int)
                and qwen_proposal.guests != parsed_guests
            ):
                return answer(
                    "Please clarify the number of guests.", "clarification", "BOOKING"
                )
            if (
                qwen_proposal.guest_name
                and "guest_name" not in changes
                and qwen_proposal.guest_name.casefold() in message.casefold()
            ):
                changes["guest_name"] = qwen_proposal.guest_name
        debug["extracted_slots"] = {
            key: value.isoformat() if hasattr(value, "isoformat") else value
            for key, value in changes.items()
        }

        policy = self.retriever.retrieve(message, language)
        if policy and question and not changes:
            debug["sources"] = [policy.source]
            return answer(
                f"{policy.answer}\n\nSource: Hotel policy [{policy.source}]",
                "policy",
                "HOTEL_POLICY",
            )

        if is_booking_request(message, state, changes) or (
            qwen_proposal is not None and qwen_proposal.intent == "BOOKING"
        ):
            if changes:
                try:
                    state.update(**changes)
                except ValueError as exc:
                    if language == "de" and str(exc) == "check-out must be after check-in":
                        return answer(
                            "Das Abreisedatum muss nach dem Anreisedatum liegen.",
                            "clarification",
                            "BOOKING",
                        )
                    return answer(
                        f"Please correct the booking details: {exc}", "clarification", "BOOKING"
                    )
            missing = state.missing_required_fields()
            if missing:
                prompt = _PROMPTS[language][missing[0]]
                lead = {
                    "en": "To continue, please provide",
                    "de": "Bitte geben Sie an:",
                    "fr": "Pour continuer, indiquez",
                }[language]
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
                assert state.check_in is not None and state.check_out is not None
                quote_text = (
                    f"Zimmer {room.room_number} ({room.room_type}) ist vom "
                    f"{state.check_in:%d.%m.%Y} bis {state.check_out:%d.%m.%Y} "
                    f"für {quote.nights} "
                    f"Nacht/Nächte verfügbar. Demo-Gesamtpreis: {quote.total_cents} Cent. "
                    f"Es wird keine Zahlung eingezogen. Antworten Sie zur Bestätigung genau: "
                    f"{confirmation}"
                )
            elif language == "fr":
                assert state.check_in is not None and state.check_out is not None
                quote_text = (
                    f"La chambre {room.room_number} ({room.room_type}) est disponible du "
                    f"{state.check_in:%d.%m.%Y} au {state.check_out:%d.%m.%Y} pour "
                    f"{quote.nights} nuit(s). "
                    f"Total de démonstration : {quote.total_cents} centimes. "
                    f"Aucun paiement n'est prélevé. Pour confirmer, répondez exactement : "
                    f"{confirmation}"
                )
            else:
                assert state.check_in is not None and state.check_out is not None
                quote_text = (
                    f"Room {room.room_number} ({room.room_type}) is available from "
                    f"{state.check_in:%d.%m.%Y} to {state.check_out:%d.%m.%Y} for "
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
            _GUIDANCE_BY_LANGUAGE[language],
            "guidance",
            "OTHER",
        )

    @staticmethod
    def _valid_generated_reply(fact: str, generated: str, route: str) -> bool:
        """Reject empty, unrelated, or booking-claiming prose before displaying it."""
        if not generated or len(generated) > 600 or _UNSAFE_REPLY.search(generated):
            return False
        if route == "policy" and Counter(_FACT_NUMBERS.findall(fact)) != Counter(
            _FACT_NUMBERS.findall(generated)
        ):
            return False
        # Small models sometimes merge two separate timings into one event.
        if route == "policy":
            for anchor in (r"check[- ]in", r"check[- ]out", r"départ"):
                if re.search(anchor, fact, re.IGNORECASE) and not re.search(
                    anchor, generated, re.IGNORECASE
                ):
                    return False
        if route == "clarification":
            requested = {
                name for name, pattern in _BOOKING_SLOT_TERMS.items() if pattern.search(fact)
            }
            mentioned = {
                name for name, pattern in _BOOKING_SLOT_TERMS.items() if pattern.search(generated)
            }
            if not requested.issubset(mentioned) or not mentioned.issubset(requested):
                return False
        return "DD.MM.YYYY" not in fact or "DD.MM.YYYY" in generated
