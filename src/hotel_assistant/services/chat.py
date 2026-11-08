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
from hotel_assistant.ml.replies import ModelRouter, ReplyRequest
from hotel_assistant.rag.policies import PolicyRetriever, Retrieval
from hotel_assistant.services.booking import BookingWorkflow
from hotel_assistant.services.interpret import (
    contextual_booking_changes,
    has_booking_terms,
    is_booking_request,
    language_of,
)

_ROOT = Path(__file__).resolve().parents[3]
_CONFIRM = {
    "confirm",
    "confirm booking",
    "yes, confirm",
    "buchung bestätigen",
    "ja, bestätigen",
    "confirmer",
    "confirmer la réservation",
}
_DECLINE = {"no", "don't book", "do not book", "cancel request", "nicht buchen", "ne pas réserver"}
_GENERATIVE_ROUTES = {"policy", "greeting"}
_UNSAFE_REPLY = re.compile(
    r"\b(?:booked|reserviert|réservée?|confirmed)\b|"
    r"\b(?:booking|reservation|buchung|réservation)\s+(?:is\s+)?"
    r"(?:created|complete|bestätigt|confirmée)\b",
    re.IGNORECASE,
)
_FACT_NUMBERS = re.compile(r"\d+(?::\d+)?(?:\.\d+)*")
_PROMPTS = {
    "en": {
        "guest_name": "What’s your full name for the booking?",
        "check_in": "What’s your check-in date? Please use DD.MM.YYYY, for example 12.06.2027.",
        "check_out": "What’s your check-out date? Please use DD.MM.YYYY, for example 14.06.2027.",
        "guests": "How many guests will be staying?",
    },
    "de": {
        "guest_name": "Wie lautet Ihr vollständiger Name für die Buchung?",
        "check_in": (
            "Wann möchten Sie anreisen? Bitte im Format DD.MM.YYYY, zum Beispiel 12.06.2027."
        ),
        "check_out": (
            "Wann möchten Sie abreisen? Bitte im Format DD.MM.YYYY, zum Beispiel 14.06.2027."
        ),
        "guests": "Wie viele Gäste werden übernachten?",
    },
    "fr": {
        "guest_name": "Quel est votre nom complet pour la réservation ?",
        "check_in": (
            "Quelle est votre date d’arrivée ? Utilisez le format DD.MM.YYYY, "
            "par exemple 12.06.2027."
        ),
        "check_out": (
            "Quelle est votre date de départ ? Utilisez le format DD.MM.YYYY, "
            "par exemple 14.06.2027."
        ),
        "guests": "Combien de personnes séjourneront ?",
    },
}
_CONFIRM_BY_LANGUAGE = {
    "en": "confirm booking",
    "de": "Buchung bestätigen",
    "fr": "confirmer la réservation",
}
_GUIDANCE_BY_LANGUAGE = {
    "en": (
        "I can help you find a room, answer a hotel question, or suggest places "
        "to explore in Beirut. "
        "If you’re planning a stay, tell me your dates (DD.MM.YYYY) and I’ll take it from there."
    ),
    "de": (
        "Ich helfe Ihnen gern bei der Zimmersuche oder einer Frage zum Hotel. "
        "Nennen Sie mir für einen Aufenthalt Ihre Reisedaten im Format DD.MM.YYYY."
    ),
    "fr": (
        "Je peux vous aider à trouver une chambre ou répondre à une question sur l’hôtel. "
        "Pour un séjour, donnez-moi vos dates au format DD.MM.YYYY et je vous guiderai."
    ),
}
_UNKNOWN_POLICY_BY_LANGUAGE = {
    "en": (
        "I don’t have reliable information on that yet. Please check with reception, "
        "or ask me about rooms, hotel policies, or Beirut sights."
    ),
    "de": (
        "Dazu habe ich keine verlässlichen Hotelinformationen. "
        "Fragen Sie bitte an der Rezeption nach oder stellen Sie mir eine Frage "
        "zu Zimmern, Hotelregeln oder Beirut."
    ),
    "fr": (
        "Je n’ai pas d’information fiable à ce sujet. Vérifiez auprès de la réception, "
        "ou demandez-moi des informations sur les chambres, l’hôtel ou Beyrouth."
    ),
}


@dataclass(frozen=True, slots=True)
class ChatResult:
    session_id: str
    message: str
    debug: dict[str, Any] = field(default_factory=dict)
    booking_state: BookingState | None = None
    card: dict[str, str] | None = None


@dataclass(slots=True)
class _Session:
    state: BookingState
    workflow: BookingWorkflow
    touched: float
    booked: bool = False
    booking_active: bool = False
    language: str = "en"
    quote_card: dict[str, str] | None = None


class ChatService:
    """Single-process conversation controller with bounded, expiring sessions."""

    def __init__(
        self,
        repository: HotelRepository | None = None,
        retriever: PolicyRetriever | None = None,
        qwen: QwenModel | None = None,
        answer_generator: ModelRouter | Callable[[str, str, str, str], str] | None = None,
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
            "llm_status": "not_used_for_route",
        }
        def answer(
            text: str, route: str, intent: str, card: dict[str, str] | None = None
        ) -> ChatResult:
            debug["selected_route"] = route
            debug["detected_intent"] = intent
            debug["llm_status"] = (
                "not_configured" if self.answer_generator is None else "fallback"
            ) if route in _GENERATIVE_ROUTES else "not_used_for_route"
            if self.answer_generator is not None and route in _GENERATIVE_ROUTES:
                # The source citation is application-owned and never entrusted to the LLM.
                body, separator, citation = text.partition("\n\nSource: ")
                try:
                    if isinstance(self.answer_generator, ModelRouter):
                        generated = self.answer_generator.generate(
                            ReplyRequest(message, body, language, route),
                            validator=lambda reply: self._valid_generated_reply(body, reply, route),
                        )
                    else:
                        generated = self.answer_generator(message, body, language, route).strip()
                except Exception as exc:
                    logging.exception("Grounded reply generation failed; using verified text")
                    debug["generation_error"] = type(exc).__name__
                    generated = ""
                if self._valid_generated_reply(body, generated, route):
                    text = generated + (separator + citation if separator else "")
                    debug["response_mode"] = "llm"
                    debug["llm_status"] = "used"
            if route == "quote":
                session.quote_card = card
            elif (
                route in {"policy", "local_guide", "unsupported_policy"}
                and session.quote_card is not None
            ):
                card = session.quote_card
            else:
                if session.quote_card is not None:
                    session.workflow.discard_quote()
                session.quote_card = None
            return ChatResult(identifier, text, debug, state, card)

        def knowledge_answer(item: Retrieval) -> ChatResult:
            debug["sources"] = [item.source]
            if item.kind == "local_guide":
                links = ", ".join(f"[{label}]({url})" for label, url in item.sources)
                citation = f"Local guide [{item.source}]"
                if "https://" not in item.answer:
                    citation += f" · {links}"
                route, intent = "local_guide", "SIGHTSEEING"
            else:
                citation = f"Hotel policy [{item.source}]"
                route, intent = "policy", "HOTEL_POLICY"
            return answer(f"{item.answer}\n\nSource: {citation}", route, intent)

        try:
            state.record_user_turn()
        except TurnLimitExceeded:
            return answer(
                f"This conversation has reached its {state.max_turns}-turn limit. "
                "Start a new conversation.",
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
            session.booking_active = False
            return answer(
                "I’ve discarded that request. Tell me new dates whenever you’re ready.",
                "booking_discarded", "BOOKING"
            )

        if message.casefold().rstrip(".!? ") in _CONFIRM:
            try:
                session.workflow.confirm(_CONFIRM_BY_LANGUAGE[language])
                reservation = session.workflow.create_booking()
            except ValueError as exc:
                return answer(str(exc), "clarification", "BOOKING")
            session.booked = True
            confirmation_text = {
                "en": "Booking confirmed! Your reference is",
                "de": "Buchung bestätigt! Ihre Referenz ist",
                "fr": "Réservation confirmée ! Votre référence est",
            }[language]
            return answer(
                f"{confirmation_text} {reservation.reference}.",
                "booking_created",
                "BOOKING",
                {"type": "success", "reference": reservation.reference, "language": language},
            )

        # Policy questions with no explicit booking details can be answered from the
        # verified knowledge base before a generative proposal can redirect them.
        question = "?" in message or any(
            word in message.casefold()
            for word in ("what", "when", "how", "wann", "wie", "quel", "quand", "comment")
        )
        try:
            initial_changes = contextual_booking_changes(
                message, state, active_booking=session.booking_active
            )
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
            and (
                session.quote_card is not None
                or not is_booking_request(message, state, initial_changes)
            )
        ):
            policy = self.retriever.retrieve(message, language)
            if policy:
                return knowledge_answer(policy)
            return answer(
                _UNKNOWN_POLICY_BY_LANGUAGE[language],
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
                    "I can’t change an existing reservation here. Please contact reception.",
                    "unsupported_reservation_change",
                    str(debug["detected_intent"]),
                )
            if debug["detected_intent"] == "GREETING":
                return answer(
                    "Hi! Looking for a room, or have a question about the hotel?",
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
        if policy and not changes and (question or not has_booking_terms(message)):
            return knowledge_answer(policy)

        if is_booking_request(message, state, changes) or (
            qwen_proposal is not None and qwen_proposal.intent == "BOOKING"
        ):
            session.booking_active = True
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
                if (
                    missing[0] == "guest_name"
                    and state.check_in is not None and state.check_out is not None
                ):
                    dates = f"{state.check_in:%d.%m.%Y}–{state.check_out:%d.%m.%Y}"
                    lead = {
                        "en": f"I have your dates, {dates}. ",
                        "de": f"Ich habe Ihre Reisedaten, {dates}. ",
                        "fr": f"J’ai vos dates de séjour, {dates}. ",
                    }[language]
                    prompt = lead + prompt
                if "guest_name" in changes and state.guest_name:
                    thanks = {
                        "en": f"Thanks, {state.guest_name}. ",
                        "de": f"Danke, {state.guest_name}. ",
                        "fr": f"Merci, {state.guest_name}. ",
                    }[language]
                    prompt = thanks + prompt
                return answer(prompt, "clarification", "BOOKING")
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
            dollars, cents = divmod(quote.total_cents, 100)
            total_usd = f"US${dollars:,}.{cents:02d}"
            if language == "de":
                assert state.check_in is not None and state.check_out is not None
                nights = f"{quote.nights} {'Nacht' if quote.nights == 1 else 'Nächte'}"
                quote_text = (
                    f"Zimmer {room.room_number} ({room.room_type}) ist vom "
                    f"{state.check_in:%d.%m.%Y} bis {state.check_out:%d.%m.%Y} "
                    f"für {nights} verfügbar. Gesamtpreis: {total_usd}. "
                    "Es wird keine Zahlung eingezogen. "
                    f"Wenn das für Sie passt, antworten Sie mit „{confirmation}“."
                )
            elif language == "fr":
                assert state.check_in is not None and state.check_out is not None
                nights = f"{quote.nights} {'nuit' if quote.nights == 1 else 'nuits'}"
                quote_text = (
                    f"La chambre {room.room_number} ({room.room_type}) est disponible du "
                    f"{state.check_in:%d.%m.%Y} au {state.check_out:%d.%m.%Y} pour {nights}. "
                    f"Prix total : {total_usd}. Aucun paiement n'est prélevé. "
                    f"Si cela vous convient, répondez « {confirmation} »."
                )
            else:
                assert state.check_in is not None and state.check_out is not None
                nights = f"{quote.nights} {'night' if quote.nights == 1 else 'nights'}"
                quote_text = (
                    f"Room {room.room_number} ({room.room_type}) is available from "
                    f"{state.check_in:%d.%m.%Y} to {state.check_out:%d.%m.%Y} for {nights}. "
                    f"Total: {total_usd}. No payment is taken. "
                    f"If that works for you, reply “{confirmation}”."
                )
            return answer(
                quote_text,
                "quote",
                "BOOKING",
                {
                    "type": "quote",
                    "language": language,
                    "room": f"{room.room_number} · {room.room_type}",
                    "dates": f"{state.check_in:%d.%m.%Y} – {state.check_out:%d.%m.%Y}",
                    "nights": str(quote.nights),
                    "guests": str(state.guests),
                    "guest_name": str(state.guest_name),
                    "total": total_usd,
                    "check_in": state.check_in.isoformat(),
                    "check_out": state.check_out.isoformat(),
                    "room_type": room.room_type,
                },
            )

        if policy:
            return knowledge_answer(policy)
        if question:
            return answer(
                _UNKNOWN_POLICY_BY_LANGUAGE[language],
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
        return "DD.MM.YYYY" not in fact or "DD.MM.YYYY" in generated
