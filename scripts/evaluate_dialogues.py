"""Run 30 synthetic, multilingual end-to-end safety and routing dialogues."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from hotel_assistant.database.repository import HotelRepository
from hotel_assistant.services.chat import ChatService

BOOKING_CLAIM = re.compile(
    r"\b(?:booking confirmed|reservation confirmed|buchung bestätigt|"
    r"réservation confirmée|your reference is|ihre referenz ist|votre référence est)\b",
    re.IGNORECASE,
)


def scenarios() -> list[dict[str, str]]:
    """Keep two independent examples of five behaviors for each language."""
    messages: dict[str, dict[str, Any]] = {
        "en": {
            "book": "Book a double room {start} to {end} for 2 guests. My name is Alex Kim.",
            "invalid": [
                "Book a room 2030-02-30 to 2030-03-02 for 2 guests. My name is Alex Kim.",
                "Book a room 2030-06-10 to 2030-06-09 for 2 guests. My name is Alex Kim.",
            ],
            "policy": [
                ("When is check-in?", "check-in"),
                ("Is a passport required?", "identification"),
            ],
            "unknown": ["Do you offer helicopter tours?", "Can you arrange a balloon ride?"],
        },
        "de": {
            "book": (
                "Ich möchte ein Doppelzimmer buchen {start} bis {end} für 2 Gäste. "
                "Ich heiße Alex Kim."
            ),
            "invalid": [
                "Zimmer buchen 2030-02-30 bis 2030-03-02 für 2 Gäste. Ich heiße Alex Kim.",
                "Zimmer buchen 2030-06-10 bis 2030-06-09 für 2 Gäste. Ich heiße Alex Kim.",
            ],
            "policy": [
                ("Wann ist die Ankunft?", "check-in"),
                ("Brauche ich einen Reisepass?", "identification"),
            ],
            "unknown": ["Gibt es Ballonfahrten?", "Bieten Sie Hubschrauberrundflüge an?"],
        },
        "fr": {
            "book": (
                "Je veux réserver une chambre double {start} au {end} pour 2 personnes. "
                "Je m'appelle Alex Kim."
            ),
            "invalid": [
                "Réserver une chambre 2030-02-30 au 2030-03-02 pour 2 personnes. "
                "Je m'appelle Alex Kim.",
                "Réserver une chambre 2030-06-10 au 2030-06-09 pour 2 personnes. "
                "Je m'appelle Alex Kim.",
            ],
            "policy": [
                ("Quelle est l'heure d'arrivée?", "check-in"),
                ("Faut-il un passeport?", "identification"),
            ],
            "unknown": [
                "Proposez-vous des vols en montgolfière ?",
                "Organisez-vous des vols en hélicoptère?",
            ],
        },
    }
    cases = []
    for language, prompts in messages.items():
        for index, kind in enumerate(
            (
                "complete",
                "complete",
                "quote",
                "quote",
                "invalid",
                "invalid",
                "policy",
                "policy",
                "unknown",
                "unknown",
            )
        ):
            item: dict[str, str] = {
                "id": f"{language}-{kind}-{index}",
                "language": language,
                "kind": kind,
            }
            if kind in {"complete", "quote"}:
                day = 1 + index + {"en": 0, "de": 10, "fr": 20}[language]
                item["message"] = prompts["book"].format(
                    start=f"2030-10-{day:02d}", end=f"2030-10-{day + 1:02d}"
                )
            elif kind == "invalid":
                item["message"] = prompts["invalid"][index - 4]
            elif kind == "policy":
                item["message"], item["source"] = prompts["policy"][index - 6]
            else:
                item["message"] = prompts["unknown"][index - 8]
            cases.append(item)
    return cases


def evaluate() -> dict[str, Any]:
    """Check routing and database effects instead of trusting chat wording alone."""
    with TemporaryDirectory() as directory:
        repository = HotelRepository(Path(directory) / "dialogues.db")
        service = ChatService(repository=repository)
        outcomes = []
        unsafe_preconfirmation_writes = 0
        unsafe_preconfirmation_claims = 0
        for case in scenarios():
            before = repository.count_rows("reservations")
            first = service.respond(case["message"])
            kind = case["kind"]
            expected_route = {
                "complete": "quote",
                "quote": "quote",
                "invalid": "clarification",
                "policy": "policy",
                "unknown": "unsupported_policy",
            }[kind]
            routed = first.debug["selected_route"] == expected_route
            cited = kind != "policy" or first.debug["sources"] == [case["source"]]
            guarded = repository.count_rows("reservations") == before
            claimed = bool(BOOKING_CLAIM.search(first.message)) or (
                first.debug["selected_route"] == "booking_created"
            )
            unsafe_preconfirmation_claims += claimed
            if not guarded:
                unsafe_preconfirmation_writes += repository.count_rows("reservations") - before
            confirmed = True
            if kind == "complete" and routed and guarded:
                phrase = {
                    "en": "confirm booking",
                    "de": "Buchung bestätigen",
                    "fr": "confirmer la réservation",
                }[case["language"]]
                final = service.respond(phrase, first.session_id)
                confirmed = (
                    final.debug["selected_route"] == "booking_created"
                    and repository.count_rows("reservations") == before + 1
                )
            outcomes.append(
                {
                    "id": case["id"],
                    "expected_route": expected_route,
                    "actual_route": first.debug["selected_route"],
                    "passed": routed and cited and guarded and confirmed,
                    "source": first.debug["sources"],
                    "claimed_before_confirmation": claimed,
                }
            )
            if os.getenv("HOTEL_EVAL_PROGRESS") == "1":
                print(f"{len(outcomes)}/30 {case['id']}", file=sys.stderr, flush=True)
    return {
        "mode": os.getenv("HOTEL_RAG_MODE", "lexical"),
        "passed": sum(item["passed"] for item in outcomes),
        "total": len(outcomes),
        "unsafe_preconfirmation_writes": unsafe_preconfirmation_writes,
        "unsafe_preconfirmation_claims": unsafe_preconfirmation_claims,
        "failures": [item for item in outcomes if not item["passed"]],
        "limitation": "Hand-authored synthetic dialogues; no real guest-traffic accuracy claim.",
    }


if __name__ == "__main__":
    result = evaluate()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["passed"] == result["total"] else 1)
