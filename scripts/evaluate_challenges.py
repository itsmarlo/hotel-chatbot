"""Exercise additional booking and sightseeing edge cases outside the original 30 dialogues."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from hotel_assistant.database.repository import HotelRepository
from hotel_assistant.services.chat import ChatService

CASES = [
    ("Beirut sight seeing tips, please?", "local_guide"),
    ("Any ideas for visiting Sursock Museum?", "local_guide"),
    ("Que voir à Beyrouth ?", "local_guide"),
    ("Is the Corniche worth a walk?", "local_guide"),
    ("Book a double room 2030-08-03 to 2030-08-05 for 2 guests. My name is Sam Lee.", "quote"),
    ("Book a room 2030-02-30 to 2030-03-02 for 2 guests. My name is Sam Lee.", "clarification"),
    ("Can you book me for tomorrow?", "clarification"),
    (
        "Book a double room 2030-08-11 to 2030-08-13 for 2 Gäste. My name is Sam Lee.",
        "quote",
    ),
    ("Was kann ich in Beirut besichtigen?", "local_guide"),
]


def evaluate() -> dict[str, object]:
    with TemporaryDirectory() as directory:
        repository = HotelRepository(Path(directory) / "challenges.db")
        service = ChatService(repository=repository)
        outcomes = []
        for prompt, expected in CASES:
            before = repository.count_rows("reservations")
            result = service.respond(prompt)
            outcomes.append({
                "prompt": prompt,
                "expected": expected,
                "actual": result.debug["selected_route"],
                "passed": result.debug["selected_route"] == expected
                and repository.count_rows("reservations") == before,
            })
        first = service.respond(
            "Book a double room 2030-09-02 to 2030-09-04 for 2 guests. My name is Sam Lee."
        )
        before = repository.count_rows("reservations")
        ambiguous = service.respond("yes, that looks good", first.session_id)
        outcomes.append({
            "prompt": "ambiguous confirmation after quote",
            "expected": "no booking write",
            "actual": ambiguous.debug["selected_route"],
            "passed": repository.count_rows("reservations") == before,
        })
        first = service.respond(
            "Book a double room 2030-09-06 to 2030-09-08 for 2 guests. My name is Sam Lee."
        )
        before = repository.count_rows("reservations")
        changed = service.respond("Actually, make checkout 09.09.2030", first.session_id)
        outcomes.append({
            "prompt": "changed checkout after quote",
            "expected": "no booking write",
            "actual": changed.debug["selected_route"],
            "passed": repository.count_rows("reservations") == before,
        })
    return {
        "passed": sum(row["passed"] for row in outcomes),
        "total": len(outcomes),
        "failures": [row for row in outcomes if not row["passed"]],
        "limitation": "Additional authored challenge prompts, not independent guest traffic.",
    }


if __name__ == "__main__":
    report = evaluate()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["passed"] == report["total"] else 1)
