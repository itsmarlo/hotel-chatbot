"""Compare a single-prompt Qwen baseline on the guarded demo's 30 dialogues.

The baseline can only claim a booking in JSON; it has no access to SQLite or tools.
"""

from __future__ import annotations

import json
import sys

from evaluate_dialogues import scenarios

from hotel_assistant.ml.qwen import QwenModel


def evaluate() -> dict[str, object]:
    model = QwenModel()
    results = []
    confirmations = {
        "en": "confirm booking",
        "de": "Buchung bestätigen",
        "fr": "confirmer la réservation",
    }
    for case in scenarios():
        history = [f"User: {case['message']}"]
        first = model.baseline(history)
        outcome: dict[str, object] = {
            "id": case["id"],
            "kind": case["kind"],
            "claimed_before_confirmation": first.booking_created,
            "first_reply": first.reply,
        }
        if case["kind"] == "complete":
            history.extend(
                (f"Assistant: {first.reply}", f"User: {confirmations[case['language']]}")
            )
            final = model.baseline(history)
            outcome["claimed_after_confirmation"] = final.booking_created
            outcome["final_reply"] = final.reply
        results.append(outcome)
        print(f"{len(results)}/30 {case['id']}", file=sys.stderr, flush=True)
    return {
        "model": "Qwen2.5-1.5B-Instruct single-prompt baseline",
        "dialogues": len(results),
        "unsafe_preconfirmation_claims": sum(
            bool(row["claimed_before_confirmation"]) for row in results
        ),
        "invalid_date_booking_claims": sum(
            row["kind"] == "invalid" and bool(row["claimed_before_confirmation"]) for row in results
        ),
        "completed_valid_bookings": sum(
            row["kind"] == "complete" and bool(row.get("claimed_after_confirmation"))
            for row in results
        ),
        "results": results,
        "limitation": (
            "Booking creation is a model claim, not a database write. "
            "This synthetic comparison does not measure real guests."
        ),
    }


if __name__ == "__main__":
    report = evaluate()
    print(json.dumps(report, indent=2, ensure_ascii=False))
