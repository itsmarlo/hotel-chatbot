"""Summarize the two raw, reproducible Qwen evaluation reports for documentation."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build() -> Path:
    guarded = json.loads((ROOT / "reports" / "qwen_guarded.json").read_text())
    baseline = json.loads((ROOT / "reports" / "qwen_baseline.json").read_text())
    if guarded["total"] != 30 or baseline["dialogues"] != 30:
        raise ValueError("Both evaluations must use all 30 scripted dialogues")
    if "unsafe_preconfirmation_writes" in guarded:
        writes = guarded["unsafe_preconfirmation_writes"]
    elif guarded["passed"] == guarded["total"]:
        # Older reports record the database guard as part of each passed case.
        writes = 0
    else:
        raise ValueError("Cannot infer database writes from a partially passing report")
    summary = {
        "dataset": "30 hand-authored synthetic dialogues, 10 each in English, German, French",
        "guarded": {
            "passed": guarded["passed"],
            "total": guarded["total"],
            "unsafe_preconfirmation_writes": writes,
        },
        "baseline": {
            "unsafe_preconfirmation_claims": baseline["unsafe_preconfirmation_claims"],
            "invalid_date_booking_claims": baseline["invalid_date_booking_claims"],
            "completed_valid_bookings": baseline["completed_valid_bookings"],
        },
        "interpretation": (
            "The baseline only generates booking claims; it cannot write to the database. "
            "These authored examples are not an estimate of real guest traffic."
        ),
    }
    output = ROOT / "docs" / "qwen_comparison.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return output


if __name__ == "__main__":
    print(build())
