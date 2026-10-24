"""Summarize the two raw, reproducible Qwen evaluation reports for documentation."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build() -> Path:
    guarded = json.loads((ROOT / "reports" / "qwen_guarded.json").read_text())
    current = json.loads((ROOT / "reports" / "guarded_current.json").read_text())
    baseline = json.loads((ROOT / "reports" / "qwen_baseline.json").read_text())
    if guarded["total"] != 30 or current["total"] != 30 or baseline["dialogues"] != 30:
        raise ValueError("All evaluations must use the same 30 scripted dialogues")
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
        "guarded_current_semantic": {
            "passed": current["passed"],
            "total": current["total"],
            "unsafe_preconfirmation_claims": current["unsafe_preconfirmation_claims"],
            "unsafe_preconfirmation_writes": current["unsafe_preconfirmation_writes"],
        },
        "baseline": {
            "unsafe_preconfirmation_claims": baseline["unsafe_preconfirmation_claims"],
            "invalid_date_booking_claims": baseline["invalid_date_booking_claims"],
            "completed_valid_bookings": baseline["completed_valid_bookings"],
        },
        "interpretation": (
            "The current guarded run checks reply wording and database writes. The historical "
            "Qwen guarded run checked writes but did not record reply wording. The baseline "
            "reports a model boolean claim and cannot write to the database. Model modes and "
            "claim definitions differ; these authored examples do not estimate guest traffic."
        ),
    }
    output = ROOT / "docs" / "qwen_comparison.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return output


if __name__ == "__main__":
    print(build())
