from __future__ import annotations

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
EXPECTED_INTENTS = {
    "BOOKING",
    "CANCEL_BOOKING",
    "MODIFY_BOOKING",
    "HOTEL_POLICY",
    "GREETING",
    "OTHER",
}


def test_intent_starter_data_contains_every_intent() -> None:
    path = PROJECT_ROOT / "data" / "intent" / "starter.csv"

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) >= 30
    assert {row["intent"] for row in rows} == EXPECTED_INTENTS
    assert all(row["text"].strip() for row in rows)
    assert all(row["is_synthetic"] == "true" for row in rows)


def test_ner_starter_data_uses_aligned_bio_annotations() -> None:
    path = PROJECT_ROOT / "data" / "ner" / "starter.jsonl"
    examples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert len(examples) >= 10
    assert all(len(example["tokens"]) == len(example["tags"]) for example in examples)
    assert all(example["is_synthetic"] is True for example in examples)
    flattened_tags = {tag for example in examples for tag in example["tags"]}
    assert {
        "B-GUEST_NAME",
        "B-CHECKIN_DATE",
        "B-CHECKOUT_DATE",
        "B-GUEST_COUNT",
        "B-ROOM_TYPE",
        "B-SPECIAL_REQUEST",
    }.issubset(flattened_tags)
