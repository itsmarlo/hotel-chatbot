"""Load repository-trained models and return advisory language predictions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from hotel_assistant.ml.training import _token_features

_TOKENS = re.compile(r"[\wÀ-ÿ]+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True, slots=True)
class ModelPrediction:
    intent: str
    confidence: float
    entities: dict[str, str]


class TrainedNLP:
    """Run supervised models while keeping their outputs outside booking authority."""

    def __init__(self, directory: Path, encoder: Any) -> None:
        self.encoder = encoder
        # Only load artifacts trained and shipped with this repository; joblib uses pickle.
        self.intent = joblib.load(directory / "intent.joblib")
        self.slot_vectorizer, self.slot_model = joblib.load(directory / "slots.joblib")

    def predict(self, message: str) -> ModelPrediction:
        vector = self.encoder.encode([message], normalize_embeddings=True)
        probabilities = self.intent.predict_proba(vector)[0]
        best = int(probabilities.argmax())
        tokens = _TOKENS.findall(message)
        features = [_token_features(tokens, index) for index in range(len(tokens))]
        tags = self.slot_model.predict(self.slot_vectorizer.transform(features))
        entities: dict[str, str] = {}
        current_name: str | None = None
        current_tokens: list[str] = []

        def flush() -> None:
            if current_name and current_tokens:
                entities[current_name] = " ".join(current_tokens)

        for token, tag in zip(tokens, tags, strict=True):
            if tag.startswith("B-"):
                flush()
                current_name = str(tag[2:])
                current_tokens = [token]
            elif tag.startswith("I-") and current_name == tag[2:]:
                current_tokens.append(token)
            else:
                flush()
                current_name = None
                current_tokens = []
        flush()
        return ModelPrediction(
            intent=str(self.intent.classes_[best]),
            confidence=float(probabilities[best]),
            entities=entities,
        )
