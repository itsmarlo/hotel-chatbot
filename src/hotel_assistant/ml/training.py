"""Reproducible supervised baselines for the synthetic intent and BIO datasets.

The small authored datasets are useful for integration checks, not hotel traffic estimates.
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

from hotel_assistant.rag.policies import MODEL_NAME, MODEL_REVISION


def _intent_rows(root: Path) -> dict[str, list[dict[str, str]]]:
    """Read fixed splits and reject duplicate texts across those splits."""
    splits: dict[str, list[dict[str, str]]] = {key: [] for key in ("train", "dev", "test")}
    for filename, default_split in (("starter.csv", "train"), ("collected.csv", None)):
        with (root / "data" / "intent" / filename).open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                split = row.get("split") or default_split
                if split not in splits or row["is_synthetic"] != "true":
                    raise ValueError(f"Invalid intent record in {filename}")
                splits[split].append(row)
    texts = [row["text"].casefold().strip() for rows in splits.values() for row in rows]
    if len(texts) != len(set(texts)):
        raise ValueError("Duplicate intent texts would leak across evaluation splits")
    return splits


def train_intent(root: Path, output: Path) -> dict[str, Any]:
    """Choose a text representation on dev, then report the untouched test split."""
    splits = _intent_rows(root)
    x_train = [row["text"] for row in splits["train"]]
    y_train = [row["intent"] for row in splits["train"]]
    x_dev = [row["text"] for row in splits["dev"]]
    y_dev = [row["intent"] for row in splits["dev"]]
    candidates = {
        "word_1_2": TfidfVectorizer(ngram_range=(1, 2), lowercase=True),
        "char_2_5": TfidfVectorizer(analyzer="char", ngram_range=(2, 5), lowercase=True),
    }
    results: dict[str, float] = {}
    fitted: dict[str, Any] = {}
    for name, vectorizer in candidates.items():
        model = make_pipeline(vectorizer, LogisticRegression(max_iter=2000, random_state=17))
        model.fit(x_train, y_train)
        results[name] = float(f1_score(y_dev, model.predict(x_dev), average="macro"))
        fitted[name] = model
    # The same pinned multilingual encoder serves policy retrieval and intent features.
    encoder = SentenceTransformer(
        MODEL_NAME,
        revision=MODEL_REVISION,
        local_files_only=os.getenv("HOTEL_RAG_LOCAL_ONLY") == "1",
    )
    embeddings = {
        split: encoder.encode([row["text"] for row in rows], normalize_embeddings=True)
        for split, rows in splits.items()
    }
    for strength in (0.1, 1.0, 10.0):
        name = f"multilingual_c{strength}"
        model = LogisticRegression(C=strength, max_iter=2000, random_state=17)
        model.fit(embeddings["train"], y_train)
        results[name] = float(f1_score(y_dev, model.predict(embeddings["dev"]), average="macro"))
        fitted[name] = model
    # Prefer the higher-C fit on ties; it separates the tiny training set more clearly.
    chosen = max(results, key=lambda name: (results[name], name == "multilingual_c10.0"))
    model = fitted[chosen]
    test = splits["test"]
    y_test = [row["intent"] for row in test]
    predicted = model.predict(
        embeddings["test"] if chosen.startswith("multilingual") else [row["text"] for row in test]
    )
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "intent.joblib", compress=3)
    return {
        "model": chosen,
        "dev_macro_f1": results,
        "test_accuracy": float(accuracy_score(y_test, predicted)),
        "test_macro_f1": float(f1_score(y_test, predicted, average="macro")),
        "test_errors": [
            {"text": row["text"], "expected": row["intent"], "predicted": str(label)}
            for row, label in zip(test, predicted, strict=True)
            if row["intent"] != label
        ],
        "counts": {split: len(rows) for split, rows in splits.items()},
    }


def _token_features(tokens: list[str], position: int) -> dict[str, object]:
    """Use local context and token shape; never use the gold tag as a feature."""
    token = tokens[position]
    previous = tokens[position - 1] if position else "<START>"
    following = tokens[position + 1] if position + 1 < len(tokens) else "<END>"
    return {
        "word": token.casefold(),
        "prefix": token[:3].casefold(),
        "suffix": token[-3:].casefold(),
        "previous": previous.casefold(),
        "following": following.casefold(),
        "digit": token.isdigit(),
        "title": token.istitle(),
        "position_zero": position == 0,
    }


def _slot_rows(root: Path) -> dict[str, list[dict[str, Any]]]:
    path = root / "data" / "ner" / "starter.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    splits: dict[str, list[dict[str, Any]]] = {key: [] for key in ("train", "dev", "test")}
    for index, row in enumerate(rows):
        if len(row["tokens"]) != len(row["tags"]) or not row["is_synthetic"]:
            raise ValueError("Invalid BIO row")
        split = "train" if index < 6 else "dev" if index < 8 else "test"
        splits[split].append(row)
    return splits


def train_slots(root: Path, output: Path) -> dict[str, Any]:
    """Fit a token classifier; report token F1 with all BIO labels except O."""
    splits = _slot_rows(root)

    def flatten(rows: list[dict[str, Any]]) -> tuple[list[dict[str, object]], list[str]]:
        features: list[dict[str, object]] = []
        tags: list[str] = []
        for row in rows:
            for index, tag in enumerate(row["tags"]):
                features.append(_token_features(row["tokens"], index))
                tags.append(tag)
        return features, tags

    train_features, train_tags = flatten(splits["train"])
    dev_features, dev_tags = flatten(splits["dev"])
    test_features, test_tags = flatten(splits["test"])
    vectorizer = DictVectorizer()
    train_vectors = vectorizer.fit_transform(train_features)
    dev_vectors = vectorizer.transform(dev_features)
    candidate_scores: dict[str, float] = {}
    candidate_models: dict[str, Any] = {}
    for strength in (0.1, 1.0, 10.0):
        model_name = f"c{strength}"
        candidate = LogisticRegression(
            C=strength, max_iter=2000, class_weight="balanced", random_state=17
        )
        candidate.fit(train_vectors, train_tags)
        candidate_scores[model_name] = float(
            f1_score(
                dev_tags,
                candidate.predict(dev_vectors),
                labels=sorted(set(train_tags) - {"O"}),
                average="micro",
                zero_division=0,
            )
        )
        candidate_models[model_name] = candidate
    chosen = max(candidate_scores, key=lambda name: candidate_scores[name])
    model = candidate_models[chosen]
    predicted = model.predict(vectorizer.transform(test_features))
    labels = sorted(set(train_tags) | set(test_tags) - {"O"})
    labels = [tag for tag in labels if tag != "O"]
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump((vectorizer, model), output / "slots.joblib", compress=3)
    return {
        "model": chosen,
        "dev_token_f1_without_o": candidate_scores,
        "test_token_f1_without_o": float(
            f1_score(test_tags, predicted, labels=labels, average="micro", zero_division=0)
        ),
        "test_tag_counts": dict(Counter(test_tags)),
        "counts": {split: len(rows) for split, rows in splits.items()},
        "limitation": (
            "Token score on two synthetic dialogues; no entity-level generalization claim."
        ),
    }


def train_all(root: Path, output: Path) -> dict[str, Any]:
    """Persist both models and a reproducible metric report."""
    results = {"intent": train_intent(root, output), "slots": train_slots(root, output)}
    (output / "evaluation.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return results
