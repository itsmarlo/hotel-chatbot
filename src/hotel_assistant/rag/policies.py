from __future__ import annotations

import importlib
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TOKEN = re.compile(r"[\wÀ-ÿ]+", re.UNICODE)
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
_STOP = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "at",
        "do",
        "for",
        "how",
        "i",
        "is",
        "of",
        "the",
        "to",
        "what",
        "when",
        "you",
        "der",
        "die",
        "das",
        "ist",
        "und",
        "wie",
        "wann",
        "les",
        "des",
        "est",
        "et",
        "le",
        "la",
    }
)


@dataclass(frozen=True, slots=True)
class Policy:
    id: str
    title: str
    keywords: tuple[str, ...]
    answers: dict[str, str]
    kind: str = "hotel_policy"
    sources: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class Retrieval:
    answer: str
    source: str
    score: float
    kind: str = "hotel_policy"
    sources: tuple[tuple[str, str], ...] = ()


class PolicyRetriever:
    """Retrieve authored hotel and local-guide answers with source metadata."""

    def __init__(
        self,
        path: Path,
        *,
        semantic: bool = False,
        device: str | None = None,
        query_encoder: Callable[[str], Any] | None = None,
    ) -> None:
        raw: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
        if not raw:
            raise ValueError("policy corpus is empty")
        self.policies = [
            Policy(
                id=item["id"],
                title=item["title"],
                keywords=tuple(item["keywords"]),
                answers={lang: item[lang] for lang in ("en", "de", "fr")},
                kind=item.get("kind", "hotel_policy"),
                sources=tuple(
                    (source["label"], source["url"])
                    for source in item.get("sources", [])
                ),
            )
            for item in raw
        ]
        self._model: Any = None
        self._index: Any = None
        self._device = device
        self.query_encoder = query_encoder
        self._indexed_policies: list[Policy] = []
        if semantic:
            self._build_semantic_index()

    def _build_semantic_index(self) -> None:
        try:
            faiss = importlib.import_module("faiss")
            np = importlib.import_module("numpy")
            sentence_transformers = importlib.import_module("sentence_transformers")
        except ImportError as exc:
            raise RuntimeError("Install the rag extra to enable semantic policy retrieval") from exc
        self._model = sentence_transformers.SentenceTransformer(
            MODEL_NAME,
            revision=MODEL_REVISION,
            local_files_only=os.getenv("HOTEL_RAG_LOCAL_ONLY") == "1",
            device=self._device,
        )
        texts = []
        for policy in self.policies:
            for language in ("en", "de", "fr"):
                texts.append(
                    f"{policy.title}. {' '.join(policy.keywords)}. {policy.answers[language]}"
                )
                self._indexed_policies.append(policy)
        vectors = np.asarray(self._model.encode(texts, normalize_embeddings=True), dtype="float32")
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)

    def retrieve(self, question: str, language: str = "en") -> Retrieval | None:
        if not question.strip():
            return None
        # Exact policy terms are more reliable than a borderline embedding score.
        keyword_scores = [
            sum(
                1
                for key in policy.keywords
                if re.search(rf"\b{re.escape(key.casefold())}\b", question.casefold())
            )
            for policy in self.policies
        ]
        keyword_best = max(range(len(keyword_scores)), key=keyword_scores.__getitem__)
        if keyword_scores[keyword_best]:
            policy = self.policies[keyword_best]
            return Retrieval(
                policy.answers.get(language, policy.answers["en"]),
                policy.id, 1.0, policy.kind, policy.sources,
            )
        if self._index is not None:
            np = importlib.import_module("numpy")

            encoded = (
                self.query_encoder(question)
                if self.query_encoder is not None
                else self._model.encode([question], normalize_embeddings=True)
            )
            query = np.asarray(encoded, dtype="float32").reshape(1, -1)
            scores, indices = self._index.search(query, 1)
            score = float(scores[0][0])
            if score < 0.48:
                return None
            policy = self._indexed_policies[int(indices[0][0])]
        else:
            terms = set(_TOKEN.findall(question.casefold())) - _STOP
            ranks = []
            for policy in self.policies:
                body_terms = set(_TOKEN.findall(" ".join(policy.answers.values()).casefold()))
                ranks.append(len(terms & body_terms))
            best = max(range(len(ranks)), key=ranks.__getitem__)
            score = float(ranks[best])
            if score < 2:
                return None
            policy = self.policies[best]
        return Retrieval(
            policy.answers.get(language, policy.answers["en"]),
            policy.id, score, policy.kind, policy.sources,
        )
