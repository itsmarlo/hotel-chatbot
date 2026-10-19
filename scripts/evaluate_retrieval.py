"""Evaluate source selection on manually authored, paraphrased policy questions."""

from __future__ import annotations

import json
import os
from pathlib import Path

from hotel_assistant.rag.policies import PolicyRetriever

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    corpus = ROOT / "data" / "knowledge_base"
    questions = json.loads((corpus / "evaluation.json").read_text(encoding="utf-8"))
    mode = os.getenv("HOTEL_RAG_MODE", "lexical")
    retriever = PolicyRetriever(corpus / "policies.json", semantic=mode == "semantic")
    correct = 0
    supported = 0
    rejected = 0
    unsupported = 0
    for item in questions:
        result = retriever.retrieve(item["question"], item["language"])
        predicted = result.source if result else None
        if item["source"] is None:
            unsupported += 1
            rejected += predicted is None
        else:
            supported += 1
            correct += predicted == item["source"]
        print(f"{item['language']} expected={item['source']} predicted={predicted}")
    print(f"Hit@1: {correct}/{supported} = {correct / supported:.1%} ({mode})")
    print(f"Unsupported rejected: {rejected}/{unsupported}")


if __name__ == "__main__":
    main()
