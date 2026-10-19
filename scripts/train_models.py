"""Train the two supervised baselines from repository-owned synthetic annotations."""

from __future__ import annotations

import json
from pathlib import Path

from hotel_assistant.ml.training import train_all

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(train_all(root, root / "models" / "trained"), indent=2, ensure_ascii=False))
