"""Download the pinned Qwen model to an ignored local directory for the demo."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

from hotel_assistant.ml.qwen import MODEL_ID, MODEL_REVISION

if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "models" / "cache" / "qwen"
    snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=target,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.tiktoken"],
    )
    print(f"Qwen model ready at {target}")
