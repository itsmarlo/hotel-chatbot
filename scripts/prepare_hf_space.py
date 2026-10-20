"""Copy the public runtime files into a small, reviewable Hugging Face Space bundle."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("/tmp/hotel-chatbot-hf-space")


def prepare(output: Path) -> Path:
    output = output.resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("The Space bundle must be staged outside the repository")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    # Copy only runtime code and published synthetic artifacts. Local databases,
    # downloaded model weights, reports and credentials never enter the Space.
    for directory in ("app", "configs", "src", "models/trained"):
        shutil.copytree(
            ROOT / directory,
            output / directory,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
        )
    (output / "data" / "knowledge_base").mkdir(parents=True)
    shutil.copy2(
        ROOT / "data" / "knowledge_base" / "policies.json",
        output / "data" / "knowledge_base" / "policies.json",
    )
    for name in ("README.md", "requirements.txt", "app.py"):
        shutil.copy2(ROOT / "deployment" / "huggingface" / name, output / name)
    shutil.copy2(ROOT / "pyproject.toml", output / "pyproject.toml")
    forbidden = [
        p
        for p in output.rglob("*")
        if p.name in {".env", ".DS_Store"} or p.suffix == ".db"
    ]
    if forbidden:
        raise RuntimeError(f"Unsafe files in Space bundle: {forbidden}")
    return output


if __name__ == "__main__":
    print(prepare(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT))
