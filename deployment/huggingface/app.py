"""Space entry point with the tested semantic retrieval and trained intent mode."""

import importlib
import importlib.util
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("HOTEL_RAG_MODE", "semantic")
os.environ.setdefault("HOTEL_NLP_MODE", "trained")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

spaces = importlib.import_module("spaces") if importlib.util.find_spec("spaces") else None

from app.gradio_app import create_demo  # noqa: E402
from hotel_assistant.rag.policies import PolicyRetriever  # noqa: E402
from hotel_assistant.services.chat import ChatService  # noqa: E402

ROOT = Path(__file__).resolve().parent
retriever = PolicyRetriever(
    ROOT / "data" / "knowledge_base" / "policies.json", semantic=True, device="cpu"
)


def gpu_embedding(question: str) -> list[list[float]]:
    """Encode one query in a stateless ZeroGPU worker when hardware is available."""
    model = retriever._model
    model.to("cuda")
    return model.encode([question], normalize_embeddings=True).tolist()


if spaces is not None:
    gpu_embedding = spaces.GPU(duration=20)(gpu_embedding)


gpu_available = True


def encode_question(question: str) -> list[list[float]]:
    global gpu_available
    if gpu_available and spaces is not None:
        try:
            return gpu_embedding(question)
        except Exception:
            # Keep retrieval available if the shared ZeroGPU worker fails.
            logging.exception("ZeroGPU query embedding failed; using CPU")
            gpu_available = False
    return retriever._model.encode([question], normalize_embeddings=True).tolist()


retriever.query_encoder = encode_question
demo = create_demo(chat_service=ChatService(retriever=retriever))

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
