"""Space entry point with grounded Qwen replies and guarded booking state."""

import importlib
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("HOTEL_RAG_MODE", "semantic")
os.environ.setdefault("HOTEL_NLP_MODE", "trained")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

spaces = importlib.import_module("spaces") if importlib.util.find_spec("spaces") else None

from app.gradio_app import UI_CSS, UI_THEME, create_demo  # noqa: E402
from hotel_assistant.ml.qwen import QwenModel  # noqa: E402
from hotel_assistant.rag.policies import PolicyRetriever  # noqa: E402
from hotel_assistant.services.chat import ChatService  # noqa: E402

ROOT = Path(__file__).resolve().parent
retriever = PolicyRetriever(
    ROOT / "data" / "knowledge_base" / "policies.json", semantic=True, device="cpu"
)
qwen = QwenModel()
if spaces is not None:
    # ZeroGPU records CUDA placement at module load and supplies the real GPU on calls.
    qwen.model.to("cuda")


def generate_reply(message: str, fact: str, language: str, route: str) -> str:
    return qwen.generate_reply(message, fact, language, route)


if spaces is not None:
    generate_reply = spaces.GPU(duration=40)(generate_reply)


demo = create_demo(chat_service=ChatService(retriever=retriever, answer_generator=generate_reply))

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        css=UI_CSS,
        theme=UI_THEME,
    )
