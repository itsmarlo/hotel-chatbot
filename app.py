"""Hugging Face Spaces entry point for the Gradio application."""

from app.gradio_app import UI_CSS, UI_THEME, create_demo
from hotel_assistant.config import Settings
from hotel_assistant.ml.replies import model_router_from_env
from hotel_assistant.services.chat import ChatService

demo = create_demo(ChatService(answer_generator=model_router_from_env()))

if __name__ == "__main__":
    settings = Settings.from_env()
    demo.launch(server_name="0.0.0.0", server_port=settings.ui_port, css=UI_CSS, theme=UI_THEME)

