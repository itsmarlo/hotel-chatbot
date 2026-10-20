"""Hugging Face Spaces entry point for the Gradio application."""

from app.gradio_app import UI_CSS, create_demo
from hotel_assistant.config import Settings

demo = create_demo()

if __name__ == "__main__":
    settings = Settings.from_env()
    demo.launch(server_name="0.0.0.0", server_port=settings.ui_port, css=UI_CSS)

