from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from hotel_assistant.config import Settings
from hotel_assistant.services.chat import ChatResult, ChatService

UI_CSS = (Path(__file__).parent / "ui.css").read_text(encoding="utf-8")
UI_THEME = gr.themes.Base().set(
    body_background_fill="#0d1416",
    body_background_fill_dark="#0d1416",
    body_text_color="#edf4f0",
    body_text_color_dark="#edf4f0",
    block_background_fill="#142023",
    block_background_fill_dark="#142023",
    input_background_fill="#1b2a2d",
    input_background_fill_dark="#1b2a2d",
    input_placeholder_color="#a6b8b2",
    input_placeholder_color_dark="#a6b8b2",
)
WELCOME_MESSAGE = (
    "Welcome to Beit Zeitouna. I can help you plan a stay or answer a question "
    "about the hotel. What can I help with?"
)
STARTER_QUESTIONS = (
    ("Book a room", "I would like to book a room"),
    ("Check-in times", "What time is check-in and check-out?"),
    ("Cancellation policy", "What is your cancellation policy?"),
)


def fresh_chat() -> tuple[list[dict[str, str]], str, None, str, Any, Any]:
    """Show the welcome message and clear state when a conversation is reset."""
    return (
        [{"role": "assistant", "content": WELCOME_MESSAGE}],
        "",
        None,
        "",
        gr.update(visible=False),
        gr.update(visible=True),
    )


def confirmation_phrase(result: ChatResult) -> str | None:
    """Only offer the confirm action when the booking guard has issued a quote."""
    if result.debug.get("selected_route") != "quote":
        return None
    if "Buchung bestätigen" in result.message:
        return "Buchung bestätigen"
    if "confirmer la réservation" in result.message:
        return "confirmer la réservation"
    return "confirm booking"


def create_demo(chat_service: ChatService | None = None) -> gr.Blocks:
    """Build a single conversation surface over the guarded chat service."""
    service = chat_service or ChatService()

    def submit_message(
        message: str,
        history: list[dict[str, Any]],
        session_id: str | None,
        pending_confirmation: str,
    ) -> tuple[str, list[dict[str, Any]], str | None, str, Any, Any]:
        if not message.strip():
            return (
                "", history, session_id, pending_confirmation,
                gr.update(visible=bool(pending_confirmation)),
                gr.update(visible=len(history) <= 1),
            )
        if len(message) > 4_000:
            updated_history = [
                *history,
                {"role": "user", "content": message[:200] + "…"},
                {"role": "assistant", "content": "Please keep messages under 4,000 characters."},
            ]
            return (
                "", updated_history, session_id, pending_confirmation,
                gr.update(visible=bool(pending_confirmation)),
                gr.update(visible=False),
            )
        result = service.respond(message, session_id)
        updated_history = [
            *history,
            {"role": "user", "content": message},
            {"role": "assistant", "content": result.message},
        ]
        phrase = confirmation_phrase(result)
        return (
            "", updated_history, result.session_id, phrase or "",
            gr.update(visible=phrase is not None),
            gr.update(visible=False),
        )

    def reset_session(
        session_id: str | None,
    ) -> tuple[list[dict[str, str]], str, None, str, Any, Any]:
        service.reset(session_id)
        return fresh_chat()

    with gr.Blocks(title="Bellhob | Beit Zeitouna") as demo:
        with gr.Column(elem_id="bellhob-shell"):
            with gr.Row(elem_id="bellhob-header"):
                gr.HTML(
                    '<div class="bh-identity">'
                    '<div class="bh-mark" aria-hidden="true">B</div>'
                    '<div><h1>Bellhob</h1><p>Beit Zeitouna · Beirut</p></div>'
                    '</div>'
                )
                reset = gr.Button("New chat", variant="secondary", size="sm", elem_id="new-chat")

            session = gr.State(value=None)
            pending_confirmation = gr.State(value="")
            with gr.Column(elem_id="chat-panel"):
                chatbot = gr.Chatbot(
                    value=fresh_chat()[0],
                    show_label=False,
                    height=560,
                    elem_id="chatbot",
                    layout="bubble",
                    buttons=[],
                    feedback_options=None,
                    allow_file_downloads=False,
                    allow_tags=False,
                )
                with gr.Row(elem_id="starter-row") as starters:
                    prompt_buttons = [
                        gr.Button(label, size="sm") for label, _ in STARTER_QUESTIONS
                    ]
                with gr.Row(elem_id="composer-row"):
                    message = gr.Textbox(
                        label="Message Bellhob",
                        show_label=False,
                        placeholder="Ask a question or tell me about your stay…",
                        lines=1,
                        max_lines=5,
                        scale=8,
                        elem_id="message-box",
                    )
                    send = gr.Button("Send", variant="primary", scale=1, elem_id="send-button")
                confirm = gr.Button(
                    "Confirm this demo booking",
                    variant="primary",
                    visible=False,
                    elem_id="confirm-booking",
                )
            gr.HTML(
                '<p class="bh-footnote">English · Deutsch · Français '
                '<span aria-hidden="true">·</span> '
                'Fictional hotel demo. No payment is taken.</p>'
            )

        inputs = [message, chatbot, session, pending_confirmation]
        outputs = [message, chatbot, session, pending_confirmation, confirm, starters]
        for button, (_, question) in zip(prompt_buttons, STARTER_QUESTIONS, strict=True):
            button.click(lambda text=question: text, outputs=[message]).success(
                submit_message, inputs=inputs, outputs=outputs
            )
        confirm.click(
            lambda phrase: phrase,
            inputs=[pending_confirmation],
            outputs=[message],
        ).success(submit_message, inputs=inputs, outputs=outputs)
        send.click(submit_message, inputs=inputs, outputs=outputs)
        message.submit(submit_message, inputs=inputs, outputs=outputs)
        reset.click(
            reset_session,
            inputs=[session],
            outputs=[chatbot, message, session, pending_confirmation, confirm, starters],
        )

    return demo


if __name__ == "__main__":
    settings = Settings.from_env()
    create_demo().launch(
        server_name="0.0.0.0", server_port=settings.ui_port, css=UI_CSS, theme=UI_THEME
    )
