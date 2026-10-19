from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from hotel_assistant.config import Settings
from hotel_assistant.dialogue import BookingState
from hotel_assistant.services.chat import ChatService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOOKING_CONFIG = PROJECT_ROOT / "configs" / "booking.json"


def create_session_state() -> BookingState:
    """Create isolated state for one browser session from versioned policy configuration."""
    return BookingState.from_config(BOOKING_CONFIG)


def format_booking_summary(state: BookingState) -> dict[str, object]:
    values = state.to_dict()
    return {
        "Guest name": values["guest_name"] or "Not provided",
        "Check-in": values["check_in"] or "Not provided",
        "Check-out": values["check_out"] or "Not provided",
        "Guests": values["guests"] or "Not provided",
        "Room type": values["room_type"] or "Not provided",
        "Special request": values["special_request"] or "None",
    }


def reset_conversation() -> tuple[
    list[dict[str, Any]], str, None, BookingState, dict[str, object], dict[str, Any]
]:
    state = create_session_state()
    return [], "", None, state, format_booking_summary(state), {}


def create_demo(chat_service: ChatService | None = None) -> gr.Blocks:
    """Build the Phase A UI against the framework-neutral chat service."""
    service = chat_service or ChatService()

    def submit_message(
        message: str,
        history: list[dict[str, Any]],
        session_id: str | None,
        booking_state: BookingState,
    ) -> tuple[
        str,
        list[dict[str, Any]],
        str,
        BookingState,
        dict[str, object],
        dict[str, Any],
    ]:
        result = service.respond(message, session_id)
        updated_history = [
            *history,
            {"role": "user", "content": message},
            {"role": "assistant", "content": result.message},
        ]
        return (
            "",
            updated_history,
            result.session_id,
            booking_state,
            format_booking_summary(booking_state),
            result.debug,
        )

    with gr.Blocks(title="Hotel AI Assistant") as demo:
        gr.Markdown(
            "# Hotel AI Assistant\n"
            "Academic MVP · deterministic booking state · supervised ML components coming next."
        )
        session = gr.State(value=None)
        booking_state = gr.State(value=create_session_state())
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=460,
                    placeholder="Start by asking about a room or hotel policy.",
                )
                with gr.Row():
                    message = gr.Textbox(
                        label="Message",
                        placeholder="For example: I need a double room for two guests.",
                        scale=5,
                    )
                    send = gr.Button("Send", variant="primary", scale=1)
                reset = gr.Button("New conversation", variant="secondary")
            with gr.Column(scale=1):
                summary = gr.JSON(
                    label="Booking summary",
                    value=format_booking_summary(create_session_state()),
                )

        with gr.Accordion("Developer / demo mode", open=False):
            gr.Markdown(
                "Structured pipeline outputs are shown here for demonstrations. "
                "No hidden reasoning is exposed."
            )
            debug = gr.JSON(label="Intent, entities, route, and sources", value={})

        inputs = [message, chatbot, session, booking_state]
        outputs = [message, chatbot, session, booking_state, summary, debug]
        send.click(submit_message, inputs=inputs, outputs=outputs)
        message.submit(submit_message, inputs=inputs, outputs=outputs)
        reset.click(
            reset_conversation,
            outputs=[chatbot, message, session, booking_state, summary, debug],
        )

    return demo


if __name__ == "__main__":
    settings = Settings.from_env()
    create_demo().launch(server_name="0.0.0.0", server_port=settings.ui_port)
