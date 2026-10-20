from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import gradio as gr

from hotel_assistant.config import Settings
from hotel_assistant.dialogue import BookingState
from hotel_assistant.services.chat import ChatService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOOKING_CONFIG = PROJECT_ROOT / "configs" / "booking.json"
UI_CSS = (Path(__file__).parent / "ui.css").read_text(encoding="utf-8")


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


def render_booking_summary(state: BookingState) -> str:
    """Render booking progress safely; model extracted text is always escaped."""
    values = format_booking_summary(state)
    fields = ("Guest name", "Check-in", "Check-out", "Guests", "Room type", "Special request")
    complete = sum(getattr(state, name) is not None for name in state.required_slots)
    total = len(state.required_slots)
    bars = "".join(
        f'<span class="{"filled" if index < complete else ""}"></span>'
        for index in range(total)
    )
    details = ""
    for label in fields:
        empty = values[label] in ("Not provided", "None")
        css_class = "empty" if empty else ""
        details += (
            f'<div class="bh-detail"><dt>{label}</dt>'
            f'<dd class="{css_class}">{escape(str(values[label]))}</dd></div>'
        )
    return (
        '<section class="bh-panel" aria-label="Booking summary">'
        '<div class="bh-eyebrow">Your stay</div><h2>Booking details</h2>'
        '<p class="bh-panel-copy">We’ll collect the essentials before checking rooms.</p>'
        f'<div class="bh-progress" aria-hidden="true">{bars}</div>'
        f'<div class="bh-progress-label">{complete} of {total} required details provided</div>'
        f'<dl class="bh-details">{details}</dl>'
        '<div class="bh-note">A reservation is created only after you review a quote '
        'and confirm it.</div>'
        '</section>'
    )


def reset_conversation() -> tuple[
    list[dict[str, Any]], str, None, BookingState, str, dict[str, Any]
]:
    state = create_session_state()
    return [], "", None, state, render_booking_summary(state), {}


def create_demo(chat_service: ChatService | None = None) -> gr.Blocks:
    """Build the hotel conversation UI against the guarded chat service."""
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
        str,
        dict[str, Any],
    ]:
        if not message.strip():
            return (
                "", history, session_id or "", booking_state,
                render_booking_summary(booking_state), {},
            )
        if len(message) > 4_000:
            updated_history = [
                *history,
                {"role": "user", "content": message[:200] + "…"},
                {"role": "assistant", "content": "Please keep messages under 4,000 characters."},
            ]
            return (
                "", updated_history, session_id or "", booking_state,
                render_booking_summary(booking_state), {},
            )
        result = service.respond(message, session_id)
        current_state = result.booking_state or booking_state
        updated_history = [
            *history,
            {"role": "user", "content": message},
            {"role": "assistant", "content": result.message},
        ]
        return (
            "",
            updated_history,
            result.session_id,
            current_state,
            render_booking_summary(current_state),
            result.debug,
        )

    def reset_session(
        session_id: str | None,
    ) -> tuple[list[dict[str, Any]], str, None, BookingState, str, dict[str, Any]]:
        service.reset(session_id)
        return reset_conversation()

    with gr.Blocks(title="Bellhob | Beit Zeitouna") as demo:
        with gr.Column(elem_id="bellhob-shell"):
            gr.HTML(
                '<header class="bh-header"><div class="bh-brand">'
                '<div class="bh-mark" aria-hidden="true">✦</div><div>'
                '<div class="bh-name">Bellhob</div>'
                '<div class="bh-subtitle">Beit Zeitouna · Beirut</div></div></div>'
                '<div class="bh-status"><span class="bh-status-dot"></span>Hotel assistant</div>'
                '</header>'
                '<section class="bh-intro"><div class="bh-eyebrow">Welcome</div>'
                '<h1>How can we help with your stay?</h1>'
                '<p>Ask about hotel policies or plan a room reservation in English, '
                'German, or French.</p>'
                '</section>'
            )
            session = gr.State(value=None)
            booking_state = gr.State(value=create_session_state())
            with gr.Row(equal_height=False):
                with gr.Column(scale=3, min_width=320):
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=480,
                        placeholder=(
                            "Your conversation will appear here. Choose a suggestion below "
                            "or type a question."
                        ),
                        elem_id="chatbot",
                    )
                    with gr.Row():
                        message = gr.Textbox(
                            label="Your message",
                            placeholder="Ask a question or tell us when you’d like to stay…",
                            scale=5,
                            lines=1,
                            max_lines=4,
                            elem_id="message-box",
                        )
                        send = gr.Button(
                            "Send →", variant="primary", scale=1, elem_id="send-button"
                        )
                    gr.Markdown("**Try asking**")
                    with gr.Row(elem_id="prompt-row"):
                        policy_prompt = gr.Button("What time is check-in?", size="sm")
                        booking_prompt = gr.Button("I’d like to book a room", size="sm")
                        cancel_prompt = gr.Button("What is your cancellation policy?", size="sm")
                    reset = gr.Button(
                        "Start a new conversation",
                        variant="secondary",
                        elem_id="new-conversation",
                    )
                    gr.HTML(
                        '<p class="bh-footnote">Academic demo with fictional hotel '
                        'information and reservations. No payment is taken.</p>'
                    )
                with gr.Column(scale=1, min_width=280):
                    summary = gr.HTML(value=render_booking_summary(create_session_state()))

            with gr.Accordion("Developer details", open=False):
                gr.Markdown(
                    "Intent, extracted details, route, and policy sources for the last turn."
                )
                debug = gr.JSON(label="Model and routing details", value={})

        inputs = [message, chatbot, session, booking_state]
        outputs = [message, chatbot, session, booking_state, summary, debug]
        # Suggestion buttons populate the composer so guests can edit before sending.
        policy_prompt.click(lambda: "What time is check-in?", outputs=[message])
        booking_prompt.click(lambda: "I'd like to book a room", outputs=[message])
        cancel_prompt.click(lambda: "What is your cancellation policy?", outputs=[message])
        send.click(submit_message, inputs=inputs, outputs=outputs)
        message.submit(submit_message, inputs=inputs, outputs=outputs)
        reset.click(
            reset_session,
            inputs=[session],
            outputs=[chatbot, message, session, booking_state, summary, debug],
        )

    return demo


if __name__ == "__main__":
    settings = Settings.from_env()
    create_demo().launch(server_name="0.0.0.0", server_port=settings.ui_port, css=UI_CSS)
