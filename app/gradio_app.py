from __future__ import annotations

import re
from datetime import date
from html import escape
from pathlib import Path
from typing import Any

import gradio as gr

from hotel_assistant.config import Settings
from hotel_assistant.dialogue import BookingState
from hotel_assistant.services.chat import ChatResult, ChatService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOOKING_CONFIG = PROJECT_ROOT / "configs" / "booking.json"
UI_CSS = (Path(__file__).parent / "ui.css").read_text(encoding="utf-8")

STARTER_QUESTIONS = (
    ("Plan a stay", "I would like to book a room"),
    ("Check-in times", "What time is check-in and check-out?"),
    ("Cancellation", "What is the cancellation policy?"),
)


def build_booking_request(
    guest_name: str, check_in: str | None, check_out: str | None, guests: int | None
) -> str:
    """Turn the simple form into the explicit wording the guarded parser accepts."""
    name = guest_name.strip()
    if not name or not check_in or not check_out or guests is None:
        raise gr.Error("Add your name, both dates, and the number of guests.")
    try:
        arrival = date.fromisoformat(check_in[:10])
        departure = date.fromisoformat(check_out[:10])
    except ValueError as exc:
        raise gr.Error("Choose valid arrival and departure dates.") from exc
    if departure <= arrival:
        raise gr.Error("Check-out must be after check-in.")
    if not isinstance(guests, int) or isinstance(guests, bool) or guests < 1:
        raise gr.Error("Enter at least one guest.")
    if not re.fullmatch(r"[A-Za-zÀ-ÿ]+(?:[ '-][A-Za-zÀ-ÿ]+){0,3}", name):
        raise gr.Error("Use a name of up to four words, with letters only.")
    return (
        f"My name is {name}; I would like to book a room from "
        f"{arrival.isoformat()} to {departure.isoformat()} for {guests} guests."
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
        str,
        Any,
    ]:
        if not message.strip():
            return (
                "", history, session_id or "", booking_state,
                render_booking_summary(booking_state), {}, "", gr.update(visible=False),
            )
        if len(message) > 4_000:
            updated_history = [
                *history,
                {"role": "user", "content": message[:200] + "…"},
                {"role": "assistant", "content": "Please keep messages under 4,000 characters."},
            ]
            return (
                "", updated_history, session_id or "", booking_state,
                render_booking_summary(booking_state), {}, "", gr.update(visible=False),
            )
        result = service.respond(message, session_id)
        current_state = result.booking_state or booking_state
        updated_history = [
            *history,
            {"role": "user", "content": message},
            {"role": "assistant", "content": result.message},
        ]
        phrase = confirmation_phrase(result)
        return (
            "",
            updated_history,
            result.session_id,
            current_state,
            render_booking_summary(current_state),
            result.debug,
            phrase or "",
            gr.update(visible=phrase is not None),
        )

    def reset_session(
        session_id: str | None,
    ) -> tuple[
        list[dict[str, Any]], str, None, BookingState, str, dict[str, Any],
        str, Any, str, None, None, int,
    ]:
        service.reset(session_id)
        return (*reset_conversation(), "", gr.update(visible=False), "", None, None, 2)

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
            pending_confirmation = gr.State(value="")
            with gr.Row(equal_height=False):
                with gr.Column(scale=3, min_width=320):
                    gr.Markdown("**Start here** · Choose a question, or type your own below.")
                    with gr.Row(elem_id="prompt-row"):
                        prompt_buttons = [
                            gr.Button(label, size="sm") for label, _ in STARTER_QUESTIONS
                        ]
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=480,
                        placeholder="Choose a question above or type a message to begin.",
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
                    confirm = gr.Button(
                        "Confirm this demo booking",
                        variant="primary",
                        visible=False,
                        elem_id="confirm-booking",
                    )
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
                    with gr.Accordion("Request a room", open=True):
                        gr.Markdown("Choose your dates and we’ll check a demo room for you.")
                        guest_name = gr.Textbox(label="Guest name", placeholder="Alex Kim")
                        arrival = gr.DateTime(
                            label="Check-in", include_time=False, type="string"
                        )
                        departure = gr.DateTime(
                            label="Check-out", include_time=False, type="string"
                        )
                        guest_count = gr.Number(
                            label="Guests", value=2, precision=0, minimum=1, maximum=8
                        )
                        check_rooms = gr.Button("Check rooms", variant="primary")
                    summary = gr.HTML(value=render_booking_summary(create_session_state()))

            with gr.Accordion("Developer details", open=False):
                gr.Markdown(
                    "Intent, extracted details, route, and policy sources for the last turn."
                )
                debug = gr.JSON(label="Model and routing details", value={})

        inputs = [message, chatbot, session, booking_state]
        outputs = [
            message, chatbot, session, booking_state, summary, debug,
            pending_confirmation, confirm,
        ]
        # One click sends each starter question through the same guarded chat path.
        for button, (_, question) in zip(prompt_buttons, STARTER_QUESTIONS, strict=True):
            button.click(lambda text=question: text, outputs=[message]).success(
                submit_message, inputs=inputs, outputs=outputs
            )
        check_rooms.click(
            build_booking_request,
            inputs=[guest_name, arrival, departure, guest_count],
            outputs=[message],
        ).success(submit_message, inputs=inputs, outputs=outputs)
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
            outputs=[
                chatbot, message, session, booking_state, summary, debug,
                pending_confirmation, confirm, guest_name, arrival, departure, guest_count,
            ],
        )

    return demo


if __name__ == "__main__":
    settings = Settings.from_env()
    create_demo().launch(server_name="0.0.0.0", server_port=settings.ui_port, css=UI_CSS)
