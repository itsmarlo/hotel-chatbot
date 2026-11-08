from __future__ import annotations

from datetime import date
from html import escape
from pathlib import Path
from typing import Any

import gradio as gr

from hotel_assistant.config import Settings
from hotel_assistant.services.chat import ChatResult, ChatService

UI_CSS = (Path(__file__).parent / "ui.css").read_text(encoding="utf-8")
UI_THEME = gr.themes.Base().set(
    background_fill_primary="#ffffff",
    background_fill_primary_dark="#ffffff",
    background_fill_secondary="#f7f6fb",
    background_fill_secondary_dark="#f7f6fb",
    body_background_fill="#f7f6fb",
    body_background_fill_dark="#f7f6fb",
    body_text_color="#26243a",
    body_text_color_dark="#26243a",
    block_background_fill="#ffffff",
    block_background_fill_dark="#ffffff",
    panel_background_fill="#ffffff",
    panel_background_fill_dark="#ffffff",
    input_background_fill="#ffffff",
    input_background_fill_dark="#ffffff",
    input_placeholder_color="#777389",
    input_placeholder_color_dark="#777389",
)
WELCOME_MESSAGE = (
    "Welcome to Beit Zeitouna. I can help with a room, hotel details, or ideas for "
    "exploring Beirut. What would you like to know?"
)
STARTER_QUESTIONS = (
    ("Book a room", "I would like to book a room"),
    ("Check-in times", "What time is check-in and check-out?"),
    ("Cancellation policy", "What is your cancellation policy?"),
    ("Explore Beirut", "What are some sightseeing ideas in Beirut?"),
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
    if result.card and result.card.get("type") == "quote":
        return {
            "en": "confirm booking", "de": "Buchung bestätigen",
            "fr": "confirmer la réservation",
        }[result.card["language"]]
    if result.debug.get("selected_route") != "quote":
        return None
    if "Buchung bestätigen" in result.message:
        return "Buchung bestätigen"
    if "confirmer la réservation" in result.message:
        return "confirmer la réservation"
    return "confirm booking"


def booking_card(card: dict[str, str] | None) -> str:
    """Render only application-owned booking facts as a compact, accessible review."""
    if not card:
        return ""
    language = card.get("language", "en")
    if card.get("type") == "success":
        title = {
            "en": "Booking confirmed", "de": "Buchung bestätigt", "fr": "Réservation confirmée"
        }[language]
        label = {"en": "Reference", "de": "Referenz", "fr": "Référence"}[language]
        return (
            '<section class="bh-booking-card bh-success-card" role="status" aria-live="polite">'
            f'<p class="bh-card-kicker">{escape(title)}</p>'
            f'<h3>{escape(label)}: {escape(card["reference"])}</h3>'
            '</section>'
        )
    if card.get("type") != "quote":
        return ""
    labels = {
        "en": (
            "Review your stay", "Room", "Dates", "Nights", "Guests", "Guest",
            "Total", "No payment is taken.",
        ),
        "de": (
            "Aufenthalt prüfen", "Zimmer", "Daten", "Nächte", "Gäste", "Name",
            "Gesamt", "Es wird keine Zahlung eingezogen.",
        ),
        "fr": (
            "Vérifiez votre séjour", "Chambre", "Dates", "Nuits", "Personnes",
            "Nom", "Total", "Aucun paiement n’est prélevé.",
        ),
    }[language]
    fields = ("room", "dates", "nights", "guests", "guest_name", "total")
    details = "".join(
        f'<div><dt>{escape(label)}</dt><dd>{escape(card[field])}</dd></div>'
        for label, field in zip(labels[1:7], fields, strict=True)
    )
    return (
        '<section class="bh-booking-card" role="status" aria-live="polite">'
        f'<p class="bh-card-kicker">{escape(labels[0])}</p>'
        f'<dl>{details}</dl>'
        f'<p class="bh-card-note">{escape(labels[7])}</p>'
        '</section>'
    )


def pipeline_card(debug: dict[str, Any]) -> str:
    """Show application decisions without rendering raw model output as HTML."""
    extracted = debug.get("extracted_slots", {})
    fields = ", ".join(
        f"{key}: {'provided' if key == 'guest_name' else value}"
        for key, value in extracted.items()
    ) or "None"
    llm_status = {
        "not_used_for_route": "Not used for this step",
        "not_configured": "No hosted model configured",
        "fallback": "Verified text used after model fallback",
        "used": "Hosted model used",
    }.get(debug.get("llm_status"), "Unknown")
    rows = (
        ("Route", debug.get("selected_route")),
        ("Intent", debug.get("detected_intent")),
        ("Confidence", debug.get("confidence")),
        ("Extracted fields", fields),
        ("Source", ", ".join(debug.get("sources", [])) or "None"),
        ("Response", debug.get("response_mode")),
        ("LLM", llm_status),
        ("Proposed tool", debug.get("proposed_tool")),
        ("Database action", (
            "Reservation created" if debug.get("selected_route") == "booking_created" else "None"
        )),
    )
    details = "".join(
        f"<div><dt>{escape(label)}</dt>"
        f"<dd>{escape(str(value if value is not None else 'None'))}</dd></div>"
        for label, value in rows
    )
    return f'<dl class="bh-pipeline">{details}</dl>'


def create_demo(chat_service: ChatService | None = None) -> gr.Blocks:
    """Build a single conversation surface over the guarded chat service."""
    service = chat_service or ChatService()

    def submit_message(
        message: str,
        history: list[dict[str, Any]],
        session_id: str | None,
        pending_confirmation: str,
    ) -> tuple[Any, ...]:
        if not message.strip():
            return (
                "", history, session_id, pending_confirmation,
                gr.update(visible=bool(pending_confirmation)),
                gr.update(visible=len(history) <= 1),
                gr.update(visible=False),
                gr.update(), gr.update(),
                *(gr.skip() for _ in range(6)),
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
                gr.update(visible=False),
                gr.update(), gr.update(),
                *(gr.skip() for _ in range(6)),
            )
        result = service.respond(message, session_id)
        chat_reply = result.message
        if result.debug["selected_route"] == "quote" and result.card:
            chat_reply = {
                "en": "I found a room for your stay. Please review the details below.",
                "de": "Ich habe ein Zimmer gefunden. Bitte prüfen Sie die Angaben unten.",
                "fr": "J’ai trouvé une chambre. Vérifiez les détails ci-dessous.",
            }[result.card["language"]]
        elif result.debug["selected_route"] == "booking_created" and result.card:
            chat_reply = {
                "en": "Your booking is confirmed. Keep the reference below for your records.",
                "de": "Ihre Buchung ist bestätigt. Bewahren Sie die Referenz unten auf.",
                "fr": "Votre réservation est confirmée. Conservez la référence ci-dessous.",
            }[result.card["language"]]
        updated_history = [
            *history,
            {"role": "user", "content": message},
            {"role": "assistant", "content": chat_reply},
        ]
        phrase = confirmation_phrase(result)
        quote = result.card if result.card and result.card["type"] == "quote" else None
        return (
            "", updated_history, result.session_id, phrase or "",
            gr.update(visible=phrase is not None),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value=booking_card(result.card), visible=bool(result.card)),
            gr.update(visible=bool(result.card and result.card["type"] == "success")),
            gr.update(visible=bool(quote)),
            gr.update(value=quote["check_in"] if quote else ""),
            gr.update(value=quote["check_out"] if quote else ""),
            gr.update(value=int(quote["guests"]) if quote else 1),
            gr.update(value=quote["room_type"] if quote else "Double"),
            gr.update(value=pipeline_card(result.debug)),
        )

    def edit_booking(
        check_in: str, check_out: str, guests: int | float | None, room_type: str,
        history: list[dict[str, Any]], session_id: str | None, pending_confirmation: str,
    ) -> tuple[Any, ...]:
        try:
            arrival = date.fromisoformat(check_in.strip())
            departure = date.fromisoformat(check_out.strip())
            if (
                departure <= arrival or guests is None or int(guests) != guests
                or guests < 1 or room_type not in {"Single", "Double", "Family", "Suite"}
            ):
                raise ValueError
        except (TypeError, ValueError):
            gr.Warning(
                "Use valid YYYY-MM-DD dates, with checkout after check-in, "
                "and at least one guest."
            )
            return tuple(gr.skip() for _ in range(15))
        update = (
            f"Change my booking to {arrival.isoformat()} to {departure.isoformat()} "
            f"for {int(guests)} guests in a {room_type.lower()} room"
        )
        return submit_message(update, history, session_id, pending_confirmation)

    def show_thinking(message: str) -> Any:
        return gr.update(visible=bool(message.strip()))

    def reset_session(
        session_id: str | None,
    ) -> tuple[Any, ...]:
        service.reset(session_id)
        return (
            *fresh_chat(), gr.update(visible=False),
            gr.update(value="", visible=False), gr.update(visible=False),
            gr.update(visible=False), gr.update(value=""), gr.update(value=""),
            gr.update(value=1), gr.update(value="Double"), gr.update(value=""),
        )

    with gr.Blocks(title="Bellhob | Beit Zeitouna") as demo:
        with gr.Column(elem_id="bellhob-shell"):
            with gr.Row(elem_id="bellhob-header"):
                gr.HTML(
                    '<div class="bh-identity">'
                    '<div class="bh-mark" aria-hidden="true">B</div>'
                    '<div><h1>Bellhob</h1><p>Beit Zeitouna · Beirut</p></div>'
                    '</div>'
                )

            gr.HTML(
                '<div class="bh-intro">'
                '<p class="bh-eyebrow">YOUR DIGITAL CONCIERGE</p>'
                '<h2>How can we help with your stay?</h2>'
                '<p>Rooms, reservations, hotel details and a few favorite places around Beirut.</p>'
                '</div>'
            )

            session = gr.State(value=None)
            pending_confirmation = gr.State(value="")
            with gr.Column(elem_id="chat-panel"):
                with gr.Row(elem_id="chat-actions"):
                    reset = gr.Button(
                        "Clear conversation", variant="secondary", size="sm",
                        elem_id="clear-conversation",
                    )
                chatbot = gr.Chatbot(
                    value=fresh_chat()[0],
                    show_label=False,
                    height=360,
                    elem_id="chatbot",
                    layout="bubble",
                    buttons=[],
                    feedback_options=None,
                    allow_file_downloads=False,
                    allow_tags=False,
                )
                thinking = gr.HTML(
                    '<div class="bh-thinking" role="status" aria-live="polite">'
                    '<span class="bh-thinking-mark" aria-hidden="true">B</span>'
                    '<span>Bellhob is thinking</span>'
                    '<span class="bh-thinking-dots" aria-hidden="true">'
                    '<i></i><i></i><i></i></span>'
                    '</div>',
                    visible=False,
                    elem_id="thinking-status",
                )
                review = gr.HTML(value="", visible=False, elem_id="booking-review")
                with gr.Accordion(
                    "Edit stay details", open=False, visible=False, elem_id="booking-editor"
                ) as editor:
                    with gr.Row():
                        edit_check_in = gr.Textbox(label="Check-in (YYYY-MM-DD)")
                        edit_check_out = gr.Textbox(label="Check-out (YYYY-MM-DD)")
                    with gr.Row():
                        edit_guests = gr.Number(label="Guests", value=1, minimum=1, precision=0)
                        edit_room = gr.Dropdown(
                            label="Room type", choices=["Single", "Double", "Family", "Suite"],
                            value="Double",
                        )
                    apply_edit = gr.Button("Update quote", variant="secondary")
                confirm = gr.Button(
                    "Confirm booking",
                    variant="primary",
                    visible=False,
                    elem_id="confirm-booking",
                )
                new_chat = gr.Button(
                    "Start new chat", variant="secondary", visible=False, elem_id="new-chat"
                )
                with gr.Row(elem_id="starter-row") as starters:
                    prompt_buttons = [
                        gr.Button(label, size="sm") for label, _ in STARTER_QUESTIONS
                    ]
                with gr.Row(elem_id="composer-row"):
                    message = gr.Textbox(
                        label="Message Bellhob",
                        show_label=False,
                        placeholder="Ask about your stay or explore Beirut…",
                        lines=1,
                        max_lines=5,
                        scale=8,
                        elem_id="message-box",
                    )
                    send = gr.Button("Send", variant="primary", scale=1, elem_id="send-button")
                with gr.Accordion(
                    "How Bellhob decided · demo view", open=False, elem_id="pipeline-panel"
                ):
                    pipeline = gr.HTML(value="Ask a question to see the decision path.")
            gr.HTML(
                '<p class="bh-footnote">English · Deutsch · Français '
                '<span aria-hidden="true">·</span> '
                'Fictional hotel demo. No payment is taken.</p>'
            )

        inputs = [message, chatbot, session, pending_confirmation]
        outputs = [
            message, chatbot, session, pending_confirmation, confirm, starters, thinking,
            review, new_chat, editor, edit_check_in, edit_check_out, edit_guests, edit_room,
            pipeline,
        ]
        reset_outputs = [
            chatbot, message, session, pending_confirmation, confirm, starters, thinking,
            review, new_chat, editor, edit_check_in, edit_check_out, edit_guests, edit_room,
            pipeline,
        ]
        apply_edit.click(
            edit_booking,
            inputs=[edit_check_in, edit_check_out, edit_guests, edit_room, chatbot, session,
                    pending_confirmation],
            outputs=outputs,
        )
        for button, (_, question) in zip(prompt_buttons, STARTER_QUESTIONS, strict=True):
            button.click(lambda text=question: text, outputs=[message], queue=False).then(
                show_thinking, inputs=[message], outputs=[thinking], queue=False
            ).then(submit_message, inputs=inputs, outputs=outputs, show_progress="hidden")
        confirm.click(
            lambda phrase: phrase,
            inputs=[pending_confirmation],
            outputs=[message],
            queue=False,
        ).then(show_thinking, inputs=[message], outputs=[thinking], queue=False).then(
            submit_message, inputs=inputs, outputs=outputs, show_progress="hidden"
        )
        send.click(
            show_thinking, inputs=[message], outputs=[thinking], queue=False
        ).then(submit_message, inputs=inputs, outputs=outputs, show_progress="hidden")
        message.submit(
            show_thinking, inputs=[message], outputs=[thinking], queue=False
        ).then(submit_message, inputs=inputs, outputs=outputs, show_progress="hidden")
        reset.click(
            reset_session,
            inputs=[session],
            outputs=reset_outputs,
        )
        new_chat.click(
            reset_session,
            inputs=[session],
            outputs=reset_outputs,
        )

    return demo


if __name__ == "__main__":
    settings = Settings.from_env()
    create_demo().launch(
        server_name="0.0.0.0", server_port=settings.ui_port, css=UI_CSS, theme=UI_THEME
    )
