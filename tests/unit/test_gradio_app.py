from __future__ import annotations

import gradio as gr

from app.gradio_app import WELCOME_MESSAGE, confirmation_phrase, create_demo, fresh_chat
from hotel_assistant.services.chat import ChatResult


def test_create_demo_builds_chat_only_interface() -> None:
    demo = create_demo()

    assert isinstance(demo, gr.Blocks)
    assert not any(
        isinstance(block, (gr.DateTime, gr.Number, gr.JSON))
        for block in demo.blocks.values()
    )


def test_fresh_chat_starts_with_a_welcome_message() -> None:
    history, message, session_id, pending, _, _ = fresh_chat()

    assert history == [{"role": "assistant", "content": WELCOME_MESSAGE}]
    assert message == ""
    assert session_id is None
    assert pending == ""


def test_confirmation_action_requires_a_quote() -> None:
    quote = ChatResult("session", "Reply exactly: confirm booking", {"selected_route": "quote"})
    policy = ChatResult("session", "Check-in starts at 15:00", {"selected_route": "policy"})

    assert confirmation_phrase(quote) == "confirm booking"
    assert confirmation_phrase(policy) is None


def test_confirmation_action_uses_quote_language() -> None:
    german = ChatResult("s", "Buchung bestätigen", {"selected_route": "quote"})
    french = ChatResult("s", "confirmer la réservation", {"selected_route": "quote"})

    assert confirmation_phrase(german) == "Buchung bestätigen"
    assert confirmation_phrase(french) == "confirmer la réservation"
