from __future__ import annotations

import gradio as gr
import pytest

from app.gradio_app import (
    build_booking_request,
    confirmation_phrase,
    create_demo,
    create_session_state,
    format_booking_summary,
    render_booking_summary,
    reset_conversation,
)
from hotel_assistant.services.chat import ChatResult


def test_create_demo_builds_gradio_blocks() -> None:
    demo = create_demo()

    assert isinstance(demo, gr.Blocks)


def test_session_state_is_isolated_between_conversations() -> None:
    first = create_session_state()
    second = create_session_state()

    first.update(guest_name="Sarah Miller")

    assert second.guest_name is None


def test_booking_summary_is_user_facing() -> None:
    state = create_session_state()
    state.update(guest_name="Sarah Miller", guests=2)

    summary = format_booking_summary(state)

    assert summary["Guest name"] == "Sarah Miller"
    assert summary["Guests"] == 2
    assert summary["Check-in"] == "Not provided"


def test_reset_conversation_returns_fresh_state() -> None:
    state = create_session_state()
    state.update(guest_name="Sarah Miller")

    history, message, session_id, fresh_state, summary, demo_details = reset_conversation()

    assert history == []
    assert message == ""
    assert session_id is None
    assert fresh_state.guest_name is None
    assert "Not provided" in summary
    assert demo_details == {}


def test_booking_summary_escapes_guest_text() -> None:
    state = create_session_state()
    state.update(guest_name="<script>alert(1)</script>")

    rendered = render_booking_summary(state)

    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered


def test_room_form_builds_a_request_the_chat_can_parse() -> None:
    request = build_booking_request("Alex Kim", "2026-11-12", "2026-11-15", 2)

    assert request == (
        "My name is Alex Kim; I would like to book a room from "
        "2026-11-12 to 2026-11-15 for 2 guests."
    )


def test_room_form_rejects_reversed_dates() -> None:
    with pytest.raises(gr.Error, match="Check-out must be after check-in"):
        build_booking_request("Alex Kim", "2026-11-15", "2026-11-12", 2)


def test_confirmation_action_requires_a_quote() -> None:
    quote = ChatResult("session", "Reply exactly: confirm booking", {"selected_route": "quote"})
    policy = ChatResult("session", "Check-in starts at 15:00", {"selected_route": "policy"})

    assert confirmation_phrase(quote) == "confirm booking"
    assert confirmation_phrase(policy) is None
