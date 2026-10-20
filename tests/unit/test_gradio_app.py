from __future__ import annotations

import gradio as gr

from app.gradio_app import (
    create_demo,
    create_session_state,
    format_booking_summary,
    render_booking_summary,
    reset_conversation,
)


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
