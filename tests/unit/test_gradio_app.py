from __future__ import annotations

from pathlib import Path

import gradio as gr

from app.gradio_app import (
    WELCOME_MESSAGE,
    booking_card,
    confirmation_phrase,
    create_demo,
    fresh_chat,
)
from hotel_assistant.database.repository import HotelRepository
from hotel_assistant.services.chat import ChatResult, ChatService


def test_create_demo_builds_guest_booking_interface() -> None:
    demo = create_demo()

    assert isinstance(demo, gr.Blocks)
    assert any(isinstance(block, gr.Number) for block in demo.blocks.values())
    assert not any(isinstance(block, gr.JSON) for block in demo.blocks.values())
    assert not any(
        isinstance(block, gr.Accordion) and block.label == "How Bellhob decided · demo view"
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


def test_booking_card_shows_review_facts_and_escapes_guest_name() -> None:
    html = booking_card({
        "type": "quote", "language": "en", "room": "202 · Double",
        "dates": "01.10.2030 – 03.10.2030", "nights": "2", "guests": "2",
        "guest_name": "<script>alert(1)</script>", "total": "US$358.00",
    })

    assert "Review your stay" in html
    assert "US$358.00" in html
    assert "No payment is taken." in html
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_booking_success_card_has_reference() -> None:
    html = booking_card({"type": "success", "language": "en", "reference": "HTL-123"})

    assert "Booking confirmed" in html
    assert "HTL-123" in html
    assert 'role="status"' in html


def test_booking_flow_supplies_review_and_success_cards(tmp_path: Path) -> None:
    service = ChatService(repository=HotelRepository(tmp_path / "hotel.db"))
    quote = service.respond(
        "Book a double room 2030-08-03 to 2030-08-05 for 2 guests. My name is Sam Lee."
    )

    assert quote.card is not None
    assert quote.card["total"] == "US$358.00"
    assert quote.card["nights"] == "2"
    assert quote.card["guests"] == "2"
    assert quote.card["guest_name"] == "Sam Lee"
    confirmed = service.respond("confirm booking", quote.session_id)
    assert confirmed.card is not None
    assert confirmed.card["type"] == "success"
    assert confirmed.card["reference"].startswith("HTL-")


def test_policy_question_keeps_quote_available_to_confirm(tmp_path: Path) -> None:
    service = ChatService(repository=HotelRepository(tmp_path / "hotel.db"))
    quote = service.respond(
        "Book a double room 2030-08-03 to 2030-08-05 for 2 guests. My name is Sam Lee."
    )
    policy = service.respond("What time is check-in?", quote.session_id)

    assert policy.debug["selected_route"] == "policy"
    assert policy.card == quote.card
    assert confirmation_phrase(policy) == "confirm booking"
    assert "15:00" in policy.message


def test_changed_quote_replaces_old_details_before_confirmation(tmp_path: Path) -> None:
    service = ChatService(repository=HotelRepository(tmp_path / "hotel.db"))
    quote = service.respond(
        "Book a double room 2030-08-03 to 2030-08-05 for 2 guests. My name is Sam Lee."
    )
    changed = service.respond(
        "Change my booking to 2030-08-04 to 2030-08-07 for 2 guests in a double room",
        quote.session_id,
    )

    assert changed.debug["selected_route"] == "quote"
    assert changed.card is not None
    assert changed.card["dates"] == "04.08.2030 – 07.08.2030"
    assert changed.card["total"] == "US$537.00"
    assert changed.card != quote.card


def test_invalid_edit_cannot_leave_hidden_quote_confirmable(tmp_path: Path) -> None:
    service = ChatService(repository=HotelRepository(tmp_path / "hotel.db"))
    quote = service.respond(
        "Book a double room 2030-08-03 to 2030-08-05 for 2 guests. My name is Sam Lee."
    )
    invalid = service.respond(
        "Change my booking to 2030-08-07 to 2030-08-04 for 2 guests in a double room",
        quote.session_id,
    )
    confirmation = service.respond("confirm booking", quote.session_id)

    assert invalid.debug["selected_route"] == "clarification"
    assert invalid.card is None
    assert confirmation.debug["selected_route"] == "clarification"
    assert "search for rooms first" in confirmation.message


def test_ui_keeps_review_during_policy_answer_and_updates_edit(tmp_path: Path) -> None:
    service = ChatService(repository=HotelRepository(tmp_path / "hotel.db"))
    demo = create_demo(service)
    submit = next(
        fn.fn for fn in demo.fns.values()
        if getattr(fn.fn, "__name__", "") == "submit_message"
    )
    edit = next(
        fn.fn for fn in demo.fns.values()
        if getattr(fn.fn, "__name__", "") == "edit_booking"
    )
    first = submit(
        "Book a double room 2030-08-03 to 2030-08-05 for 2 guests. My name is Sam Lee.",
        fresh_chat()[0], None, "",
    )
    policy = submit("What time is check-in?", first[1], first[2], first[3])
    changed = edit("2030-08-04", "2030-08-07", 2, "Double", policy[1], policy[2], policy[3])

    assert "15:00" in policy[1][-1]["content"]
    assert policy[7]["visible"] is True
    assert policy[4]["visible"] is True
    assert changed[10]["value"] == "2030-08-04"
    assert "US$537.00" in changed[7]["value"]
