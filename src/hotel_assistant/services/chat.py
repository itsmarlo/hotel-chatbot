from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class ChatResult:
    """Framework-neutral result returned by the dialogue entry point."""

    session_id: str
    message: str
    debug: dict[str, Any] = field(default_factory=dict)


class ChatService:
    """Phase A chat seam; supervised ML routing is deliberately deferred to Phase B."""

    def respond(self, message: str, session_id: str | None = None) -> ChatResult:
        if not message.strip():
            raise ValueError("message must not be empty")
        return ChatResult(
            session_id=session_id or str(uuid4()),
            message=(
                "Phase A is online. The isolated booking state and application shell are ready; "
                "trained intent detection will be added in Phase B."
            ),
            debug={
                "detected_intent": None,
                "confidence": None,
                "extracted_slots": {},
                "selected_route": "phase_a_placeholder",
                "sources": [],
            },
        )
