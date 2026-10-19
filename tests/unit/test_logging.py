from __future__ import annotations

import json
import logging

from hotel_assistant.logging_config import JsonFormatter


def test_json_formatter_emits_structured_context() -> None:
    record = logging.LogRecord(
        name="hotel_assistant.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=12,
        msg="database ready",
        args=(),
        exc_info=None,
    )
    record.event = "database_initialized"

    payload = json.loads(JsonFormatter().format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "hotel_assistant.test"
    assert payload["message"] == "database ready"
    assert payload["event"] == "database_initialized"
