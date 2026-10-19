from __future__ import annotations

import logging

from hotel_assistant.config import Settings
from hotel_assistant.database.repository import HotelRepository
from hotel_assistant.logging_config import configure_logging


def main() -> None:
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    repository = HotelRepository(settings.database_path)
    repository.initialize()
    repository.seed()
    logging.getLogger(__name__).info(
        "Hotel database initialized",
        extra={"event": "database_initialized", "database_path": settings.database_path},
    )


if __name__ == "__main__":
    main()

