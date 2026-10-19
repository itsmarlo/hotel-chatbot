from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings loaded from HOTEL_* environment variables."""

    database_path: Path
    log_level: str = "INFO"
    api_url: str = "http://127.0.0.1:8000"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    ui_port: int = 7860

    @classmethod
    def from_env(cls) -> Settings:
        default_database = Path.cwd() / "data" / "synthetic" / "hotel.db"
        log_level = os.getenv("HOTEL_LOG_LEVEL", "INFO").upper()
        if log_level not in _LOG_LEVELS:
            raise ValueError(
                f"HOTEL_LOG_LEVEL must be one of {sorted(_LOG_LEVELS)}, got {log_level!r}"
            )
        return cls(
            database_path=Path(os.getenv("HOTEL_DATABASE_PATH", default_database)),
            log_level=log_level,
            api_url=os.getenv("HOTEL_API_URL", "http://127.0.0.1:8000").rstrip("/"),
            api_host=os.getenv("HOTEL_API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("HOTEL_API_PORT", "8000")),
            ui_port=int(os.getenv("HOTEL_UI_PORT", "7860")),
        )
