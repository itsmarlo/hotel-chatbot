from __future__ import annotations

from pathlib import Path

from hotel_assistant.config import Settings


def test_settings_use_safe_local_defaults(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HOTEL_DATABASE_PATH", raising=False)
    monkeypatch.delenv("HOTEL_LOG_LEVEL", raising=False)

    settings = Settings.from_env()

    assert settings.database_path == tmp_path / "data" / "synthetic" / "hotel.db"
    assert settings.log_level == "INFO"
    assert settings.api_url == "http://127.0.0.1:8000"
    assert settings.api_port == 8000
    assert settings.ui_port == 7860


def test_settings_accept_environment_overrides(monkeypatch, tmp_path: Path) -> None:
    database_path = tmp_path / "custom.db"
    monkeypatch.setenv("HOTEL_DATABASE_PATH", str(database_path))
    monkeypatch.setenv("HOTEL_LOG_LEVEL", "debug")
    monkeypatch.setenv("HOTEL_API_URL", "http://api:9000")
    monkeypatch.setenv("HOTEL_API_PORT", "9000")
    monkeypatch.setenv("HOTEL_UI_PORT", "7861")

    settings = Settings.from_env()

    assert settings.database_path == database_path
    assert settings.log_level == "DEBUG"
    assert settings.api_url == "http://api:9000"
    assert settings.api_port == 9000
    assert settings.ui_port == 7861


def test_settings_reject_invalid_log_level(monkeypatch) -> None:
    monkeypatch.setenv("HOTEL_LOG_LEVEL", "verbose")

    try:
        Settings.from_env()
    except ValueError as error:
        assert "HOTEL_LOG_LEVEL" in str(error)
    else:
        raise AssertionError("invalid log level should fail")
