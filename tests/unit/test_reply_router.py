from __future__ import annotations

from typing import Any

import httpx
import pytest

from hotel_assistant.ml.replies import (
    ModelRouter,
    OpenAICompatibleBackend,
    ReplyRequest,
    model_router_from_env,
)


class StubBackend:
    def __init__(self, name: str, result: str | Exception) -> None:
        self.name = name
        self.result = result
        self.calls = 0

    def generate(self, request: ReplyRequest) -> str:
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_router_switches_models_after_a_failure() -> None:
    primary = StubBackend("primary", httpx.TimeoutException("slow"))
    fallback = StubBackend("fallback", "Check-in starts at 15:00.")
    router = ModelRouter([primary, fallback])

    reply = router("When is check-in?", "Check-in starts at 15:00.", "en", "policy")

    assert reply == "Check-in starts at 15:00."
    assert primary.calls == 1
    assert fallback.calls == 1


def test_router_reports_all_failed_models() -> None:
    router = ModelRouter(
        [StubBackend("one", ValueError("bad")), StubBackend("two", RuntimeError("down"))]
    )

    with pytest.raises(RuntimeError, match="one: ValueError, two: RuntimeError"):
        router("Hello", "Welcome.", "en", "greeting")


def test_compatible_backend_sends_grounded_prompt_and_validates_response() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(__import__("json").loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "Willkommen!"}}]})

    backend = OpenAICompatibleBackend(
        provider="huggingface",
        endpoint="https://example.test/v1/chat/completions",
        api_key="secret",
        model="strong-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    reply = backend.generate(ReplyRequest("Hallo", "Willkommen im Hotel.", "de", "greeting"))

    assert reply == "Willkommen!"
    assert captured["model"] == "strong-model"
    assert "Verified hotel text: Willkommen im Hotel." in captured["messages"][1]["content"]
    assert captured["temperature"] == 0.1


def test_openai_reasoning_model_disables_reasoning_for_chat_completion() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(__import__("json").loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "Welcome."}}]})

    backend = OpenAICompatibleBackend(
        provider="openai",
        endpoint="https://example.test/v1/chat/completions",
        api_key="secret",
        model="gpt-6-luna",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    backend.generate(ReplyRequest("Hello", "Welcome.", "en", "greeting"))

    assert captured["reasoning_effort"] == "none"


def test_router_from_env_uses_provider_and_model_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    monkeypatch.setenv("HF_TOKEN", "hf-key")
    monkeypatch.setenv("HOTEL_LLM_ORDER", "openrouter,huggingface")
    monkeypatch.setenv("HOTEL_OPENROUTER_MODELS", "model-a,model-b")
    monkeypatch.setenv("HOTEL_HF_MODELS", "model-c")

    router = model_router_from_env()

    assert router is not None
    assert [backend.name for backend in router.backends] == [
        "openrouter/model-a",
        "openrouter/model-b",
        "huggingface/model-c",
    ]


def test_router_from_env_is_disabled_without_provider_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in ("HF_TOKEN", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    assert model_router_from_env() is None
