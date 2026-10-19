from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from hotel_assistant import __version__
from hotel_assistant.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    PipelineDebug,
)
from hotel_assistant.config import Settings
from hotel_assistant.database.repository import HotelRepository
from hotel_assistant.logging_config import configure_logging
from hotel_assistant.services.chat import ChatService


def create_app(settings: Settings | None = None) -> FastAPI:
    runtime_settings = settings or Settings.from_env()
    repository = HotelRepository(runtime_settings.database_path)
    chat_service = ChatService(repository)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        configure_logging(runtime_settings.log_level)
        repository.initialize()
        repository.seed()
        yield

    app = FastAPI(
        title="Hotel AI Assistant API",
        description="Hotel booking and grounded policy assistant.",
        version=__version__,
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse, tags=["operations"])
    def health() -> HealthResponse:
        repository.count_rows("rooms")
        return HealthResponse(
            status="ok",
            service="hotel-ai-assistant",
            version=__version__,
            database="ready",
            retrieval=chat_service.retrieval_mode,
        )

    @app.post("/chat", response_model=ChatResponse, tags=["chat"])
    def chat(request: ChatRequest) -> ChatResponse:
        result = chat_service.respond(request.message, request.session_id)
        return ChatResponse(
            session_id=result.session_id,
            message=result.message,
            debug=PipelineDebug.model_validate(result.debug),
        )

    return app


app = create_app()
