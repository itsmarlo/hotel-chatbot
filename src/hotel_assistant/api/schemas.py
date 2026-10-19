from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    database: str
    retrieval: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=4_000)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)


class PipelineDebug(BaseModel):
    detected_intent: str | None
    confidence: float | None
    extracted_slots: dict[str, Any]
    selected_route: str
    sources: list[str]
    model_entities: dict[str, str] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    session_id: str
    message: str
    debug: PipelineDebug

