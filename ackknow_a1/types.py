from __future__ import annotations

from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class RouteKind(str, Enum):
    FAST = "fast"
    REASON = "reason"
    CODE = "code"
    VISION = "vision"


class Source(BaseModel):
    title: str
    url: str | None = None
    score: float = 0.0
    snippet: str = ""
    kind: str = "local"


class ChatMessage(BaseModel):
    role: str
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AckRequest(BaseModel):
    user_id: str
    prompt: str
    locale: str = "en-IN"
    image_b64: str | None = None


class ReasoningTrace(BaseModel):
    route: RouteKind
    checks: list[str]
    risk_flags: list[str] = Field(default_factory=list)


class AckResponse(BaseModel):
    answer: str
    confidence: float
    reasoning_summary: str
    sources: list[Source]
    safety_notes: list[str]
    model_used: str
    route: RouteKind
