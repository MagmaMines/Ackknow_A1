from datetime import datetime
from pydantic import BaseModel, Field


class Source(BaseModel):
    title: str
    url: str | None = None
    score: float = 0.0
    snippet: str = ""


class ChatMessage(BaseModel):
    role: str
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AckRequest(BaseModel):
    user_id: str
    prompt: str
    locale: str = "en-IN"
    image_b64: str | None = None


class AckResponse(BaseModel):
    answer: str
    confidence: float
    reasoning_summary: str
    sources: list[Source]
    safety_notes: list[str]
    model_used: str
