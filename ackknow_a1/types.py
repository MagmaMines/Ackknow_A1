from dataclasses import dataclass, field
from typing import Any


@dataclass
class Evidence:
    source: str
    url: str | None = None
    confidence: float = 0.5


@dataclass
class QueryContext:
    user_id: str
    locale: str = "en-US"
    consent_memory: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AckResponse:
    answer: str
    reasoning_summary: str
    evidence: list[Evidence]
    safeguards: list[str]
