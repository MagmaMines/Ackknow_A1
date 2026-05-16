from __future__ import annotations

from ackknow_a1.config.settings import AckknowSettings
from ackknow_a1.types import RouteKind


class ModelRouter:
    def __init__(self, settings: AckknowSettings) -> None:
        self.settings = settings

    def route(self, prompt: str, has_image: bool = False) -> tuple[RouteKind, str]:
        p = prompt.lower()
        if has_image:
            return RouteKind.VISION, self.settings.ollama_vision_model
        if any(k in p for k in ["bug", "code", "refactor", "python", "api", "sql"]):
            return RouteKind.CODE, self.settings.ollama_reasoning_model
        if any(k in p for k in ["compare", "design", "prove", "analyze", "strategy", "deep"]):
            return RouteKind.REASON, self.settings.ollama_reasoning_model
        return RouteKind.FAST, self.settings.ollama_fast_model
