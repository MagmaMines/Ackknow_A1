from __future__ import annotations

import re

from ackknow_a1.config.settings import AckknowSettings
from ackknow_a1.types import RouteKind

CODE_PATTERN = re.compile(
    r"\b(bugs?|debug\w*|code|coding|refactor\w*|python|javascript|typescript|java|kotlin|swift|rust|golang|c\+\+|c#|"
    r"php|ruby|bash|shell|sql|html|css|react|django|flask|fastapi|node(?:js)?|api|function|class|method|script|"
    r"regex|algorithm|compile\w*|traceback|exception|stack ?trace|unit ?tests?|pytest|implement\w*|program\w*)\b"
)
CODE_MARKERS = ("```", "def ", "import ", "#include", "function(", "=>", "SELECT ", "Traceback (most recent call last)")
REASON_PATTERN = re.compile(r"\b(compare|design|prove|analy[sz]e|strategy|deep|explain why|trade-?offs?)\b")


def is_code_request(prompt: str) -> bool:
    return bool(CODE_PATTERN.search(prompt.lower())) or any(m in prompt for m in CODE_MARKERS)


class ModelRouter:
    def __init__(self, settings: AckknowSettings) -> None:
        self.settings = settings

    def route(self, prompt: str, has_image: bool = False) -> tuple[RouteKind, str]:
        if has_image:
            return RouteKind.VISION, self.settings.ollama_vision_model
        if is_code_request(prompt):
            return RouteKind.CODE, self.settings.ollama_code_model
        if REASON_PATTERN.search(prompt.lower()):
            return RouteKind.REASON, self.settings.ollama_reasoning_model
        return RouteKind.FAST, self.settings.ollama_fast_model
