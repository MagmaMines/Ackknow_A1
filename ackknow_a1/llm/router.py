from ackknow_a1.config.settings import AckknowSettings


class ModelRouter:
    def __init__(self, settings: AckknowSettings) -> None:
        self.settings = settings

    def route(self, prompt: str, has_image: bool = False) -> str:
        if has_image:
            return self.settings.ollama_vision_model
        complex_markers = ["analyze", "compare", "design", "architecture", "reason", "proof", "deep"]
        return self.settings.ollama_reasoning_model if any(m in prompt.lower() for m in complex_markers) else self.settings.ollama_fast_model
