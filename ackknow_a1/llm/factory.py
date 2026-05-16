from ackknow_a1.config.settings import AckknowSettings
from ackknow_a1.llm.own_model import OwnAckknowLLM, OwnModelConfig
from ackknow_a1.llm.providers import OllamaClient


class LLMAdapter:
    def __init__(self, settings: AckknowSettings) -> None:
        self.settings = settings
        self.own_backend = OwnAckknowLLM(
            OwnModelConfig(
                checkpoint_dir=settings.own_model_path,
                device=settings.own_model_device,
                max_new_tokens=settings.own_model_max_new_tokens,
                temperature=settings.own_model_temperature,
            )
        )
        self.ollama_backend = OllamaClient(settings.ollama_base_url)

    async def generate(self, model: str, prompt: str) -> str:
        if self.settings.llm_backend == "ollama":
            return await self.ollama_backend.generate(model=model, prompt=prompt)
        return await self.own_backend.generate(prompt=prompt)
