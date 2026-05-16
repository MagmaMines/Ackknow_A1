from ackknow_a1.config.settings import AckknowSettings
from ackknow_a1.llm.own_model import OwnAckknowLLM, OwnModelConfig
from ackknow_a1.llm.providers import OllamaClient


class LLMAdapter:
    def __init__(self, settings: AckknowSettings) -> None:
        self.settings = settings
        self.own_backend = OwnAckknowLLM(
            OwnModelConfig(
                checkpoint_dir=settings.own_model_path,
                checkpoint_name=settings.own_model_checkpoint,
                device=settings.own_model_device,
                max_input_tokens=settings.own_model_max_input_tokens,
                max_new_tokens=settings.own_model_max_new_tokens,
                temperature=settings.own_model_temperature,
                use_compile=settings.own_model_compile,
                quantization=settings.own_model_quantization,
                termux_mode=settings.termux_mode,
            )
        )
        self.ollama_backend = OllamaClient(settings.ollama_base_url)

    async def generate(self, route_model: str, prompt: str, prefer_own: bool = True) -> tuple[str, str]:
        if prefer_own or self.settings.llm_backend == "own":
            return await self.own_backend.generate(prompt=prompt), "own"
        return await self.ollama_backend.generate(model=route_model, prompt=prompt), "ollama"
