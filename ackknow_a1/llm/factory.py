import httpx
from loguru import logger

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

    async def _own(self, prompt: str, system: str | None, temperature: float | None) -> tuple[str, str]:
        full_prompt = f"System: {system}\n{prompt}" if system else prompt
        kwargs = {} if temperature is None else {"temperature": temperature}
        return await self.own_backend.generate(prompt=full_prompt, **kwargs), "own"

    async def generate(
        self,
        route_model: str,
        prompt: str,
        prefer_own: bool = True,
        system: str | None = None,
        temperature: float | None = None,
    ) -> tuple[str, str]:
        own_ready = getattr(self.own_backend, "has_checkpoint", True)
        fallback = self.settings.llm_fallback_enabled
        if prefer_own and (own_ready or not fallback):
            return await self._own(prompt, system, temperature)
        if prefer_own:
            logger.info("Own Ackknow checkpoint not found; falling back to Ollama model {}", route_model)
        try:
            text = await self.ollama_backend.generate(
                model=route_model,
                prompt=prompt,
                temperature=0.2 if temperature is None else temperature,
                system=system,
            )
            return text, "ollama"
        except httpx.HTTPError as exc:
            if not fallback:
                raise
            logger.warning("Ollama unavailable ({}); falling back to own model", exc)
            return await self._own(prompt, system, temperature)
