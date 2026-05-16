from typing import Any
import httpx


class OllamaClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    async def generate(self, model: str, prompt: str, temperature: float = 0.2) -> str:
        async with httpx.AsyncClient(timeout=120) as client:
            payload: dict[str, Any] = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}}
            resp = await client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            return resp.json().get("response", "")
