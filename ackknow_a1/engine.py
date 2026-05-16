from ackknow_a1.config.settings import AckknowSettings
from ackknow_a1.llm.providers import OllamaClient
from ackknow_a1.llm.router import ModelRouter
from ackknow_a1.memory.store import MemoryStore
from ackknow_a1.retrieval.searcher import HybridSearcher
from ackknow_a1.safety.guider import apply_safety
from ackknow_a1.core.unip import synthesize
from ackknow_a1.types import AckRequest, AckResponse, ChatMessage
from ackknow_a1.workflows.iqa import reflective_reasoning


class AckknowA1:
    def __init__(self, settings: AckknowSettings | None = None) -> None:
        self.settings = settings or AckknowSettings()
        self.router = ModelRouter(self.settings)
        self.llm = OllamaClient(self.settings.ollama_base_url)
        self.searcher = HybridSearcher(self.settings.web_search_enabled, self.settings.max_web_results)
        self.memory = MemoryStore(self.settings.sqlite_url)

    async def run(self, req: AckRequest) -> AckResponse:
        history = self.memory.history(req.user_id)
        sources = self.searcher.search(req.prompt)
        reasoning = reflective_reasoning(req.prompt, sources)
        model = self.router.route(req.prompt, has_image=bool(req.image_b64))
        context = "\n".join([f"{m.role}: {m.content}" for m in history[-6:]])
        prompt = f"Context:\n{context}\n\nReasoning plan:\n{reasoning}\n\nUser: {req.prompt}\nUse citations from provided sources."
        draft = await self.llm.generate(model=model, prompt=prompt)
        answer = synthesize(req.prompt, draft, sources)
        safe_answer, notes = apply_safety(answer)
        self.memory.append(req.user_id, ChatMessage(role="user", content=req.prompt))
        self.memory.append(req.user_id, ChatMessage(role="assistant", content=safe_answer))
        return AckResponse(answer=safe_answer, confidence=0.72, reasoning_summary=reasoning, sources=sources[:8], safety_notes=notes, model_used=model)
