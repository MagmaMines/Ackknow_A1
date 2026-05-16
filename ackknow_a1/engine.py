from __future__ import annotations

from ackknow_a1.config.settings import AckknowSettings
from ackknow_a1.llm.factory import LLMAdapter
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
        self.llm = LLMAdapter(self.settings)
        self.searcher = HybridSearcher(self.settings.web_search_enabled, self.settings.max_web_results)
        self.memory = MemoryStore(self.settings.sqlite_url)

    async def run(self, req: AckRequest) -> AckResponse:
        history = self.memory.history(req.user_id)
        route, model = self.router.route(req.prompt, has_image=bool(req.image_b64))
        sources = self.searcher.search(req.prompt)
        trace = reflective_reasoning(req.prompt, sources, route)

        context = "\n".join([f"{m.role}: {m.content}" for m in history[-8:]])
        plan = " | ".join(trace.checks)
        prompt = (
            f"System: You are Ackknow A1. Respond in {req.locale}.\n"
            f"Route: {route.value}\n"
            f"Context:\n{context}\n\n"
            f"Reasoning Plan:\n{plan}\n"
            f"Risk Flags: {', '.join(trace.risk_flags) if trace.risk_flags else 'none'}\n\n"
            f"User: {req.prompt}\n"
            "Return concise but complete answer with explicit assumptions where needed."
        )
        draft = await self.llm.generate(model=model, prompt=prompt)
        answer = synthesize(req.prompt, draft, sources, locale=req.locale)
        safe_answer, notes, confidence = apply_safety(answer, req.prompt)

        self.memory.append(req.user_id, ChatMessage(role="user", content=req.prompt))
        self.memory.append(req.user_id, ChatMessage(role="assistant", content=safe_answer))

        return AckResponse(
            answer=safe_answer,
            confidence=confidence,
            reasoning_summary=" | ".join(trace.checks),
            sources=sources,
            safety_notes=notes,
            model_used=model,
            route=route,
        )
