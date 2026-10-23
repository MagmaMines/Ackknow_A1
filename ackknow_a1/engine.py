from __future__ import annotations

from ackknow_a1.config.settings import AckknowSettings
from ackknow_a1.llm.factory import LLMAdapter
from ackknow_a1.llm.router import ModelRouter
from ackknow_a1.memory.store import MemoryStore
from ackknow_a1.retrieval.searcher import HybridSearcher
from ackknow_a1.safety.guider import apply_safety
from ackknow_a1.core.unip import synthesize
from ackknow_a1.tools.orchestrator import intent_double_check
from ackknow_a1.types import AckRequest, AckResponse, ChatMessage, RouteKind
from ackknow_a1.workflows.coding import CodingAgent
from ackknow_a1.workflows.iqa import reflective_reasoning

PREDATA = {
    "who are you": "I am Ackknow A1, your self-hosted AI assistant.",
    "who own you": "Ackknow A1 is owned and developed by MagmaMinesTeam.",
    "hello": "Hello! I am ready to help with research, coding, and analysis.",
    "whatsup": "I am active and ready. Tell me your objective and constraints.",
}


class AckknowA1:
    def __init__(self, settings: AckknowSettings | None = None) -> None:
        self.settings = settings or AckknowSettings()
        self.router = ModelRouter(self.settings)
        self.llm = LLMAdapter(self.settings)
        self.searcher = HybridSearcher(self.settings.web_search_enabled, self.settings.max_web_results)
        self.memory = MemoryStore(self.settings.sqlite_url)
        self.coder = CodingAgent(
            verify=self.settings.code_verify_enabled,
            execute=self.settings.code_exec_enabled,
            max_repairs=self.settings.code_max_repair_rounds,
            timeout=self.settings.code_exec_timeout_s,
        )

    async def run(self, req: AckRequest) -> AckResponse:
        pl = req.prompt.lower().strip()
        if pl in PREDATA:
            return AckResponse(answer=PREDATA[pl], confidence=0.99, reasoning_summary="predata pattern", sources=[], safety_notes=["predata_matched"], model_used="own:predata", route=RouteKind.FAST)

        history = self.memory.history(req.user_id)
        route, route_model = self.router.route(req.prompt, has_image=bool(req.image_b64))
        sources = self.searcher.search(req.prompt)
        trace = reflective_reasoning(req.prompt, sources, route)

        context = "\n".join([f"{m.role}: {m.content}" for m in history[-12:]])
        prefer_own = self.settings.llm_backend == "own"
        backend = "own"
        code_note: str | None = None
        code_verified: bool | None = None
        if route == RouteKind.CODE:
            async def generate_code(system: str, user: str) -> str:
                nonlocal backend
                text, backend = await self.llm.generate(
                    route_model=route_model,
                    prompt=user,
                    prefer_own=prefer_own,
                    system=system,
                    temperature=self.settings.code_temperature,
                )
                return text

            result = await self.coder.solve(req.prompt, generate_code, context=context)
            draft, code_note = result.answer, result.summary
            code_verified = result.verified if result.checks else None
        else:
            prompt = (
                f"System: You are Ackknow A1. Respond in {req.locale}.\n"
                f"Route: {route.value}\nContext:\n{context}\n\n"
                f"Reasoning Plan:\n{' | '.join(trace.checks)}\n"
                f"Risk Flags: {', '.join(trace.risk_flags) if trace.risk_flags else 'none'}\n"
                f"User: {req.prompt}\n"
                "Return robust solution with assumptions, checks, and implementation detail."
            )
            draft, backend = await self.llm.generate(route_model=route_model, prompt=prompt, prefer_own=prefer_own)
        check = intent_double_check(req.prompt, draft, [m.content for m in history])
        if not check.should_resume:
            draft = f"[Patched after double-check: {check.reason}]\n{draft}\n\nAdded: explicit intent reconciliation."

        answer = synthesize(req.prompt, draft, sources, locale=req.locale)
        safe_answer, notes, confidence = apply_safety(answer, req.prompt)
        if code_verified is not None and "blocked_harmful" not in notes:
            notes.append("code_verified" if code_verified else "code_verification_failed")
            confidence = max(0.05, min(0.99, confidence + (0.1 if code_verified else -0.2)))

        self.memory.append(req.user_id, ChatMessage(role="user", content=req.prompt))
        self.memory.append(req.user_id, ChatMessage(role="assistant", content=safe_answer))

        return AckResponse(
            answer=safe_answer,
            confidence=confidence,
            reasoning_summary=" | ".join(trace.checks + [check.reason] + ([code_note] if code_note else [])),
            sources=sources,
            safety_notes=notes,
            model_used=f"{backend}:{route_model if backend == 'ollama' else self.settings.own_model_checkpoint}",
            route=route,
        )
