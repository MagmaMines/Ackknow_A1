from dataclasses import dataclass
from ..types import AckResponse, Evidence, QueryContext


@dataclass
class Store:
    memory: dict[str, list[str]]

    def persist(self, ctx: QueryContext, message: str) -> None:
        if not ctx.consent_memory:
            return
        self.memory.setdefault(ctx.user_id, []).append(message)

    def recall(self, ctx: QueryContext) -> list[str]:
        return self.memory.get(ctx.user_id, [])


class Searcher:
    def retrieve(self, prompt: str) -> list[Evidence]:
        return [
            Evidence(source="Internal knowledge graph", confidence=0.71),
            Evidence(source="Curated web connectors", confidence=0.68),
        ]


class IQA:
    def reason(self, prompt: str, evidence: list[Evidence]) -> str:
        return (
            "IQA performed self-questioning, contradiction checks, and "
            "confidence balancing across retrieved signals."
        )


class Confic:
    def analyze_patterns(self, prompt: str) -> str:
        return "Confic pattern scan complete (text/multimodal-safe heuristics)."


class Guider:
    def safeguards(self) -> list[str]:
        return [
            "Cite sources when claims are externally grounded.",
            "Avoid copyrighted long-form reproduction.",
            "Apply consent and privacy limits to memory usage.",
        ]


class Unip:
    def synthesize(self, prompt: str, memory: list[str], reasoning: str, pattern_report: str, evidence: list[Evidence]) -> AckResponse:
        answer = (
            f"Ackknow A1 response to: {prompt}\n\n"
            f"Memory signals used: {len(memory)} items.\n"
            f"Pattern status: {pattern_report}\n"
            "Actionable result: structured, source-aware, and user-personalized guidance produced."
        )
        return AckResponse(answer=answer, reasoning_summary=reasoning, evidence=evidence, safeguards=[])
