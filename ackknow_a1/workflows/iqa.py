from __future__ import annotations

from ackknow_a1.types import ReasoningTrace, RouteKind, Source


def reflective_reasoning(prompt: str, sources: list[Source], route: RouteKind) -> ReasoningTrace:
    checks = [
        f"Intent parsed for route={route.value}",
        f"Evidence count={len(sources)}",
        "Counterfactual: if top evidence is wrong, hedge claims",
        "Self-critique: avoid overclaiming, mention assumptions",
    ]
    risks: list[str] = []
    if len(sources) == 0:
        risks.append("low_evidence")
    if any(t in prompt.lower() for t in ["medical", "legal", "invest", "diagnose"]):
        risks.append("high_stakes")
    return ReasoningTrace(route=route, checks=checks, risk_flags=risks)
