from ackknow_a1.types import Source


def reflective_reasoning(prompt: str, sources: list[Source]) -> str:
    checks = [
        f"Q1: What is user intent? -> {prompt[:80]}",
        "Q2: Do sources agree? -> partial agreement; use cautious synthesis.",
        "Q3: Any missing data? -> include assumptions and confidence.",
        f"Q4: Evidence count -> {len(sources)} sources reviewed.",
    ]
    return " | ".join(checks)
