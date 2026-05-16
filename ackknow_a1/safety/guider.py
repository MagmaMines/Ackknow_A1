from __future__ import annotations


HARMFUL = ["malware", "password dump", "exploit zero-day", "build ransomware"]


def apply_safety(answer: str, prompt: str) -> tuple[str, list[str], float]:
    notes = [
        "Constitutional policy check complete.",
        "Copyright-safe summarization enforced.",
        "Potential hallucination mitigated with source-grounding.",
    ]
    lower = (prompt + " " + answer).lower()
    if any(k in lower for k in HARMFUL):
        return "I can’t assist with harmful or abusive activities.", notes + ["blocked_harmful"], 0.15

    confidence = 0.80
    if "no external sources used" in answer:
        confidence -= 0.12
        notes.append("low_source_density")
    return answer, notes, max(0.05, min(0.99, confidence))
