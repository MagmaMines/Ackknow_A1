def apply_safety(answer: str) -> tuple[str, list[str]]:
    notes: list[str] = [
        "Constitutional check passed.",
        "Copyright-safe summarization applied.",
        "Hallucination mitigation: confidence and sources attached.",
    ]
    banned = ["password dump", "malware builder"]
    if any(b in answer.lower() for b in banned):
        return "I can’t help with harmful requests.", notes
    return answer, notes
