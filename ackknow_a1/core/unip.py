from ackknow_a1.types import Source


def synthesize(prompt: str, model_text: str, sources: list[Source], user_style: str = "professional") -> str:
    citations = "\n".join([f"- {s.title} ({s.url or 'local'})" for s in sources[:5]])
    tone = "clear and concise" if user_style == "professional" else "friendly"
    return (
        f"Tone: {tone}\n"
        f"Prompt: {prompt}\n\n"
        f"Answer:\n{model_text}\n\n"
        f"Sources:\n{citations}"
    )
