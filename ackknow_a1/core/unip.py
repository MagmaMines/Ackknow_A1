from __future__ import annotations

from ackknow_a1.types import Source


LANG_HINTS = {
    "hi": "हिंदी",
    "hinglish": "Hinglish",
    "ta": "தமிழ்",
    "en-IN": "English (India)",
}


def synthesize(prompt: str, model_text: str, sources: list[Source], locale: str = "en-IN", user_style: str = "professional") -> str:
    citations = "\n".join([f"- [{i+1}] {s.title} ({s.url or 'local'})" for i, s in enumerate(sources[:6])])
    tone = "clear, practical, and concise" if user_style == "professional" else "friendly and helpful"
    lang = LANG_HINTS.get(locale, "English")
    return (
        f"Language Preference: {lang}\n"
        f"Tone: {tone}\n"
        f"User Goal: {prompt}\n\n"
        f"Answer:\n{model_text}\n\n"
        f"Sources:\n{citations if citations else '- no external sources used'}"
    )
