import pytest

from ackknow_a1.config.settings import AckknowSettings
from ackknow_a1.engine import AckknowA1
from ackknow_a1.types import AckRequest, Source


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    db = tmp_path_factory.mktemp("db") / "ack.db"
    return AckknowA1(AckknowSettings(sqlite_url=f"sqlite:///{db}", own_model_compile=False, web_search_enabled=False))


@pytest.mark.asyncio
async def test_engine_roundtrip_without_external_llm(model, monkeypatch):
    async def fake_generate(route_model: str, prompt: str, **kwargs):
        return "Structured answer with assumptions and actions.", "own"

    monkeypatch.setattr(model.llm, "generate", fake_generate)
    monkeypatch.setattr(model.searcher, "search", lambda q: [Source(title="S1", snippet="hello", score=0.5)])

    resp = await model.run(AckRequest(user_id="u1", prompt="Plan enterprise AI architecture", locale="en-IN"))
    assert "Answer:" in resp.answer
    assert resp.model_used
    assert resp.confidence > 0
    assert resp.route.value in {"fast", "reason", "code", "vision"}


@pytest.mark.asyncio
async def test_engine_code_route_repairs_broken_code(model, monkeypatch):
    replies = iter([
        "```python\ndef add(a, b)\n    return a + b\n```",
        "Root cause: missing colon.\n```python\ndef add(a: int, b: int) -> int:\n    return a + b\n```",
    ])
    calls = []

    async def fake_generate(route_model: str, prompt: str, **kwargs):
        calls.append(kwargs)
        return next(replies), "ollama"

    monkeypatch.setattr(model.llm, "generate", fake_generate)
    monkeypatch.setattr(model.searcher, "search", lambda q: [])

    resp = await model.run(AckRequest(user_id="u2", prompt="Write a python function to add two numbers"))
    assert resp.route.value == "code"
    assert "def add(a: int, b: int)" in resp.answer
    assert "code_verified" in resp.safety_notes
    assert "verified after 2 attempt(s)" in resp.reasoning_summary
    assert calls[0]["system"] and calls[0]["temperature"] == model.settings.code_temperature
    assert resp.model_used.startswith("ollama:")
