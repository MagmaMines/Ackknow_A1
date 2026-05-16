import pytest
from ackknow_a1.engine import AckknowA1
from ackknow_a1.types import AckRequest, Source


@pytest.mark.asyncio
async def test_engine_roundtrip_without_external_llm(monkeypatch):
    model = AckknowA1()

    async def fake_generate(model: str, prompt: str):
        return "Structured answer with assumptions and actions."

    monkeypatch.setattr(model.llm, "generate", fake_generate)
    monkeypatch.setattr(model.searcher, "search", lambda q: [Source(title="S1", snippet="hello", score=0.5)])

    resp = await model.run(AckRequest(user_id="u1", prompt="Plan enterprise AI architecture", locale="en-IN"))
    assert "Answer:" in resp.answer
    assert resp.model_used
    assert resp.confidence > 0
    assert resp.route.value in {"fast", "reason", "code", "vision"}
