import pytest
from ackknow_a1.engine import AckknowA1
from ackknow_a1.types import AckRequest


@pytest.mark.asyncio
async def test_engine_roundtrip_without_external_llm(monkeypatch):
    model = AckknowA1()

    async def fake_generate(model: str, prompt: str, temperature: float = 0.2):
        return "Here is a robust answer with steps and rationale."

    monkeypatch.setattr(model.llm, "generate", fake_generate)
    monkeypatch.setattr(model.searcher, "search", lambda q: [])

    resp = await model.run(AckRequest(user_id="u1", prompt="Plan enterprise AI architecture"))
    assert "Answer:" in resp.answer
    assert resp.model_used
    assert resp.confidence > 0
