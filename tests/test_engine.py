from ackknow_a1.engine import AckknowA1
from ackknow_a1.types import QueryContext


def test_engine_roundtrip():
    model = AckknowA1()
    ctx = QueryContext(user_id="u1")
    resp = model.run("Plan my coding workflow", ctx)
    assert "Ackknow A1 response" in resp.answer
    assert len(resp.safeguards) >= 1
    assert len(model.datasets) >= 100
