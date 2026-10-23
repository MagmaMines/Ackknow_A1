import httpx
import pytest

from ackknow_a1.config.settings import AckknowSettings
from ackknow_a1.llm.router import ModelRouter, is_code_request
from ackknow_a1.tools.orchestrator import run_python_sandboxed
from ackknow_a1.types import RouteKind
from ackknow_a1.workflows.coding import (
    CodeBlock,
    CodingAgent,
    detect_language,
    extract_code_blocks,
    verify_block,
)


@pytest.mark.parametrize("prompt", [
    "Write a Python function to reverse a list",
    "Fix this bug in my React component",
    "Traceback (most recent call last):\n  File x",
    "how do I implement binary search in rust?",
    "SELECT * FROM users is slow",
])
def test_code_requests_detected(prompt):
    assert is_code_request(prompt)


@pytest.mark.parametrize("prompt", ["What is the capital of India?", "decode the meaning of this poem", "hello there"])
def test_non_code_requests_not_detected(prompt):
    assert not is_code_request(prompt)


def test_router_uses_code_model():
    s = AckknowSettings()
    assert ModelRouter(s).route("refactor my python script") == (RouteKind.CODE, s.ollama_code_model)


def test_extract_and_detect():
    text = "Intro\n```py\nprint(1)\n```\nand\n```js\nconsole.log(1)\n```"
    blocks = extract_code_blocks(text)
    assert [b.lang for b in blocks] == ["python", "javascript"]
    assert detect_language("write a node script") == "javascript"
    assert extract_code_blocks("```\nx = 1\n```", default_lang="python")[0].lang == "python"


def test_verify_block_python():
    assert verify_block(CodeBlock("python", "x = 1\n")).ok
    bad = verify_block(CodeBlock("python", "def f(:\n  pass\n"))
    assert not bad.ok and bad.stage == "syntax"
    todo = verify_block(CodeBlock("python", "def f():\n    # TODO implement\n    pass\n"))
    assert not todo.ok and todo.stage == "completeness"


def test_verify_block_executes_when_enabled():
    ok = verify_block(CodeBlock("python", "assert 1 + 1 == 2\nprint('ok')\n"), execute=True)
    assert ok.ok and ok.stage == "execution"
    fail = verify_block(CodeBlock("python", "assert 1 + 1 == 3\n"), execute=True)
    assert not fail.ok and "AssertionError" in fail.detail


def test_sandbox_timeout():
    res = run_python_sandboxed("while True: pass\n", timeout=1)
    assert not res.ok and res.timed_out


@pytest.mark.asyncio
async def test_agent_repairs_failing_tests():
    replies = iter([
        "```python\ndef sq(x):\n    return x * 2\n\nassert sq(3) == 9\n```",
        "```python\ndef sq(x):\n    return x * x\n\nassert sq(3) == 9\n```",
    ])
    prompts = []

    async def gen(system: str, user: str) -> str:
        prompts.append(user)
        return next(replies)

    result = await CodingAgent(execute=True, max_repairs=2).solve("python square function", gen)
    assert result.verified and result.attempts == 2
    assert "AssertionError" in prompts[1]


@pytest.mark.asyncio
async def test_agent_stops_after_max_repairs():
    async def gen(system: str, user: str) -> str:
        return "```python\ndef broken(:\n```"

    result = await CodingAgent(max_repairs=1).solve("python thing", gen)
    assert not result.verified and result.attempts == 2


@pytest.mark.asyncio
async def test_adapter_falls_back_when_no_checkpoint(monkeypatch, tmp_path):
    from ackknow_a1.llm.factory import LLMAdapter

    adapter = LLMAdapter(AckknowSettings(own_model_path=str(tmp_path), own_model_compile=False, own_model_quantization="none"))
    assert adapter.own_backend.has_checkpoint is False

    async def fake_ollama(model, prompt, temperature=0.2, system=None):
        return f"ollama:{model}:{system}"

    async def fake_own(prompt, **kwargs):
        return "own-text"

    monkeypatch.setattr(adapter.ollama_backend, "generate", fake_ollama)
    monkeypatch.setattr(adapter.own_backend, "generate", fake_own)
    text, backend = await adapter.generate("qwen2.5-coder:7b-instruct", "hi", system="sys")
    assert (text, backend) == ("ollama:qwen2.5-coder:7b-instruct:sys", "ollama")

    async def down(*args, **kwargs):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(adapter.ollama_backend, "generate", down)
    assert await adapter.generate("m", "hi") == ("own-text", "own")


@pytest.mark.asyncio
async def test_agent_strips_echo_and_keeps_best():
    replies = iter([
        "```python\ndef f(:\n```",
        "```python\nx = 1\n```\nAutomatic verification found these problems:\n- junk",
    ])

    async def gen(system: str, user: str) -> str:
        return next(replies)

    result = await CodingAgent(max_repairs=1).solve("python thing", gen)
    assert result.verified
    assert "Automatic verification" not in result.answer
