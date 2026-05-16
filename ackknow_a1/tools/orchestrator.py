from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass
class DoubleCheckResult:
    should_resume: bool
    reason: str


def sandbox_python(code: str) -> str:
    p = subprocess.run(["python", "-c", code], capture_output=True, text=True, timeout=15, check=False)
    return p.stdout if p.returncode == 0 else p.stderr


def intent_double_check(user_prompt: str, draft: str, history: list[str]) -> DoubleCheckResult:
    """Pause-and-verify gate to reduce hallucination and intent mismatch."""
    need = ["must", "required", "exact", "all", "full"]
    strict = any(k in user_prompt.lower() for k in need)
    missing = strict and len(draft.strip()) < 80
    history_conflict = any(user_prompt.lower() in h.lower() and "not" in h.lower() for h in history[-5:]) if history else False
    if missing:
        return DoubleCheckResult(False, "Draft too short for strict request; patch required")
    if history_conflict:
        return DoubleCheckResult(False, "Conflict with prior chat-history intent")
    return DoubleCheckResult(True, "Pattern matches user goal")


def c_to_wasm_hint() -> str:
    return "emcc input.c -O3 -s WASM=1 -o app.html"
