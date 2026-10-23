from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass


@dataclass
class DoubleCheckResult:
    should_resume: bool
    reason: str


@dataclass
class SandboxResult:
    ok: bool
    output: str
    timed_out: bool = False


def run_python_sandboxed(code: str, timeout: float = 15) -> SandboxResult:
    """Run code in an isolated interpreter (-I), throwaway cwd, minimal env, no stdin."""
    with tempfile.TemporaryDirectory(prefix="ackknow_sbx_") as tmp:
        script = os.path.join(tmp, "main.py")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(code)
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8", "HOME": tmp}
        try:
            p = subprocess.run(
                [sys.executable, "-I", script],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp,
                env=env,
                stdin=subprocess.DEVNULL,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(False, f"Execution timed out after {timeout}s", timed_out=True)
    output = (p.stdout + p.stderr).replace(tmp, "<sandbox>")
    return SandboxResult(p.returncode == 0, output[-4000:])


def sandbox_python(code: str) -> str:
    return run_python_sandboxed(code).output


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
