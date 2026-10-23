from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from ackknow_a1.tools.orchestrator import run_python_sandboxed

GenerateFn = Callable[[str, str], Awaitable[str]]

CODE_SYSTEM_PROMPT = """You are Ackknow A1, an expert senior software engineer.
When asked to write or fix code:
1. Restate the goal in one sentence and list key assumptions (inputs, outputs, edge cases).
2. Briefly outline the approach before the code.
3. Write complete, runnable code in fenced blocks tagged with the language (```python, ```javascript, ...).
   No placeholders, no "TODO", no "...", no omitted sections. Include all imports.
4. Prefer the standard library, clear names, type hints, docstrings, input validation and error handling.
5. Handle edge cases (empty input, None, large values, invalid types) explicitly.
6. For Python, add a small self-test under `if __name__ == "__main__":` using `assert` statements
   that exercises normal and edge cases. Do not read from stdin, the network, or the filesystem in the self-test.
7. When fixing a bug, explain the root cause first, then show the full corrected code.
8. End with a short note on complexity and how to run it.
Never invent APIs or library functions. If something is unknown, say so."""

LANG_ALIASES = {
    "py": "python", "python3": "python", "js": "javascript", "node": "javascript", "mjs": "javascript",
    "ts": "typescript", "sh": "bash", "shell": "bash", "zsh": "bash", "console": "bash",
}
LANG_HINTS = [
    ("python", r"\b(python|py|django|flask|fastapi|pandas|numpy|pytest)\b"),
    ("javascript", r"\b(javascript|js|node(js)?|react|express)\b"),
    ("typescript", r"\b(typescript|ts)\b"),
    ("bash", r"\b(bash|shell script|sh)\b"),
    ("sql", r"\bsql\b"),
    ("java", r"\bjava\b"),
    ("cpp", r"\bc\+\+|cpp\b"),
    ("rust", r"\brust\b"),
    ("go", r"\b(golang|go lang)\b"),
]
PLACEHOLDER_PATTERN = re.compile(r"(#|//)\s*(TODO|FIXME|your code here|implement (this|me))|^\s*\.\.\.\s*$", re.IGNORECASE | re.MULTILINE)
FENCE_PATTERN = re.compile(r"```[ \t]*([\w+#.-]*)[^\n]*\n(.*?)```", re.DOTALL)


@dataclass
class CodeBlock:
    lang: str
    code: str


@dataclass
class CheckResult:
    ok: bool
    lang: str
    stage: str
    detail: str = ""


@dataclass
class CodingResult:
    answer: str
    verified: bool
    attempts: int
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if not self.checks:
            return f"code: no runnable blocks found (attempts={self.attempts})"
        status = "verified" if self.verified else "unverified"
        stages = ", ".join(f"{c.lang}:{c.stage}:{'ok' if c.ok else 'fail'}" for c in self.checks)
        return f"code {status} after {self.attempts} attempt(s) [{stages}]"


def detect_language(prompt: str) -> str | None:
    p = prompt.lower()
    for lang, pattern in LANG_HINTS:
        if re.search(pattern, p):
            return lang
    return None


def normalize_lang(lang: str) -> str:
    lang = lang.strip().lower()
    return LANG_ALIASES.get(lang, lang)


def extract_code_blocks(text: str, default_lang: str | None = None) -> list[CodeBlock]:
    blocks = []
    for lang, code in FENCE_PATTERN.findall(text):
        lang = normalize_lang(lang) or (default_lang or "")
        if code.strip():
            blocks.append(CodeBlock(lang=lang, code=code.rstrip() + "\n"))
    return blocks


def _run_checker(cmd: list[str], code: str, suffix: str, timeout: float) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="ackknow_chk_") as tmp:
        path = Path(tmp) / f"snippet{suffix}"
        path.write_text(code, encoding="utf-8")
        try:
            p = subprocess.run([*cmd, str(path)], capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return False, "checker timed out"
        return p.returncode == 0, (p.stdout + p.stderr).replace(str(path), "snippet")[-2000:]


def verify_block(block: CodeBlock, execute: bool = False, timeout: float = 10) -> CheckResult:
    lang, code = block.lang, block.code
    placeholder = PLACEHOLDER_PATTERN.search(code)
    if placeholder:
        return CheckResult(False, lang, "completeness", f"Placeholder found: {placeholder.group(0).strip()!r}")

    if lang == "python":
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return CheckResult(False, lang, "syntax", f"SyntaxError line {exc.lineno}: {exc.msg}\n{(exc.text or '').rstrip()}")
        if not execute:
            return CheckResult(True, lang, "syntax")
        res = run_python_sandboxed(code, timeout=timeout)
        return CheckResult(res.ok, lang, "execution", res.output)
    if lang == "json":
        try:
            json.loads(code)
        except json.JSONDecodeError as exc:
            return CheckResult(False, lang, "syntax", str(exc))
        return CheckResult(True, lang, "syntax")
    if lang == "javascript" and shutil.which("node"):
        ok, out = _run_checker(["node", "--check"], code, ".js", timeout)
        return CheckResult(ok, lang, "syntax", out)
    if lang == "bash" and shutil.which("bash"):
        ok, out = _run_checker(["bash", "-n"], code, ".sh", timeout)
        return CheckResult(ok, lang, "syntax", out)
    return CheckResult(True, lang or "text", "skipped", "no checker available")


def build_code_prompt(user_prompt: str, context: str = "", language: str | None = None) -> str:
    parts = []
    if context.strip():
        parts.append(f"Conversation so far:\n{context.strip()}\n")
    if language:
        parts.append(f"Target language: {language}")
    parts.append(f"Task:\n{user_prompt.strip()}")
    return "\n".join(parts)


REPAIR_HEADER = "Automatic verification found these problems"


def build_repair_prompt(user_prompt: str, previous: str, failures: list[CheckResult], language: str | None = None) -> str:
    code = "\n\n".join(f"```{b.lang}\n{b.code}```" for b in extract_code_blocks(previous, language)) or previous.strip()
    issues = "\n\n".join(f"[{f.lang} / {f.stage}]\n{f.detail.strip()}" for f in failures)
    return (
        "Fix the code below so it passes verification. Explain the root cause in one or two sentences, "
        "then give the complete corrected code (full code, not a diff). "
        "Do not repeat these instructions or the error log.\n\n"
        f"Original task:\n{user_prompt.strip()}\n\n"
        f"Code that failed:\n{code}\n\n"
        f"{REPAIR_HEADER}:\n{issues}"
    )


def strip_echo(answer: str) -> str:
    """Drop prompt text that small models sometimes parrot back after their answer."""
    idx = answer.find(REPAIR_HEADER)
    return answer[:idx].rstrip() if idx > 0 else answer


class CodingAgent:
    """Generate -> verify -> repair loop for code requests."""

    def __init__(self, verify: bool = True, execute: bool = False, max_repairs: int = 2, timeout: float = 10) -> None:
        self.verify = verify
        self.execute = execute
        self.max_repairs = max(0, max_repairs)
        self.timeout = timeout

    def check(self, answer: str, language: str | None) -> list[CheckResult]:
        if not self.verify:
            return []
        return [verify_block(b, self.execute, self.timeout) for b in extract_code_blocks(answer, language)]

    async def solve(self, user_prompt: str, generate: GenerateFn, context: str = "") -> CodingResult:
        language = detect_language(user_prompt)
        answer = strip_echo(await generate(CODE_SYSTEM_PROMPT, build_code_prompt(user_prompt, context, language)))
        checks = self.check(answer, language)
        best = (answer, checks)
        attempts = 1
        while attempts <= self.max_repairs:
            failures = [c for c in checks if not c.ok]
            if not failures:
                break
            prompt = build_repair_prompt(user_prompt, answer, failures, language)
            candidate = strip_echo(await generate(CODE_SYSTEM_PROMPT, prompt))
            attempts += 1
            candidate_checks = self.check(candidate, language)
            if not candidate_checks and checks:
                continue
            answer, checks = candidate, candidate_checks
            if _failures(checks) <= _failures(best[1]):
                best = (answer, checks)
        answer, checks = best
        verified = bool(checks) and all(c.ok for c in checks)
        return CodingResult(answer=answer, verified=verified, attempts=attempts, checks=checks)


def _failures(checks: list[CheckResult]) -> int:
    return sum(not c.ok for c in checks)
