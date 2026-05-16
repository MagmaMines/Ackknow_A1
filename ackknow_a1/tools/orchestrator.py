import subprocess


def sandbox_python(code: str) -> str:
    proc = subprocess.run(["python", "-c", code], capture_output=True, text=True, timeout=10, check=False)
    return proc.stdout if proc.returncode == 0 else proc.stderr


def c_to_wasm_hint() -> str:
    return "emcc input.c -O3 -s WASM=1 -o app.html"
