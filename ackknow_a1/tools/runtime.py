"""Runtime connectors for sandbox tasks, code tools, and wasm pipelines."""


def c_to_wasm_hint() -> str:
    return "Use emscripten in sandbox terminal: emcc source.c -O3 -s WASM=1 -o output.html"
