# Ackknow A1 v2

Production-grade, self-hostable AI assistant with modular architecture (Unip, IQA, Confic, Searcher, Guider, Store), Ollama-native inference, hybrid retrieval, persistent memory, and FastAPI serving.

## Updated Project Structure

```text
ackknow_a1/
  api/app.py
  core/unip.py
  llm/providers.py
  llm/router.py
  retrieval/searcher.py
  memory/store.py
  workflows/iqa.py
  safety/guider.py
  tools/orchestrator.py
  engine.py
  cli.py
  config/settings.py
  types.py
Dockerfile
docker-compose.yml
```

## Key Features

- **LLM integration layer**: Ollama provider + smart routing (fast/reasoning/vision).
- **Searcher (real hybrid)**: BM25 + web fallback via DuckDuckGo (extensible to Tavily/Serper/Brave).
- **IQA reasoning loop**: reflective multi-check chain with evidence checks.
- **Confic-ready**: model router supports vision model path and image-aware routing.
- **Store**: SQLite-backed persistent memory.
- **Unip**: final synthesis with citations and confidence output.
- **Guider**: constitutional-style safety checks and anti-hallucination notes.
- **Tooling**: sandbox Python execution and C→WASM helper.
- **Production API**: FastAPI `/v1/chat` + health endpoint.
- **Self-hostable**: Docker + docker-compose.

## Run with Ollama (Local)

1. Install and start Ollama.
2. Pull models:
   - `ollama pull qwen2.5:7b-instruct`
   - `ollama pull qwen2.5:14b-instruct`
   - `ollama pull llava:13b`
3. Install app:
   - `python -m venv .venv && source .venv/bin/activate`
   - `pip install -e .`
4. Start API:
   - `uvicorn ackknow_a1.api.app:app --reload --port 8000`
5. Test:
   - `curl -X POST http://localhost:8000/v1/chat -H 'content-type: application/json' -d '{"user_id":"u1","prompt":"Design a multilingual AI support strategy for India"}'`

## Docker

- `docker compose up --build`

## Notes

- Default is local-first and self-hosted.
- You can swap in Chroma/Qdrant embeddings and rerankers in `retrieval/searcher.py`.
