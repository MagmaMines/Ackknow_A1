# Ackknow A1 v3 — Own Neural LLM + Full Framework

Ackknow A1 v3 is a self-hostable assistant framework with an **Ackknow-native neural language model** (no `transformers` runtime dependency), integrated reasoning, retrieval, safety, memory, tools, API, and training pipeline.

## Updated Project Structure

```text
ackknow_a1/
  api/app.py
  core/unip.py
  llm/
    own_model.py           # Ackknow-native byte tokenizer + GRU neural LM
    factory.py             # LLM backend abstraction
    providers.py           # Optional Ollama fallback
    router.py              # fast/reasoning/vision route policy
    training/sft.py        # Own model trainer (JSONL -> checkpoint)
  retrieval/searcher.py    # BM25 + web fallback
  memory/store.py          # SQLite memory store
  workflows/iqa.py         # reflective reasoning workflow
  safety/guider.py         # safety policy filter
  tools/orchestrator.py    # sandbox exec helpers
  engine.py                # full orchestration pipeline
  cli.py
  config/settings.py
  types.py
Dockerfile
docker-compose.yml
```

## Ackknow Native LLM

- `OwnAckknowLLM` is built on:
  - Byte tokenizer (`ByteTokenizer`) for universal multilingual coverage.
  - Deep recurrent neural net (`ReasoningRNN`) for next-token prediction.
  - Temperature + top-k sampling for controllable generation.
- Model weights are loaded from local checkpoint path (`./models/ackknow_a1_rnn/model.pt`).
- No hosted API requirement.

## Framework Capabilities

- **Reasoning/IQA**: reflective checklist loop over intent, source consistency, and confidence.
- **Searcher**: hybrid local BM25 + optional web fallback.
- **Store**: persistent SQLite chat memory.
- **Unip**: final synthesis with source listing.
- **Guider**: policy checks + hallucination caution notes.
- **Tools**: sandbox Python execution + C→WASM helper.
- **API**: FastAPI `POST /v1/chat`, `GET /health`.

## Train Your Own Ackknow Model

```bash
ackknow-train-sft --data ./data/sft.jsonl --out ./models/ackknow_a1_rnn --epochs 1 --device cpu
```

JSONL format:

```json
{"prompt":"How do I deploy fastapi?", "response":"Use uvicorn ..."}
{"text":"User: ...\nAssistant: ..."}
```

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn ackknow_a1.api.app:app --host 0.0.0.0 --port 8000
```

Test:

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H 'content-type: application/json' \
  -d '{"user_id":"u1","prompt":"Design robust AI strategy for Indian multilingual users"}'
```

## Docker

```bash
docker compose up --build
```
