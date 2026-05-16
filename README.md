# Ackknow A1 v2.1 — Own LLM Production Grade

Ackknow A1 now supports a **true own-LLM backend** (local weights, local inference, local fine-tuning) with no dependency on proprietary hosted APIs.

## Updated Project Structure

```text
ackknow_a1/
  api/app.py
  core/unip.py
  llm/
    own_model.py
    factory.py
    providers.py
    router.py
    training/sft.py
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

## Own LLM (Production Path)

- `llm_backend=own` by default.
- `OwnAckknowLLM` runs local Hugging Face-compatible checkpoints.
- `ackknow-train-sft` command provides supervised fine-tuning entrypoint.
- Ollama backend remains optional fallback for operations.

## Run Locally (Own LLM)

1. Create/activate venv and install:
   - `python -m venv .venv && source .venv/bin/activate`
   - `pip install -e .`
2. Prepare model weights locally (example path):
   - `./models/ackknow-a1`
3. Set env (optional):
   - `export ACKKNOW_LLM_BACKEND=own`
   - `export ACKKNOW_OWN_MODEL_PATH=./models/ackknow-a1`
4. Run API:
   - `uvicorn ackknow_a1.api.app:app --host 0.0.0.0 --port 8000`
5. Call:
   - `curl -X POST http://localhost:8000/v1/chat -H 'content-type: application/json' -d '{"user_id":"u1","prompt":"Design Indian multilingual AI support strategy"}'`

## Fine-tune your own Ackknow model

```bash
ackknow-train-sft --base mistralai/Mistral-7B-Instruct-v0.3 --data ./data/sft.jsonl --out ./models/ackknow-a1 --epochs 1
```

Expected JSONL examples:

```json
{"prompt":"...", "response":"..."}
{"text":"User: ...\nAssistant: ..."}
```

## Docker

- `docker compose up --build`

## Notes

- All critical inference logic can run fully local and self-hosted.
- You can scale further with quantization, vLLM/TGI serving, LoRA/QLoRA and distributed training.
