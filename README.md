# Ackknow A1 v5 — Intelligent Hybrid Routing Framework and Native LLM 

---

Ackknow A1 is a **10K** context window Large Language Model. A Experimental Project by **MagmaMinesTeam**.

**Usecase:**
- Learning.
- Small tasks.
- Daily usecases.
- Mobile users.
  
  ---
  **Note: this is a experimental open-source project. Bugs & issue and Ai misinformation is **normal** for this type of project's.**
  
## Information ℹ️ 

- **Intelligent hybrid routing**:
  - Default smartest path is `Own Ackknow A1`.
  - Optional Ollama fallback remains available (`ACKKNOW_LLM_BACKEND=ollama`).
  - Engine now returns `model_used` with backend and checkpoint/model.

- **Model versioning & checkpoints**:
  - `own_model_checkpoint` setting selects checkpoint.
  - Training writes `checkpoints.json` manifest with historical runs.
  - Runtime reads manifest for version-aware profile metadata.

- **KV-cache style state management**:
  - Added transformer KV cache in `MiniTransformer` and threaded through generation loop.
  - RNN + memory + KV state reused per token for faster incremental decode.

- **ReasoningRNN intelligence jump**:
  - Added **MiniTransformer block** (8 heads) after recurrent fusion.
  - Hybrid architecture now: GRU + LSTM + gated memory + attention + verification head.

- **Training quality strategy**:
  - Curriculum phases for chat → coding → math → tools → Indian multilingual.
  - Synthetic generation expanded and identity/pattern priors added.
  - Dataset mix metadata includes coding/math/Indian-language corpora.

- **Termux / mobile-first support**:
  - `termux_mode` disables heavy compile/quant defaults and limits generation length.
  - `mobile_profile` config available for future per-device tuning.

## Coding agent (v5.1) 🧑‍💻

Code requests (detected by language names, code fences, tracebacks, `implement`, `refactor`, …) go through a
**generate → verify → repair** loop (`ackknow_a1/workflows/coding.py`):

1. A senior-engineer system prompt asks for complete, runnable, fenced code with edge cases and self-tests.
2. Every code block is checked: placeholder/TODO detection, Python `ast` syntax check, `json`, `node --check`, `bash -n`.
3. Optionally (`ACKKNOW_CODE_EXEC_ENABLED=true`) Python blocks are executed in an isolated subprocess
   (`python -I`, temp dir, minimal env, timeout) so failing `assert` self-tests are caught.
4. Failures are fed back to the model for up to `ACKKNOW_CODE_MAX_REPAIR_ROUNDS` repair attempts.
5. The response reports `code_verified` / `code_verification_failed` and adjusts confidence.

Code routes use `ACKKNOW_OLLAMA_CODE_MODEL` (default `qwen2.5-coder:7b-instruct`) at `ACKKNOW_CODE_TEMPERATURE=0.2`.
If no trained own checkpoint exists, Ackknow automatically falls back to Ollama (and back to the own model if
Ollama is unreachable); disable with `ACKKNOW_LLM_FALLBACK_ENABLED=false`.

```bash
ollama pull qwen2.5-coder:7b-instruct
ACKKNOW_CODE_EXEC_ENABLED=true ackknow "Write a Python function that merges overlapping intervals"
```

## Realism note

A experimental  219 Million parameters model with 4,000 – 8,000 output tokens,  requires distributed training/inference infra. This repo now provides robust architecture, routing, checkpointing, and optimization scaffolding for that target while remaining locally runnable for development.

## Misinformation and presentation:

Ai can't say the truth.
it's just a pattern.
and patterns can break or wrong.

