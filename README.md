# Ackknow A1 v5 — Intelligent Hybrid Routing + Attention-Enhanced Native LLM

## Major upgrades

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

## Realism note

A experimental  219 Million parameters model with 4,000 – 8,000 output tokens,  requires distributed training/inference infra. This repo now provides robust architecture, routing, checkpointing, and optimization scaffolding for that target while remaining locally runnable for development.
