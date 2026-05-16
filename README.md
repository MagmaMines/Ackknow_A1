# Ackknow A1

Ackknow A1 is a self-hostable MIT-licensed AI assistant framework by MagmaMinesTeam for daily use and professional workflows.

## Architecture

Ackknow A1 is organized into required components:

- **Unip**: final answer shaping and response synthesis.
- **IQA**: reflective reasoning, self-questioning, validation loops.
- **Confic**: multimodal pattern and image-signal analysis.
- **Searcher**: retrieval, ranking, fetch orchestration and evidence packaging.
- **DataSets**: dataset catalog and restoration metadata for training/runtime grounding.
- **Guider**: policy, safety, copyright-link guidance output layer.
- **Store**: conversations, user profile, consent flags, state and memory.

It includes:
- long-context tokenizer settings (500k input / 230k output targets)
- personalization hooks
- multilingual routing for English variants, Hindi dialect-space, Hinglish, Tamil
- preloaded domain facts and update hooks
- pluggable tools for sandbox tasks (including code tasks and C-to-WASM workflows)

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
ackknow chat --user "Explain Apple vs Samsung strategy in India"
```

## Example

```bash
ackknow ask "Create a copyright-safe answer with sources for modern AI model selection"
```

## Design goals

- Production-oriented modularity instead of a toy prototype
- Clear component contracts
- Source-aware answer generation
- Research + assistant workflows under one runtime
- Easy local extension for teams
