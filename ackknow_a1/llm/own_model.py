from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class OwnModelConfig:
    model_id: str
    device: str = "cpu"
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95


class OwnAckknowLLM:
    """Local-only production runtime for Ackknow-owned LLM weights.

    Uses open-source checkpoints and/or fine-tuned local artifacts with no SaaS API dependency.
    """

    def __init__(self, cfg: OwnModelConfig) -> None:
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype="auto")
        self.model.to(cfg.device)
        self.model.eval()

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        max_new_tokens = int(kwargs.get("max_new_tokens", self.cfg.max_new_tokens))
        temperature = float(kwargs.get("temperature", self.cfg.temperature))
        top_p = float(kwargs.get("top_p", self.cfg.top_p))

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.cfg.device)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return text[len(prompt):].strip() if text.startswith(prompt) else text.strip()
