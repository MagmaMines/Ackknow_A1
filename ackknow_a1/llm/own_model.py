from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import torch
from torch import nn


class ByteTokenizer:
    vocab_size = 258
    bos_id = 256
    eos_id = 257

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        ids = [min(255, b) for b in text.encode("utf-8", errors="ignore")]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        data = bytes([i for i in ids if 0 <= i <= 255])
        return data.decode("utf-8", errors="ignore")


class ReasoningRNN(nn.Module):
    def __init__(self, vocab_size: int = 258, emb: int = 384, hidden: int = 768, layers: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb)
        self.rnn = nn.GRU(emb, hidden, num_layers=layers, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, idx: torch.Tensor, hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embed(idx)
        y, h = self.rnn(x, hidden)
        y = self.norm(y)
        logits = self.head(y)
        return logits, h


@dataclass
class OwnModelConfig:
    checkpoint_dir: str = "./models/ackknow_a1_rnn"
    device: str = "cpu"
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_k: int = 40


class OwnAckknowLLM:
    """Ackknow-native neural LLM runtime without transformers dependency."""

    def __init__(self, cfg: OwnModelConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.tok = ByteTokenizer()
        self.model = ReasoningRNN(vocab_size=self.tok.vocab_size).to(self.device)
        self.model.eval()
        self._load_checkpoint(cfg.checkpoint_dir)

    def _load_checkpoint(self, checkpoint_dir: str) -> None:
        path = Path(checkpoint_dir)
        weight_file = path / "model.pt"
        if weight_file.exists():
            state = torch.load(weight_file, map_location=self.device)
            self.model.load_state_dict(state)

    async def generate(self, prompt: str, **kwargs: float | int) -> str:
        max_new_tokens = int(kwargs.get("max_new_tokens", self.cfg.max_new_tokens))
        temperature = float(kwargs.get("temperature", self.cfg.temperature))
        top_k = int(kwargs.get("top_k", self.cfg.top_k))

        ids = self.tok.encode(prompt, add_bos=True, add_eos=False)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        hidden = None

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                logits, hidden = self.model(x[:, -1:], hidden)
                nxt = logits[:, -1, :] / max(1e-5, temperature)
                values, indices = torch.topk(nxt, k=min(top_k, nxt.shape[-1]))
                probs = torch.softmax(values, dim=-1)
                pick = torch.multinomial(probs, num_samples=1)
                token = indices.gather(-1, pick)
                if token.item() == self.tok.eos_id:
                    break
                x = torch.cat([x, token], dim=1)

        generated = x[0].tolist()[len(ids):]
        return self.tok.decode(generated).strip()
