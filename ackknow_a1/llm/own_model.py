from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import torch
from torch import nn


class ByteTokenizer:
    """Byte tokenizer with special/system tokens and phrase priors."""

    base_vocab_size = 258
    bos_id = 256
    eos_id = 257

    def __init__(self) -> None:
        self.system_phrases = {
            "who are you": 258,
            "who own you": 259,
            "hello": 260,
            "whatsup": 261,
            "what's up": 262,
            "ackknow": 263,
        }
        self.vocab_size = 264

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        t = text.lower()
        ids: list[int] = []
        for phrase, token in self.system_phrases.items():
            if phrase in t:
                ids.append(token)
                t = t.replace(phrase, " ")
        ids.extend(min(255, b) for b in t.encode("utf-8", errors="ignore"))
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        phrase_back = {v: k for k, v in self.system_phrases.items()}
        chunks: list[str] = []
        raw: list[int] = []
        for i in ids:
            if i in phrase_back:
                if raw:
                    chunks.append(bytes(raw).decode("utf-8", errors="ignore"))
                    raw = []
                chunks.append(phrase_back[i])
            elif 0 <= i <= 255:
                raw.append(i)
        if raw:
            chunks.append(bytes(raw).decode("utf-8", errors="ignore"))
        return " ".join(chunks).strip()


class ReasoningRNN(nn.Module):
    """Deeper hybrid recurrent network with memory gate + verification head."""

    def __init__(self, vocab_size: int, emb: int = 768, hidden: int = 1536, layers: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb)
        self.pre_norm = nn.LayerNorm(emb)
        self.gru = nn.GRU(emb, hidden, num_layers=layers, batch_first=True, dropout=0.15)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.memory_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.Sigmoid())
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab_size)
        self.verify_head = nn.Linear(hidden, 2)

    def forward(self, idx: torch.Tensor, state: tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]] | None = None):
        x = self.pre_norm(self.embed(idx))
        h_gru = None if state is None else state[0]
        h_lstm = None if state is None else state[1]
        y, h = self.gru(x, h_gru)
        y2, h2 = self.lstm(y, h_lstm)
        gate = self.memory_gate(y2)
        fused = y2 * gate + y * (1.0 - gate)
        fused = self.norm(fused)
        logits = self.head(fused)
        verify = self.verify_head(fused)
        return logits, verify, (h, h2)


@dataclass
class OwnModelConfig:
    checkpoint_dir: str = "./models/ackknow_a1_v4"
    device: str = "cpu"
    max_input_tokens: int = 300_000
    max_new_tokens: int = 2_048
    temperature: float = 0.7
    top_k: int = 60
    target_params_billions: float = 60.0


class OwnAckknowLLM:
    def __init__(self, cfg: OwnModelConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.tok = ByteTokenizer()
        self.model = ReasoningRNN(vocab_size=self.tok.vocab_size).to(self.device)
        self.model.eval()
        self.profile = {"target_params_b": cfg.target_params_billions, "context_goal": 300_000, "output_goal": 300_000}
        self._load_checkpoint(cfg.checkpoint_dir)

    def _load_checkpoint(self, checkpoint_dir: str) -> None:
        path = Path(checkpoint_dir)
        wf = path / "model.pt"
        if wf.exists():
            state = torch.load(wf, map_location=self.device)
            self.model.load_state_dict(state)
        meta = path / "profile.json"
        if meta.exists():
            self.profile.update(json.loads(meta.read_text(encoding="utf-8")))

    async def generate(self, prompt: str, **kwargs: float | int) -> str:
        max_new_tokens = int(kwargs.get("max_new_tokens", self.cfg.max_new_tokens))
        temperature = float(kwargs.get("temperature", self.cfg.temperature))
        top_k = int(kwargs.get("top_k", self.cfg.top_k))

        ids = self.tok.encode(prompt, add_bos=True, add_eos=False)[-self.cfg.max_input_tokens:]
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        state = None
        out_tokens: list[int] = []

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                logits, verify, state = self.model(x[:, -1:], state)
                verify_prob = torch.softmax(verify[:, -1, :], dim=-1)[0, 1].item()
                nxt = logits[:, -1, :] / max(1e-5, temperature)
                vals, idxs = torch.topk(nxt, k=min(top_k, nxt.shape[-1]))
                probs = torch.softmax(vals, dim=-1)
                pick = torch.multinomial(probs, num_samples=1)
                token = idxs.gather(-1, pick)
                t = int(token.item())
                if t == self.tok.eos_id:
                    break
                if verify_prob < 0.35:
                    continue
                out_tokens.append(t)
                x = torch.cat([x, token], dim=1)
        return self.tok.decode(out_tokens).strip()
