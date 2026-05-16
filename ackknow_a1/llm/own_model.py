from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import torch
from torch import nn


class ByteTokenizer:
    """Byte tokenizer with phrase priors and optional merge table.

    Keeps deterministic byte fallback while allowing learned merges stored in merges.json.
    """

    base_vocab_size = 258
    bos_id = 256
    eos_id = 257

    def __init__(self, merge_path: str | None = None) -> None:
        self.system_phrases = {
            "who are you": 258,
            "who own you": 259,
            "hello": 260,
            "whatsup": 261,
            "what's up": 262,
            "ackknow": 263,
        }
        self.vocab_size = 264
        self.merges: dict[str, int] = {}
        if merge_path and Path(merge_path).exists():
            self.merges = json.loads(Path(merge_path).read_text(encoding="utf-8"))
            self.vocab_size = max(self.vocab_size, max(self.merges.values(), default=self.vocab_size - 1) + 1)

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        t = text.lower()
        ids: list[int] = []
        for phrase, token in self.system_phrases.items():
            if phrase in t:
                ids.append(token)
                t = t.replace(phrase, " ")

        for piece in t.split():
            if piece in self.merges:
                ids.append(self.merges[piece])
            else:
                ids.extend(min(255, b) for b in piece.encode("utf-8", errors="ignore"))
                ids.append(32)

        if ids and ids[-1] == 32:
            ids.pop()
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        phrase_back = {v: k for k, v in self.system_phrases.items()}
        merge_back = {v: k for k, v in self.merges.items()}
        chunks: list[str] = []
        raw: list[int] = []
        for i in ids:
            if i in phrase_back:
                if raw:
                    chunks.append(bytes(raw).decode("utf-8", errors="ignore"))
                    raw = []
                chunks.append(phrase_back[i])
            elif i in merge_back:
                if raw:
                    chunks.append(bytes(raw).decode("utf-8", errors="ignore"))
                    raw = []
                chunks.append(merge_back[i])
            elif 0 <= i <= 255:
                raw.append(i)
        if raw:
            chunks.append(bytes(raw).decode("utf-8", errors="ignore"))
        return " ".join([c for c in chunks if c]).strip()


class ReasoningRNN(nn.Module):
    """High-capacity recurrent reasoning network with chunk memory."""

    def __init__(self, vocab_size: int, emb: int = 1024, hidden: int = 2048, layers: int = 10, chunk_mem: int = 1024) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb)
        self.pre_norm = nn.LayerNorm(emb)
        self.gru = nn.GRU(emb, hidden, num_layers=layers, batch_first=True, dropout=0.15)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=3, batch_first=True, dropout=0.1)
        self.memory_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.Sigmoid())
        self.chunk_proj = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab_size)
        self.verify_head = nn.Linear(hidden, 2)
        self.chunk_mem = chunk_mem

    def forward(self, idx: torch.Tensor, state=None, memory: torch.Tensor | None = None):
        x = self.pre_norm(self.embed(idx))
        h_gru = None if state is None else state[0]
        h_lstm = None if state is None else state[1]
        y, h = self.gru(x, h_gru)
        y2, h2 = self.lstm(y, h_lstm)

        if memory is not None:
            y2 = y2 + self.chunk_proj(memory).unsqueeze(1)

        gate = self.memory_gate(y2)
        fused = self.norm(y2 * gate + y * (1.0 - gate))
        logits = self.head(fused)
        verify = self.verify_head(fused)
        new_memory = fused[:, -1, :].detach()
        return logits, verify, (h, h2), new_memory


@dataclass
class OwnModelConfig:
    checkpoint_dir: str = "./models/ackknow_a1_v4"
    device: str = "cpu"
    max_input_tokens: int = 100_000
    max_new_tokens: int = 25_000
    temperature: float = 0.65
    top_k: int = 80
    target_params_billions: float = 40.0
    use_compile: bool = True
    quantization: str = "int8_dynamic"  # none|int8_dynamic


class OwnAckknowLLM:
    def __init__(self, cfg: OwnModelConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        merge_path = str(Path(cfg.checkpoint_dir) / "merges.json")
        self.tok = ByteTokenizer(merge_path=merge_path)
        self.model = ReasoningRNN(vocab_size=self.tok.vocab_size).to(self.device)
        self.profile = {
            "target_params_b": cfg.target_params_billions,
            "context_goal": cfg.max_input_tokens,
            "output_goal": cfg.max_new_tokens,
            "runtime_quant": cfg.quantization,
        }
        self._load_checkpoint(cfg.checkpoint_dir)
        self._optimize_runtime()
        self.model.eval()

    def _optimize_runtime(self) -> None:
        if self.cfg.quantization == "int8_dynamic" and self.device.type == "cpu":
            self.model = torch.quantization.quantize_dynamic(self.model, {nn.Linear}, dtype=torch.qint8)
        if self.cfg.use_compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model, mode="reduce-overhead")

    def _load_checkpoint(self, checkpoint_dir: str) -> None:
        path = Path(checkpoint_dir)
        wf = path / "model.pt"
        if wf.exists():
            state = torch.load(wf, map_location=self.device)
            self.model.load_state_dict(state, strict=False)
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
        mem = None
        out: list[int] = []

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                logits, verify, state, mem = self.model(x[:, -1:], state, mem)
                verify_prob = torch.softmax(verify[:, -1, :], dim=-1)[0, 1].item()
                if verify_prob < 0.30:
                    continue
                nxt = logits[:, -1, :] / max(1e-6, temperature)
                vals, idxs = torch.topk(nxt, k=min(top_k, nxt.shape[-1]))
                probs = torch.softmax(vals, dim=-1)
                tok = idxs.gather(-1, torch.multinomial(probs, num_samples=1))
                tid = int(tok.item())
                if tid == self.tok.eos_id:
                    break
                out.append(tid)
                x = torch.cat([x, tok], dim=1)

        return self.tok.decode(out).strip()
