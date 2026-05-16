from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import torch
from torch import nn


class ByteTokenizer:
    base_vocab_size = 258
    bos_id = 256
    eos_id = 257

    def __init__(self, merge_path: str | None = None) -> None:
        self.system_phrases = {
            "who are you": 258,
            "who own you": 259,
            "hello": 260,
            "whatsup": 261,
            "termux": 262,
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
        raw: list[int] = []
        out: list[str] = []
        for i in ids:
            if i in phrase_back:
                if raw:
                    out.append(bytes(raw).decode("utf-8", errors="ignore"))
                    raw = []
                out.append(phrase_back[i])
            elif i in merge_back:
                if raw:
                    out.append(bytes(raw).decode("utf-8", errors="ignore"))
                    raw = []
                out.append(merge_back[i])
            elif 0 <= i <= 255:
                raw.append(i)
        if raw:
            out.append(bytes(raw).decode("utf-8", errors="ignore"))
        return " ".join([x for x in out if x]).strip()


class MiniTransformer(nn.Module):
    def __init__(self, dim: int, heads: int = 8, ff_mult: int = 4) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff_mult * dim), nn.GELU(), nn.Linear(ff_mult * dim, dim))

    def forward(self, x: torch.Tensor, cache_kv: tuple[torch.Tensor, torch.Tensor] | None = None):
        z = self.ln1(x)
        if cache_kv is not None:
            k_prev, v_prev = cache_kv
            k = torch.cat([k_prev, z], dim=1)
            v = torch.cat([v_prev, z], dim=1)
        else:
            k = v = z
        attn_out, _ = self.attn(z, k, v, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, (k.detach(), v.detach())


class ReasoningRNN(nn.Module):
    def __init__(self, vocab_size: int, emb: int = 896, hidden: int = 1792, layers: int = 8, attn_heads: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb)
        self.pre_norm = nn.LayerNorm(emb)
        self.gru = nn.GRU(emb, hidden, num_layers=layers, batch_first=True, dropout=0.12)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.mem_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.Sigmoid())
        self.bridge = nn.Linear(hidden, emb)
        self.tx = MiniTransformer(emb, heads=attn_heads)
        self.post = nn.LayerNorm(emb)
        self.lm_head = nn.Linear(emb, vocab_size)
        self.verify_head = nn.Linear(emb, 2)

    def forward(self, idx: torch.Tensor, rnn_state=None, kv_cache=None, memory: torch.Tensor | None = None):
        x = self.pre_norm(self.embed(idx))
        h_gru = None if rnn_state is None else rnn_state[0]
        h_lstm = None if rnn_state is None else rnn_state[1]
        y, h1 = self.gru(x, h_gru)
        y2, h2 = self.lstm(y, h_lstm)
        gate = self.mem_gate(y2)
        fused = y2 * gate + y * (1.0 - gate)
        if memory is not None:
            fused = fused + memory.unsqueeze(1)
        z = self.bridge(fused)
        z, new_kv = self.tx(z, cache_kv=kv_cache)
        z = self.post(z)
        logits = self.lm_head(z)
        verify = self.verify_head(z)
        new_mem = fused[:, -1, :].detach()
        return logits, verify, (h1, h2), new_kv, new_mem


@dataclass
class OwnModelConfig:
    checkpoint_dir: str = "./models/ackknow_a1_v5"
    checkpoint_name: str = "model.pt"
    device: str = "cpu"
    max_input_tokens: int = 100_000
    max_new_tokens: int = 25_000
    temperature: float = 0.65
    top_k: int = 80
    use_compile: bool = True
    quantization: str = "int8_dynamic"
    termux_mode: bool = False


class OwnAckknowLLM:
    def __init__(self, cfg: OwnModelConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        merge_path = str(Path(cfg.checkpoint_dir) / "merges.json")
        self.tok = ByteTokenizer(merge_path=merge_path)
        self.model = ReasoningRNN(vocab_size=self.tok.vocab_size).to(self.device)
        self.profile: dict[str, object] = {"runtime": "own", "context_goal": cfg.max_input_tokens, "output_goal": cfg.max_new_tokens}
        self._load_checkpoint(cfg.checkpoint_dir, cfg.checkpoint_name)
        self._optimize_runtime()
        self.model.eval()

    def _load_checkpoint(self, checkpoint_dir: str, checkpoint_name: str) -> None:
        path = Path(checkpoint_dir)
        ckpt = path / checkpoint_name
        if ckpt.exists():
            state = torch.load(ckpt, map_location=self.device)
            self.model.load_state_dict(state, strict=False)
        manifest = path / "checkpoints.json"
        if manifest.exists():
            self.profile["checkpoints"] = json.loads(manifest.read_text(encoding="utf-8"))

    def _optimize_runtime(self) -> None:
        if self.cfg.termux_mode:
            self.cfg.use_compile = False
            self.cfg.quantization = "none"
            self.cfg.max_new_tokens = min(self.cfg.max_new_tokens, 512)
        if self.cfg.quantization == "int8_dynamic" and self.device.type == "cpu":
            self.model = torch.quantization.quantize_dynamic(self.model, {nn.Linear}, dtype=torch.qint8)
        if self.cfg.use_compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model, mode="reduce-overhead")

    async def generate(self, prompt: str, **kwargs: float | int) -> str:
        max_new = int(kwargs.get("max_new_tokens", self.cfg.max_new_tokens))
        temperature = float(kwargs.get("temperature", self.cfg.temperature))
        top_k = int(kwargs.get("top_k", self.cfg.top_k))

        ids = self.tok.encode(prompt, add_bos=True, add_eos=False)[-self.cfg.max_input_tokens:]
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        rnn_state = None
        kv_cache = None
        memory = None
        out: list[int] = []

        with torch.inference_mode():
            for _ in range(max_new):
                logits, verify, rnn_state, kv_cache, memory = self.model(x[:, -1:], rnn_state, kv_cache, memory)
                if torch.softmax(verify[:, -1, :], dim=-1)[0, 1].item() < 0.30:
                    continue
                next_logits = logits[:, -1, :] / max(1e-6, temperature)
                vals, idxs = torch.topk(next_logits, k=min(top_k, next_logits.shape[-1]))
                probs = torch.softmax(vals, dim=-1)
                tok = idxs.gather(-1, torch.multinomial(probs, num_samples=1))
                tid = int(tok.item())
                if tid == self.tok.eos_id:
                    break
                out.append(tid)
                x = torch.cat([x, tok], dim=1)
        return self.tok.decode(out).strip()
