from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from ackknow_a1.llm.own_model import ByteTokenizer, ReasoningRNN


CURRICULUM = [
    "warmup: conversation basics + identity patterns",
    "phase-1: coding tasks + bugfix + patch reasoning",
    "phase-2: mathematics proofs + long derivations",
    "phase-3: tool-use traces + verification",
    "phase-4: multilingual Indian dialogue and intent fidelity",
]

DATASET_MIX = {
    "FineWeb-Edu": "reasoning and language",
    "TheStack": "code synthesis",
    "StarCoderData": "coding style",
    "OpenMathInstruct": "math reasoning",
    "NuminaMath": "competition math",
    "ProofPile2": "formal reasoning",
    "Wikipedia": "factual grounding",
    "IndicCorp": "Hindi/Hinglish/Tamil",
    "UltraChat": "assistant dialogue",
}


def synthetic_pairs(n: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for i in range(n):
        a = random.randint(1000, 9999)
        b = random.randint(1000, 9999)
        out.append({"text": f"User: solve {a}+{b}\nAssistant: {a+b}"})
        out.append({"text": f"User: write python fn square for {i}\nAssistant: def square(x):\n    return x*x"})
    return out


class JsonlDataset(Dataset):
    def __init__(self, file_path: str, seq_len: int = 4096, synth_count: int = 2000) -> None:
        self.tok = ByteTokenizer()
        rows: list[dict[str, str]] = []
        for line in Path(file_path).read_text(encoding="utf-8").splitlines():
            rows.append(json.loads(line))
        rows.extend(synthetic_pairs(synth_count))

        self.samples: list[list[int]] = []
        for obj in rows:
            text = obj.get("text") or f"User: {obj.get('prompt','')}\nAssistant: {obj.get('response','')}"
            ids = self.tok.encode(text, add_bos=True, add_eos=True)
            for i in range(0, max(1, len(ids)-1), seq_len):
                ch = ids[i:i+seq_len+1]
                if len(ch) > 2:
                    self.samples.append(ch)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        c = self.samples[i]
        return torch.tensor(c[:-1]), torch.tensor(c[1:])


def collate(batch):
    m = max(x.shape[0] for x, _ in batch)
    xs, ys = [], []
    for x, y in batch:
        pad = m - x.shape[0]
        xs.append(torch.nn.functional.pad(x, (0, pad), value=0))
        ys.append(torch.nn.functional.pad(y, (0, pad), value=-100))
    return torch.stack(xs), torch.stack(ys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--synthetic", type=int, default=2000)
    args = ap.parse_args()

    ds = JsonlDataset(args.data, args.seq_len, synth_count=args.synthetic)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    tok = ByteTokenizer()
    model = ReasoningRNN(vocab_size=tok.vocab_size).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lm_loss = nn.CrossEntropyLoss(ignore_index=-100)
    vf_loss = nn.CrossEntropyLoss()

    for _ in range(args.epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(args.device), yb.to(args.device)
            logits, verify, _, _ = model(xb)
            l1 = lm_loss(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            vtarget = (yb != -100).long().reshape(-1)
            l2 = vf_loss(verify.reshape(-1, 2), vtarget)
            loss = l1 + 0.2 * l2
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    (out / "profile.json").write_text(json.dumps({
        "target_params_b": 40.0,
        "context_goal": 100000,
        "output_goal": 25000,
        "curriculum": CURRICULUM,
        "dataset_mix": DATASET_MIX,
        "synthetic_examples": args.synthetic,
    }, indent=2), encoding="utf-8")
