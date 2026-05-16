"""Train Ackknow-native RNN LLM from JSONL corpora.

Format per line:
{"text":"User: ...\\nAssistant: ..."}
or
{"prompt":"...", "response":"..."}
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from ackknow_a1.llm.own_model import ByteTokenizer, ReasoningRNN


class JsonlDataset(Dataset):
    def __init__(self, file_path: str, seq_len: int = 512) -> None:
        self.seq_len = seq_len
        self.tok = ByteTokenizer()
        self.samples: list[list[int]] = []
        for line in Path(file_path).read_text(encoding="utf-8").splitlines():
            obj = json.loads(line)
            text = obj.get("text") or f"User: {obj.get('prompt','')}\nAssistant: {obj.get('response','')}"
            ids = self.tok.encode(text, add_bos=True, add_eos=True)
            for i in range(0, max(1, len(ids)-1), seq_len):
                chunk = ids[i:i+seq_len+1]
                if len(chunk) > 2:
                    self.samples.append(chunk)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.samples[idx]
        x = torch.tensor(c[:-1], dtype=torch.long)
        y = torch.tensor(c[1:], dtype=torch.long)
        return x, y


def collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    maxlen = max(x.shape[0] for x, _ in batch)
    xs, ys = [], []
    for x, y in batch:
        pad = maxlen - x.shape[0]
        xs.append(torch.nn.functional.pad(x, (0, pad), value=0))
        ys.append(torch.nn.functional.pad(y, (0, pad), value=-100))
    return torch.stack(xs), torch.stack(ys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ds = JsonlDataset(args.data, seq_len=args.seq_len)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    model = ReasoningRNN().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    for _ in range(args.epochs):
        for xb, yb in dl:
            xb, yb = xb.to(args.device), yb.to(args.device)
            logits, _ = model(xb)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    (out / "config.json").write_text(json.dumps({"arch": "ReasoningRNN", "vocab": 258}), encoding="utf-8")


if __name__ == "__main__":
    main()
