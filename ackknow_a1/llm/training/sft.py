from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from ackknow_a1.llm.own_model import ByteTokenizer, ReasoningRNN


DATASET_MIX = {
    "FineWeb-Edu": "reasoning and language breadth",
    "TheStack": "coding competency",
    "OpenMathInstruct": "math depth",
    "NuminaMath": "symbolic math",
    "Wikipedia": "general knowledge",
    "IndicCorp": "Indian language fluency",
    "OpenHermes": "assistant behavior",
}


class JsonlDataset(Dataset):
    def __init__(self, file_path: str, seq_len: int = 2048) -> None:
        self.tok = ByteTokenizer()
        self.samples: list[list[int]] = []
        for line in Path(file_path).read_text(encoding="utf-8").splitlines():
            obj = json.loads(line)
            text = obj.get("text") or f"User: {obj.get('prompt','')}\nAssistant: {obj.get('response','')}"
            ids = self.tok.encode(text, add_bos=True, add_eos=True)
            for i in range(0, max(1, len(ids) - 1), seq_len):
                ch = ids[i:i + seq_len + 1]
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
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ds = JsonlDataset(args.data, args.seq_len)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    tok = ByteTokenizer()
    model = ReasoningRNN(vocab_size=tok.vocab_size).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lm_loss = nn.CrossEntropyLoss(ignore_index=-100)
    vf_loss = nn.CrossEntropyLoss()

    model.train()
    for _ in range(args.epochs):
        for xb, yb in dl:
            xb, yb = xb.to(args.device), yb.to(args.device)
            logits, verify, _ = model(xb)
            l1 = lm_loss(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            # verification target: token!=padding implies valid reasoning step
            vtarget = (yb != -100).long().reshape(-1)
            l2 = vf_loss(verify.reshape(-1, 2), vtarget)
            loss = l1 + 0.15 * l2
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    (out / "profile.json").write_text(json.dumps({
        "target_params_b": 60.0,
        "context_goal": 300000,
        "output_goal": 300000,
        "dataset_mix": DATASET_MIX,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
