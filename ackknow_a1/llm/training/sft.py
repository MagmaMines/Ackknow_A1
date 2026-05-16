"""Supervised fine-tuning entrypoint for Ackknow-owned LLM.

Example:
python -m ackknow_a1.llm.training.sft --base mistralai/Mistral-7B-Instruct-v0.3 --data ./data/sft.jsonl --out ./models/ackknow-a1
"""

from __future__ import annotations
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.base)
    ds = load_dataset("json", data_files=args.data)["train"]

    def prep(x):
        text = x.get("text") or f"User: {x['prompt']}\nAssistant: {x['response']}"
        enc = tok(text, truncation=True, max_length=4096)
        enc["labels"] = enc["input_ids"].copy()
        return enc

    ds = ds.map(prep, remove_columns=ds.column_names)
    targs = TrainingArguments(output_dir=args.out, num_train_epochs=args.epochs, per_device_train_batch_size=1, gradient_accumulation_steps=8)
    trainer = Trainer(model=model, args=targs, train_dataset=ds)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)


if __name__ == "__main__":
    main()
