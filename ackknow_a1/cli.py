import argparse
import asyncio
from ackknow_a1.api.app import engine
from ackknow_a1.types import AckRequest


def main() -> None:
    parser = argparse.ArgumentParser(prog="ackknow")
    parser.add_argument("prompt")
    parser.add_argument("--user", default="default")
    args = parser.parse_args()

    resp = asyncio.run(engine.run(AckRequest(user_id=args.user, prompt=args.prompt)))
    print(resp.answer)
    print("\nModel:", resp.model_used)
    print("Confidence:", resp.confidence)
