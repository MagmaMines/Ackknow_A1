import argparse
from .engine import AckknowA1
from .types import QueryContext


def main() -> None:
    parser = argparse.ArgumentParser(prog="ackknow", description="Ackknow A1 assistant")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ask = sub.add_parser("ask")
    ask.add_argument("prompt")
    ask.add_argument("--user", default="default")

    chat = sub.add_parser("chat")
    chat.add_argument("--user", default="default")
    chat.add_argument("--locale", default="en-US")

    args = parser.parse_args()
    model = AckknowA1()

    if args.cmd == "ask":
        resp = model.run(args.prompt, QueryContext(user_id=args.user))
        print(resp.answer)
        print("\nReasoning:", resp.reasoning_summary)
        print("Safeguards:")
        for g in resp.safeguards:
            print("-", g)
        return

    if args.cmd == "chat":
        print("Ackknow A1 interactive mode. Type 'exit' to stop.")
        while True:
            prompt = input("you> ").strip()
            if prompt.lower() in {"exit", "quit"}:
                break
            resp = model.run(prompt, QueryContext(user_id=args.user, locale=args.locale))
            print("ack>", resp.answer)
