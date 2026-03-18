"""CLI for neuroevo."""
import sys, json, argparse
from .core import Neuroevo

def main():
    parser = argparse.ArgumentParser(description="NeuroEvo — Neural Architecture Evolution. Evolutionary algorithms for discovering novel neural architectures.")
    parser.add_argument("command", nargs="?", default="status", choices=["status", "run", "info"])
    parser.add_argument("--input", "-i", default="")
    args = parser.parse_args()
    instance = Neuroevo()
    if args.command == "status":
        print(json.dumps(instance.get_stats(), indent=2))
    elif args.command == "run":
        print(json.dumps(instance.search(input=args.input or "test"), indent=2, default=str))
    elif args.command == "info":
        print(f"neuroevo v0.1.0 — NeuroEvo — Neural Architecture Evolution. Evolutionary algorithms for discovering novel neural architectures.")

if __name__ == "__main__":
    main()
