#!/usr/bin/env python3
import json
import sys


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: compare.py <linfer_tok_s> <ollama_tok_s>")
        return 1

    linfer = float(sys.argv[1])
    ollama = float(sys.argv[2])
    speedup = linfer / ollama if ollama > 0 else 0.0

    print(json.dumps({
        "linfer_tok_s": linfer,
        "ollama_tok_s": ollama,
        "speedup": speedup,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
