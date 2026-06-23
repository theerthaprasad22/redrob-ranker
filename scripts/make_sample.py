#!/usr/bin/env python3
"""
make_sample.py — extract a small candidate sample from the full pool.

Useful for the sandbox/demo (which only needs <=100 candidates) and for quick
local iteration. Can take the first N lines, or a random sample, or a "mixed"
sample that guarantees some strong contenders alongside random decoys so the
demo actually shows good ranking.

Examples:
    python scripts/make_sample.py --candidates candidates.jsonl --out sample_data/demo.jsonl --n 150
    python scripts/make_sample.py --candidates candidates.jsonl --out sample_data/demo.jsonl --n 150 --mode random
"""

from __future__ import annotations

import argparse
import gzip
import json
import random


def _open(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") \
        else open(path, "r", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--mode", choices=["head", "random", "mixed"], default="mixed")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    if args.mode == "head":
        lines = []
        with _open(args.candidates) as f:
            for line in f:
                if line.strip():
                    lines.append(line)
                if len(lines) >= args.n:
                    break
    else:
        # reservoir sample, plus (for "mixed") keep candidates that look like
        # AI/ML engineers so the demo has clear signal
        reservoir: list[str] = []
        strong: list[str] = []
        ai_titles = ("ai engineer", "machine learning", "ml engineer", "applied",
                     "nlp", "search engineer", "recommendation", "data scientist")
        with _open(args.candidates) as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                if args.mode == "mixed" and len(strong) < args.n // 3:
                    try:
                        t = json.loads(line)["profile"]["current_title"].lower()
                        if any(a in t for a in ai_titles):
                            strong.append(line)
                            continue
                    except Exception:
                        pass
                if len(reservoir) < args.n:
                    reservoir.append(line)
                else:
                    j = random.randint(0, i)
                    if j < args.n:
                        reservoir[j] = line
        lines = (strong + reservoir)[: args.n]
        random.shuffle(lines)

    with open(args.out, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line if line.endswith("\n") else line + "\n")
    print(f"Wrote {len(lines)} candidates to {args.out} (mode={args.mode}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
