#!/usr/bin/env python3
"""
rank.py — produce the top-100 submission CSV from the candidate pool.

This is the single command the challenge asks for (Stage-3 reproduction):

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

It runs CPU-only, in-memory, with NO network and NO model download when using
the default LSA encoder — well within the 5-min / 16 GB budget. The optional
`--encoder st` (sentence-transformers) and `--reasoning-llm` paths use a locally
cached model / a free LLM and are documented in the README; neither is required.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from redrob_ranker.llm import LLMClient, load_dotenv     # noqa: E402
from redrob_ranker.pipeline import rank_candidates     # noqa: E402
from redrob_ranker.role_spec import RoleSpec           # noqa: E402
from redrob_ranker.schema import load_candidates       # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Redrob candidate ranker")
    ap.add_argument("--candidates", required=True,
                    help="Path to candidates.jsonl (or .jsonl.gz)")
    ap.add_argument("--out", default="submission.csv", help="Output CSV path")
    ap.add_argument("--role-spec", default=None,
                    help="Path to role_spec.yaml (default: config/role_spec.yaml)")
    ap.add_argument("--encoder", default="lsa", choices=["lsa", "st"],
                    help="Semantic encoder: lsa (default, local) or st (sentence-transformers)")
    ap.add_argument("--top", type=int, default=100, help="Number of candidates to output")
    ap.add_argument("--shortlist", type=int, default=6000,
                    help="Deep-scoring shortlist size (recall funnel)")
    ap.add_argument("--reasoning-llm", action="store_true",
                    help="Polish reasoning with a free LLM (offline step; see README)")
    ap.add_argument("--quiet", action="store_true", help="Suppress progress logs")
    args = ap.parse_args()

    # Load a local .env (repo root or current dir) so LLM keys can live in a
    # gitignored file instead of being exported every session.
    load_dotenv(os.path.join(HERE, ".env"))
    load_dotenv(".env")

    t0 = time.time()
    if not args.quiet:
        print(f"Loading candidates from {args.candidates} ...", flush=True)
    candidates = load_candidates(args.candidates)
    spec = RoleSpec.load(args.role_spec)

    llm_client = LLMClient() if args.reasoning_llm else None
    if args.reasoning_llm and not (llm_client and llm_client.enabled):
        print("  [info] --reasoning-llm set but no LLM provider configured; "
              "using deterministic reasoning.", flush=True)
        llm_client = None

    rows = rank_candidates(
        candidates, spec,
        encoder_kind=args.encoder,
        shortlist_k=args.shortlist,
        n_results=args.top,
        llm_client=llm_client,
        verbose=not args.quiet,
    )

    # write CSV — header EXACTLY as the validator requires
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in rows:
            w.writerow([r.candidate_id, r.rank, f"{r.score:.6f}", r.reasoning])

    if not args.quiet:
        print(f"\nWrote {len(rows)} rows to {args.out} in {time.time() - t0:.1f}s total.",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
