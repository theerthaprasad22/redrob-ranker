#!/usr/bin/env python3
"""
precompute.py — OPTIONAL offline preparation. NOT part of the timed ranking step.

Two independent jobs:

  1. --cache-model    Download and cache a sentence-transformers model so that
                      `rank.py --encoder st` can run later with NO network.

  2. --augment-spec   Use a FREE LLM (Ollama / Gemini / Groq) to read the JD and
                      suggest extra synonym terms for the role spec — demonstrating
                      LLM-driven JD understanding. Writes suggestions to a file you
                      can fold into config/role_spec.yaml. Falls back gracefully if
                      no LLM is configured.

Examples:
    python precompute.py --cache-model
    python precompute.py --cache-model --model BAAI/bge-small-en-v1.5
    LLM_PROVIDER=ollama LLM_MODEL=llama3.1 python precompute.py --augment-spec --jd job_description.txt
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from redrob_ranker.llm import LLMClient, load_dotenv    # noqa: E402
from redrob_ranker.role_spec import RoleSpec       # noqa: E402

load_dotenv(os.path.join(HERE, ".env"))
load_dotenv(".env")


def cache_model(model_name: str) -> int:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers not installed. Run: pip install sentence-transformers")
        return 1
    print(f"Downloading + caching '{model_name}' (one-time, needs network)...")
    SentenceTransformer(model_name, device="cpu")
    print("Done. The model is cached locally; `rank.py --encoder st` now runs offline.")
    return 0


_AUGMENT_SYSTEM = (
    "You are helping build a candidate-ranking system. Given a job description, "
    "list additional concrete synonym terms / tools / phrases (lowercase) that a "
    "strong candidate might use to describe each capability, so a matcher can "
    "recognise plain-language profiles. Output compact YAML only, mapping each "
    "must-have concept to a list of extra terms. No prose."
)


def augment_spec(jd_path: str, out_path: str) -> int:
    if not os.path.exists(jd_path):
        print(f"JD file not found: {jd_path} (provide the JD as plain text/markdown).")
        return 1
    jd = open(jd_path, "r", encoding="utf-8", errors="ignore").read()[:8000]
    spec = RoleSpec.load()
    concepts = list(spec.must_have.keys())

    client = LLMClient()
    if not client.enabled:
        print("No LLM provider configured (set LLM_PROVIDER=ollama|gemini|groq). "
              "Nothing to augment.")
        return 1

    prompt = (
        f"Must-have concepts: {concepts}\n\n"
        f"Job description:\n{jd}\n\n"
        "Return YAML mapping each concept to a list of extra lowercase terms."
    )
    out = client.complete(prompt, system=_AUGMENT_SYSTEM, max_tokens=600, temperature=0.2)
    if not out:
        print("LLM returned nothing (provider error or empty). No file written.")
        return 1
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# LLM-suggested extra terms — review, then merge into role_spec.yaml\n")
        f.write(out.strip() + "\n")
    print(f"Wrote LLM term suggestions to {out_path}. Review and merge as desired.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Optional offline precompute (not the ranking step)")
    ap.add_argument("--cache-model", action="store_true",
                    help="Download + cache a sentence-transformers model for --encoder st")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5",
                    help="Model name to cache")
    ap.add_argument("--augment-spec", action="store_true",
                    help="Use a free LLM to suggest extra role-spec terms from the JD")
    ap.add_argument("--jd", default="job_description.txt", help="Path to JD text/markdown")
    ap.add_argument("--out", default="role_spec_suggestions.yaml",
                    help="Where to write LLM suggestions")
    args = ap.parse_args()

    if not (args.cache_model or args.augment_spec):
        ap.print_help()
        return 0

    rc = 0
    if args.cache_model:
        rc |= cache_model(args.model)
    if args.augment_spec:
        rc |= augment_spec(args.jd, args.out)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
