#!/usr/bin/env python3
"""
parse_resumes.py — OFFLINE résumé/profile → structured candidate JSON.

This is a *pre-step*, NOT part of the timed ranking run. It reads free-text
résumés and uses a FREE LLM (Ollama / Gemini / Groq, configured exactly like the
rest of the project) to extract a structured record per candidate, then writes
`candidates.jsonl` that `rank.py` can consume normally:

    # 1) parse résumés -> structured JSONL (needs a configured LLM; offline OK with Ollama)
    LLM_PROVIDER=ollama LLM_MODEL=llama3.1 \
        python parse_resumes.py --in resumes.txt --out candidates.jsonl

    # 2) rank exactly as usual (no network, no LLM)
    python rank.py --candidates candidates.jsonl --out submission.csv

Why this exists
---------------
Without parsing, a plain résumé becomes just a `summary` blob, so the ranker's
trust-weighted-skills and career-evidence dimensions have almost nothing to work
with. Extracting `skills` (with proficiency), `career_history`, and `education`
unlocks those dimensions.

Honest limits (by design)
-------------------------
* The LLM only extracts what is **present or clearly implied** in the text. The
  prompt forbids inventing employers, skills, dates, or numbers.
* It **cannot** produce the Redrob behavioral signals (recruiter_response_rate,
  last_active_date, github_activity_score, …) — those are platform-internal data
  that simply do not exist in a résumé. `redrob_signals` is therefore left empty,
  and the signal modifier treats it neutrally (same for every parsed candidate),
  so résumé scores lean on the semantic / skills / career dimensions.
* If the LLM is unavailable for a given résumé, that record falls back to a
  minimal text-only record (whole text → summary) so no candidate is lost.

The transport degrades gracefully (LLMClient returns None on any error); this
tool never crashes mid-run.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Callable

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from redrob_ranker.llm import LLMClient, load_dotenv          # noqa: E402
from redrob_ranker.ingest import _split_text_blocks, _text_block_to_record  # noqa: E402

load_dotenv(os.path.join(HERE, ".env"))
load_dotenv(".env")

_PROFICIENCY = {"beginner", "intermediate", "advanced", "expert"}

_SYSTEM = (
    "You extract structured data from a candidate's résumé / professional profile "
    "for a hiring system. Output ONLY a single JSON object — no markdown, no code "
    "fences, no commentary. Use ONLY information that is present or clearly implied "
    "in the text. NEVER invent employers, job titles, skills, dates, numbers, or "
    "credentials. If a field is unknown, omit it (or use null). Do not add skills "
    "the text does not support."
)

# Compact target shape shown to the model (a subset of candidate_schema.json that
# is realistically extractable from a résumé).
_SHAPE = """{
  "profile": {
    "anonymized_name": str, "headline": str, "summary": str,
    "current_title": str, "current_company": str, "current_industry": str,
    "location": str, "country": str, "years_of_experience": number
  },
  "career_history": [
    {"title": str, "company": str, "industry": str,
     "start_date": "YYYY-MM-DD|null", "end_date": "YYYY-MM-DD|null",
     "duration_months": int, "is_current": bool, "description": str}
  ],
  "education": [
    {"institution": str, "degree": str, "field_of_study": str,
     "start_year": int, "end_year": int}
  ],
  "skills": [
    {"name": str, "proficiency": "beginner|intermediate|advanced|expert",
     "endorsements": int, "duration_months": int}
  ],
  "certifications": [{"name": str, "issuer": str, "year": int}],
  "languages": [{"language": str, "proficiency": str}]
}"""


def _build_prompt(text: str) -> str:
    return (
        "Extract a JSON object with EXACTLY this shape (omit any unknown field):\n"
        f"{_SHAPE}\n\n"
        "Rules:\n"
        "- proficiency must be one of: beginner, intermediate, advanced, expert. "
        "Infer conservatively (use 'expert'/'advanced' only when the text shows "
        "deep/long use; otherwise 'intermediate' or 'beginner').\n"
        "- endorsements: 0 unless the text gives a number.\n"
        "- years_of_experience: a number; estimate from the career history if not stated.\n"
        "- Keep 'summary' to the candidate's own words / a faithful condensation.\n"
        "- Do NOT include any field you are not confident is supported by the text.\n\n"
        "Résumé text:\n<<<\n" + text.strip() + "\n>>>\n\n"
        "Return ONLY the JSON object."
    )


# --------------------------------------------------------------------------- #
# Core extraction (LLM-agnostic; `complete` is injectable for testing)
# --------------------------------------------------------------------------- #

def extract_record(
    text: str,
    complete: Callable[[str, str], str | None],
    *,
    candidate_id: str | None = None,
    index: int = 0,
) -> tuple[dict[str, Any], bool]:
    """Return (record, used_llm).

    ``complete(prompt, system) -> str | None`` does the generation. On any LLM
    failure / unparseable output, fall back to a minimal text-only record so the
    candidate is still rankable.
    """
    raw = None
    try:
        raw = complete(_build_prompt(text), _SYSTEM)
    except Exception:
        raw = None

    rec = _parse_json_blob(raw) if raw else None
    if rec is None:
        return _minimal_record(text, candidate_id, index), False

    rec = _normalize(rec, text)
    rec["candidate_id"] = _resolve_id(candidate_id, rec, index)
    return rec, True


def _parse_json_blob(s: str) -> dict[str, Any] | None:
    """Parse a JSON object from possibly-noisy model output."""
    if not s:
        return None
    t = s.strip()
    # Strip ``` / ```json fences if present.
    if t.startswith("```"):
        t = t.split("```", 2)[1] if t.count("```") >= 2 else t.strip("`")
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # Last resort: slice from the first '{' to the last '}'.
    i, j = t.find("{"), t.rfind("}")
    if 0 <= i < j:
        try:
            obj = json.loads(t[i:j + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _normalize(rec: dict[str, Any], text: str) -> dict[str, Any]:
    """Light, defensive normalisation into the candidate schema's shape."""
    out: dict[str, Any] = {}

    prof = rec.get("profile")
    out["profile"] = dict(prof) if isinstance(prof, dict) else {}
    # Always keep the full text as summary so the semantic encoder has material.
    if not str(out["profile"].get("summary", "")).strip():
        out["profile"]["summary"] = text.strip()
    if "years_of_experience" in out["profile"]:
        out["profile"]["years_of_experience"] = _num(out["profile"]["years_of_experience"], 0.0)

    out["skills"] = _clean_skills(rec.get("skills"))
    out["career_history"] = _clean_list(
        rec.get("career_history"),
        int_fields=("duration_months",),
        bool_fields=("is_current",),
    )
    out["education"] = _clean_list(
        rec.get("education"), int_fields=("start_year", "end_year"),
    )
    out["certifications"] = _clean_list(rec.get("certifications"), int_fields=("year",))
    out["languages"] = _clean_list(rec.get("languages"))

    # redrob_signals deliberately left empty — not extractable from a résumé.
    out["redrob_signals"] = {}
    return out


def _clean_skills(skills: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(skills, list):
        return out
    for s in skills:
        if not isinstance(s, dict) or not str(s.get("name", "")).strip():
            continue
        prof = str(s.get("proficiency", "")).strip().lower()
        entry: dict[str, Any] = {"name": str(s["name"]).strip()}
        entry["proficiency"] = prof if prof in _PROFICIENCY else "intermediate"
        entry["endorsements"] = int(_num(s.get("endorsements", 0), 0))
        if s.get("duration_months") is not None:
            entry["duration_months"] = int(_num(s.get("duration_months", 0), 0))
        out.append(entry)
    return out


def _clean_list(items: Any, *, int_fields: tuple[str, ...] = (),
                bool_fields: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        entry = {k: v for k, v in it.items() if v is not None}
        for f in int_fields:
            if f in entry:
                entry[f] = int(_num(entry[f], 0))
        for f in bool_fields:
            if f in entry:
                entry[f] = bool(entry[f]) if not isinstance(entry[f], str) \
                    else entry[f].strip().lower() in ("true", "yes", "1")
        if entry:
            out.append(entry)
    return out


def _minimal_record(text: str, candidate_id: str | None, index: int) -> dict[str, Any]:
    """Fallback when the LLM is unavailable / output is unusable.

    Reuses the ingest text parser so behaviour matches feeding the résumé
    straight to rank.py: known ``key: value`` lines are picked up and the whole
    block becomes the summary.
    """
    rec = _text_block_to_record(text)
    rec.setdefault("redrob_signals", {})
    rec["candidate_id"] = _resolve_id(candidate_id, rec, index)
    return rec


def _resolve_id(candidate_id: str | None, rec: dict[str, Any], index: int) -> str:
    if candidate_id and str(candidate_id).strip():
        return str(candidate_id).strip()
    existing = rec.get("candidate_id")
    if existing and str(existing).strip():
        return str(existing).strip()
    return f"CAND_PARSED_{index:07d}"


def _num(v: Any, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# Input reading
# --------------------------------------------------------------------------- #

def read_resumes(path: str) -> list[tuple[str | None, str]]:
    """Return a list of (candidate_id_hint, text) résumé blocks.

    * A directory → each ``*.txt`` / ``*.md`` file is one résumé (id hint = stem).
    * A single file → split into blocks on ``---`` dividers or blank-line gaps.
    """
    if os.path.isdir(path):
        out: list[tuple[str | None, str]] = []
        for fp in sorted(glob.glob(os.path.join(path, "*.txt")) +
                         glob.glob(os.path.join(path, "*.md"))):
            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                txt = f.read().strip()
            if txt:
                out.append((os.path.splitext(os.path.basename(fp))[0], txt))
        return out
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    return [(None, b) for b in _split_text_blocks(text) if b.strip()]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Offline résumé → structured candidate JSONL (LLM pre-step; "
                    "NOT part of the timed ranking run).")
    ap.add_argument("--in", dest="inp", required=True,
                    help="Résumé file (blocks separated by '---' or blank lines) "
                         "OR a directory of .txt/.md résumés (one per file).")
    ap.add_argument("--out", default="candidates.jsonl",
                    help="Output JSONL path (feed this to rank.py).")
    ap.add_argument("--provider", default=None,
                    help="Override LLM_PROVIDER (ollama|gemini|groq).")
    ap.add_argument("--model", default=None, help="Override LLM_MODEL.")
    ap.add_argument("--max-tokens", type=int, default=1500,
                    help="Max output tokens per résumé (default 1500).")
    ap.add_argument("--allow-textfallback", action="store_true",
                    help="If no LLM is configured, still write minimal text-only "
                         "records instead of exiting (equivalent to feeding the "
                         "résumé straight to rank.py).")
    args = ap.parse_args()

    if not os.path.exists(args.inp):
        print(f"Input not found: {args.inp}")
        return 1

    resumes = read_resumes(args.inp)
    if not resumes:
        print(f"No résumé text found in {args.inp}.")
        return 1

    client = LLMClient(provider=args.provider, model=args.model)
    if not client.enabled:
        msg = ("No LLM provider configured. Set LLM_PROVIDER=ollama|gemini|groq "
               "(and the relevant key) — see README / .env.example.")
        if not args.allow_textfallback:
            print(msg + "\n(Or pass --allow-textfallback to write minimal "
                        "text-only records without an LLM.)")
            return 1
        print(msg + "\nProceeding with text-only fallback (no extraction).")

    def _complete(prompt: str, system: str) -> str | None:
        return client.complete(prompt, system=system,
                               max_tokens=args.max_tokens, temperature=0.1)

    print(f"Parsing {len(resumes)} résumé(s) "
          f"(provider={client.provider or 'none'})...", flush=True)

    records: list[dict[str, Any]] = []
    n_llm = 0
    for i, (cid_hint, text) in enumerate(resumes):
        rec, used_llm = extract_record(text, _complete, candidate_id=cid_hint, index=i)
        n_llm += int(used_llm)
        records.append(rec)
        tag = "llm" if used_llm else "fallback"
        title = rec.get("profile", {}).get("current_title", "") or "(no title)"
        print(f"  [{i + 1}/{len(resumes)}] {rec['candidate_id']} "
              f"[{tag}] {title} — {len(rec.get('skills', []))} skills, "
              f"{len(rec.get('career_history', []))} roles", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(records)} records to {args.out} "
          f"({n_llm} via LLM, {len(records) - n_llm} text-fallback).")
    print(f"Next: python rank.py --candidates {args.out} --out submission.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
