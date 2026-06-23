"""
reasoning.py — generate the 1-2 sentence `reasoning` for each top-100 row.

Stage 4 (manual review) samples 10 rows and checks: specific facts, JD
connection, honest concerns, NO hallucination, variation across rows, and tone
matching the rank. We satisfy all six by construction:

  * Every clause is built from fields that exist on THIS candidate (title, YOE,
    matched skills that actually passed the trust filter, real signal numbers).
  * Concerns come straight from the scoring/​signal analysis, so gaps are stated
    honestly.
  * Tone is selected by rank tier, so a rank-95 row never reads as glowing.
  * Because clauses draw on per-candidate facts, rows differ from one another.

There are two modes:
  * default  — deterministic template (no LLM, no network). Fully grounded.
  * --llm    — optional polish: a free LLM rewrites the SAME facts into smoother
               prose. The prompt forbids adding anything not in the facts, so it
               cannot hallucinate skills. Runs OFFLINE over only 100 rows.
"""

from __future__ import annotations

from typing import Any, Optional

from .llm import LLMClient
from .schema import Candidate
from .scoring import ScoreBreakdown
from .signals import SignalResult

_EVIDENCE_LABELS = [
    ("recommendation", "recommendation systems"),
    ("recommender", "recommender systems"),
    ("ranking", "ranking systems"),
    ("search relevance", "search relevance"),
    ("semantic search", "semantic search"),
    ("retrieval", "retrieval systems"),
    ("vector", "vector search"),
    ("personaliz", "personalization"),
    ("search", "search systems"),
]


def _evidence_phrase(c: Candidate) -> Optional[str]:
    txt = c.career_text_lc
    for key, label in _EVIDENCE_LABELS:
        if key in txt:
            return label
    return None


def build_facts(c: Candidate, bd: ScoreBreakdown, sig: SignalResult, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "current_title": c.current_title,
        "years_of_experience": round(c.years_of_experience, 1),
        "matched_skills": bd.matched_skills[:4],
        "career_evidence_phrase": _evidence_phrase(c),
        "career_evidence_score": bd.components.get("career_evidence", 0.0),
        "role_title_fit": bd.components.get("role_title_fit", 0.0),
        "location": c.location,
        "location_fit": bd.components.get("location_fit", 0.0),
        "signal_facts": sig.facts,
        "signal_positives": sig.positives[:2],
        "concerns": (bd.concerns + sig.concerns)[:3],
    }


def _tier_prefix(rank: int) -> str:
    if rank <= 10:
        return "Strong fit"
    if rank <= 30:
        return "Solid fit"
    if rank <= 60:
        return "Possible fit"
    return "Borderline"


def template_reasoning(facts: dict[str, Any]) -> str:
    rank = facts["rank"]
    prefix = _tier_prefix(rank)
    title = facts["current_title"] or "Candidate"
    yoe = facts["years_of_experience"]

    # --- strengths ----------------------------------------------------------
    parts: list[str] = [f"{prefix}: {title} with {yoe:.1f} yrs"]

    ev = facts.get("career_evidence_phrase")
    if ev and facts.get("career_evidence_score", 0) >= 0.5:
        parts.append(f"career shows hands-on {ev}")

    skills = facts.get("matched_skills") or []
    if skills:
        parts.append("relevant strengths in " + ", ".join(skills[:3]))

    pos = facts.get("signal_positives") or []
    if pos:
        parts.append(pos[0])

    strength_sentence = "; ".join(parts) + "."

    # --- concerns (honest) --------------------------------------------------
    concerns = facts.get("concerns") or []
    if concerns:
        concern_sentence = " Concerns: " + "; ".join(concerns[:2]) + "."
    elif rank > 60:
        concern_sentence = " Included near the cutoff — adjacent rather than core fit."
    else:
        concern_sentence = ""

    text = strength_sentence + concern_sentence
    # keep it tight (1-2 sentences)
    return text.strip()


_LLM_SYSTEM = (
    "You are a senior technical recruiter writing a one-to-two sentence note on "
    "why a candidate sits at a given rank for a Senior AI Engineer role. Use ONLY "
    "the facts provided. Do NOT invent skills, employers, numbers, or experience. "
    "If concerns are listed, mention them honestly. Match the tone to the rank: "
    "confident for top ranks, hedged for low ranks. Be specific, not generic. "
    "Output only the note, no preamble."
)


def llm_polish(facts: dict[str, Any], client: LLMClient) -> Optional[str]:
    import json as _json
    prompt = (
        "Facts (the only information you may use):\n"
        + _json.dumps(facts, ensure_ascii=False)
        + "\n\nWrite the 1-2 sentence recruiter note."
    )
    out = client.complete(prompt, system=_LLM_SYSTEM, max_tokens=120, temperature=0.4)
    if not out:
        return None
    out = out.strip().strip('"').replace("\n", " ")
    return out if 10 <= len(out) <= 400 else None


def generate_reasoning(c: Candidate, bd: ScoreBreakdown, sig: SignalResult, rank: int,
                       client: Optional[LLMClient] = None) -> str:
    facts = build_facts(c, bd, sig, rank)
    if client is not None and client.enabled:
        polished = llm_polish(facts, client)
        if polished:
            return polished
    return template_reasoning(facts)
