"""
explain.py — recruiter-facing trust layer.

Given a ranked candidate, the role spec, and (optionally) the score breakdown,
produce the things a recruiter needs to *trust* a shortlist entry:

  * confidence — how much to trust this particular score, given how complete
    and internally consistent the candidate's evidence is;
  * matched / missing — which of the role's must-have signals the candidate
    does and doesn't evidence;
  * narrative — a short plain-language summary a recruiter can read at a glance.

This is presentation-only. It is computed for the handful of candidates shown
in the UI; it never feeds the ranking math or the timed bulk path, so it cannot
change scores or the submission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .role_spec import RoleSpec
from .schema import Candidate


@dataclass
class Explanation:
    confidence: str                      # "high" | "medium" | "low"
    confidence_reason: str
    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    narrative: str = ""


def _label(key: str, terms: list[str]) -> str:
    """A human label for a must-have concept: prefer the first concrete term
    (e.g. 'react', 'python') over the internal key ('javascript_ts')."""
    raw = (terms[0] if terms else key.replace("_", " ")).strip()
    return raw[:1].upper() + raw[1:] if raw else key


def _concept_present(c: Candidate, terms: list[str]) -> bool:
    names = c.skill_names_lc or set()
    blob = c.blob_lc or ""
    for t in terms:
        tl = str(t).lower()
        if not tl:
            continue
        if any(tl in n or n in tl for n in names):
            return True
        if tl in blob:
            return True
    return False


def explain_candidate(c: Candidate, spec: RoleSpec, breakdown: Any = None) -> Explanation:
    must = spec.must_have or {}
    matched: list[str] = []
    missing: list[str] = []
    for key, cfg in must.items():
        terms = list(cfg.get("terms", []))
        lab = _label(key, terms)
        (matched if _concept_present(c, terms) else missing).append(lab)

    # --- confidence: how complete and consistent is the evidence? -----------
    n_hist = sum(1 for r in c.career_history if isinstance(r, dict))
    n_skills = sum(1 for s in c.skills if isinstance(s, dict))
    pcs = float((c.signals or {}).get("profile_completeness_score", 50) or 50)

    comps = getattr(breakdown, "components", {}) or {}
    title_fit = float(comps.get("role_title_fit", 0.0) or 0.0)
    career = float(comps.get("career_evidence", 0.0) or 0.0)

    thin = (n_hist <= 1 and n_skills < 3) or n_skills == 0
    conflict = abs(title_fit - career) >= 0.6  # title says one thing, history another
    rich = n_hist >= 2 and n_skills >= 4 and pcs >= 60

    if thin:
        confidence, reason = "low", "thin profile — little career history or few listed skills"
    elif conflict:
        confidence, reason = "low", "mixed signals — title and demonstrated experience don't line up"
    elif rich:
        confidence, reason = "high", "rich, consistent profile"
    else:
        confidence, reason = "medium", "moderate evidence"

    narrative = _narrative(c, comps, matched, missing, confidence)
    return Explanation(confidence=confidence, confidence_reason=reason,
                       matched=matched, missing=missing, narrative=narrative)


def _narrative(c: Candidate, comps: dict[str, Any], matched: list[str],
               missing: list[str], confidence: str) -> str:
    title = (c.current_title or "Candidate").strip()
    yrs = float(c.years_of_experience or 0.0)

    lead = title
    if yrs > 0:
        lead += f", ~{yrs:.0f} yrs experience"
    parts = [lead]

    if matched:
        parts.append("evidences " + ", ".join(matched[:4]))
    if float(comps.get("career_evidence", 0.0) or 0.0) >= 0.6:
        parts.append("with a relevant hands-on track record")

    s = "; ".join(parts) + "."
    if missing:
        s += " Gaps for this role: " + ", ".join(missing[:4]) + "."

    sig = c.signals or {}
    rr = sig.get("recruiter_response_rate")
    if isinstance(rr, (int, float)) and rr < 0.15:
        s += " Note: low recruiter responsiveness."
    npd = sig.get("notice_period_days")
    if isinstance(npd, (int, float)) and npd >= 90:
        s += f" Long notice (~{int(npd)}d)."

    s += f" Confidence: {confidence}."
    return s
