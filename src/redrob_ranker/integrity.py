"""
integrity.py — detect "subtly impossible" honeypot profiles.

The spec ships ~80 honeypots forced to relevance tier 0; ranking >10% of them in
the top 100 is an automatic disqualification. The given signatures are:

  * "8 years of experience at a company founded 3 years ago"
        -> years_of_experience far exceeds the span the career history covers.
  * '"expert" proficiency in 10 skills with 0 years used'
        -> many advanced/expert skills with duration_months == 0.

We add a few more internal-consistency checks (dates that don't add up). Every
check is CONSERVATIVE — we only hard-flag clear impossibilities so we never
push a real candidate down. Borderline oddness becomes a small soft penalty
instead. This is exactly the "read the profile, don't just embed keywords"
behaviour the organisers are testing for.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from .schema import Candidate


def _d(s: Any) -> date | None:
    if not s or not isinstance(s, str):
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _months_between(a: date, b: date) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


@dataclass
class Integrity:
    is_honeypot: bool          # hard impossibility -> force to the bottom
    soft_penalty: float        # multiplicative in (0, 1] for minor oddities
    reasons: list[str]


def assess_integrity(c: Candidate, reference_date: date | None = None) -> Integrity:
    now = reference_date or date.today()
    reasons: list[str] = []
    hard = False
    soft = 1.0

    if not c.parse_ok:
        return Integrity(is_honeypot=True, soft_penalty=0.0,
                         reasons=["record failed to parse"])

    # --- 1. Experience exceeds the span the career actually covers ----------
    earliest = c.career_start_date()
    if earliest is not None:
        span_years = max(0.0, _months_between(earliest, now) / 12.0)
        # Allow generous slack (pre-listed experience, gaps): 2.5 years.
        if c.years_of_experience > span_years + 2.5 and c.years_of_experience >= 5:
            hard = True
            reasons.append(
                f"claims {c.years_of_experience:.0f}y experience but career history "
                f"spans only ~{span_years:.0f}y"
            )

    # --- 2. Per-role date contradictions ------------------------------------
    for r in c.career_history:
        if not isinstance(r, dict):
            continue
        sd, ed = _d(r.get("start_date")), _d(r.get("end_date"))
        dur = int(r.get("duration_months", 0) or 0)
        if sd and sd > now:
            hard = True
            reasons.append("a role starts in the future")
        if sd and ed and ed < sd:
            hard = True
            reasons.append("a role ends before it starts")
        if sd:
            end_for_span = ed or now
            span = _months_between(sd, end_for_span)
            # duration_months wildly larger than the actual date window.
            if dur > span + 6 and dur > 12:
                hard = True
                reasons.append("a role's duration exceeds its date range")

    # --- 3. "Expert in many skills with 0 months used" ----------------------
    zero_dur_expert = 0
    for s in c.skills:
        if not isinstance(s, dict):
            continue
        prof = str(s.get("proficiency", "")).lower()
        dur = int(s.get("duration_months", 0) or 0)
        endo = int(s.get("endorsements", 0) or 0)
        if prof in ("advanced", "expert") and dur == 0:
            if endo == 0:
                zero_dur_expert += 1
    if zero_dur_expert >= 5:
        hard = True
        reasons.append(
            f"{zero_dur_expert} advanced/expert skills claimed with 0 months used "
            f"and 0 endorsements"
        )
    elif zero_dur_expert >= 3:
        soft *= 0.7
        reasons.append(f"{zero_dur_expert} high-proficiency skills with 0 months used")

    # --- 4. Education date sanity -------------------------------------------
    for e in c.education:
        if not isinstance(e, dict):
            continue
        try:
            sy, ey = int(e.get("start_year", 0)), int(e.get("end_year", 0))
        except (TypeError, ValueError):
            continue
        if sy and ey and ey < sy:
            soft *= 0.8
            reasons.append("education ends before it starts")

    return Integrity(is_honeypot=hard, soft_penalty=soft, reasons=reasons)


def estimate_reference_date(candidates: list[Candidate]) -> date:
    """Use the dataset's own latest activity as 'now' (robust to wall clock)."""
    latest: date | None = None
    for c in candidates[:20000]:  # sample is plenty to find the max
        for key in ("last_active_date", "signup_date"):
            d = _d(c.signals.get(key))
            if d and (latest is None or d > latest):
                latest = d
    return latest or date.today()
