"""
signals.py — behavioral-signal modifier.

The 23 Redrob signals describe whether a candidate is *actually hireable right
now*, independent of how good their profile looks. We fold them into a single
multiplier applied on top of base_fit:

    final = base_fit * behavioral_modifier        (modifier in ~[0.60, 1.15])

Recency-of-activity and recruiter-response-rate carry the most weight because
those are the JD's explicit availability concern. A strong-on-paper candidate
who is stale and unresponsive is pulled down; an engaged, responsive, verified
candidate gets a modest boost.

We also return a few plain-language signal facts so the reasoning module can
cite concrete numbers (e.g. "response rate 0.08", "120-day notice").
"""

from __future__ import annotations

from dataclasses import dataclass, field
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


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


@dataclass
class SignalResult:
    modifier: float
    quality: float                       # composite in [0,1]
    facts: dict[str, Any] = field(default_factory=dict)
    positives: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)


def behavioral_modifier(c: Candidate, spec_behavioral: dict[str, Any],
                        reference_date: date | None) -> SignalResult:
    s = c.signals or {}
    b = spec_behavioral
    now = reference_date or date.today()
    facts: dict[str, Any] = {}
    pos: list[str] = []
    con: list[str] = []

    # --- recency of activity ------------------------------------------------
    la = _d(s.get("last_active_date"))
    stale_days = b.get("stale_days", 120)
    if la is not None:
        days = (now - la).days
        facts["last_active_days_ago"] = days
        if days <= 14:
            recency = 1.0
            pos.append("active in the last 2 weeks")
        elif days <= stale_days:
            recency = 1.0 - 0.6 * (days - 14) / max(1, (stale_days - 14))
        else:
            recency = 0.2
            con.append(f"inactive for ~{days} days")
    else:
        recency = 0.5

    # --- recruiter responsiveness ------------------------------------------
    rr = s.get("recruiter_response_rate")
    if isinstance(rr, (int, float)):
        facts["recruiter_response_rate"] = round(float(rr), 2)
        response = _clip01(float(rr))
        if rr >= b.get("good_response_rate", 0.5):
            pos.append(f"responsive to recruiters ({rr:.0%})")
        elif rr < 0.15:
            con.append(f"very low recruiter response rate ({rr:.0%})")
    else:
        response = 0.5

    # --- open to work -------------------------------------------------------
    otw = bool(s.get("open_to_work_flag"))
    facts["open_to_work"] = otw
    if otw:
        pos.append("open to work")

    # --- demand (recruiters saving / searching / viewing) -------------------
    saved = float(s.get("saved_by_recruiters_30d", 0) or 0)
    appear = float(s.get("search_appearance_30d", 0) or 0)
    views = float(s.get("profile_views_received_30d", 0) or 0)
    demand = (
        0.5 * min(1.0, saved / 10.0)
        + 0.25 * min(1.0, appear / 50.0)
        + 0.25 * min(1.0, views / 50.0)
    )
    if saved >= 5:
        pos.append(f"saved by {int(saved)} recruiters recently")

    # --- track record -------------------------------------------------------
    icr = s.get("interview_completion_rate")
    interview = _clip01(float(icr)) if isinstance(icr, (int, float)) else 0.6
    oar = s.get("offer_acceptance_rate")
    offer = 0.6 if (not isinstance(oar, (int, float)) or oar < 0) else _clip01(float(oar))

    # --- profile completeness & verification --------------------------------
    pcs = float(s.get("profile_completeness_score", 50) or 50) / 100.0
    verified = sum(bool(s.get(k)) for k in ("verified_email", "verified_phone",
                                            "linkedin_connected")) / 3.0

    # --- github (engineering signal; -1 means no GitHub -> neutral) ---------
    gh = s.get("github_activity_score", -1)
    if isinstance(gh, (int, float)) and gh >= 0:
        facts["github_activity_score"] = round(float(gh), 0)
        github = _clip01(float(gh) / 100.0)
        if gh >= 50:
            pos.append(f"active GitHub (score {gh:.0f})")
    else:
        github = 0.5  # neutral

    # --- notice period ------------------------------------------------------
    npd = s.get("notice_period_days")
    if isinstance(npd, (int, float)):
        facts["notice_period_days"] = int(npd)
        good = b.get("good_notice_days", 30)
        if npd <= good:
            notice = 1.0
        else:
            notice = max(0.4, 1.0 - 0.6 * (npd - good) / 150.0)
            if npd >= 90:
                con.append(f"long notice period ({int(npd)} days)")
    else:
        notice = 0.7

    # --- composite quality --------------------------------------------------
    quality = (
        0.26 * recency
        + 0.24 * response
        + 0.08 * (1.0 if otw else 0.4)
        + 0.10 * demand
        + 0.08 * interview
        + 0.04 * offer
        + 0.06 * pcs
        + 0.04 * verified
        + 0.06 * github
        + 0.04 * notice
    )
    quality = _clip01(quality)

    lo, hi = b.get("modifier_min", 0.60), b.get("modifier_max", 1.15)
    modifier = lo + (hi - lo) * quality

    # Hard availability dampener: strong-on-paper but clearly gone + silent.
    if recency < 0.3 and response < 0.15:
        modifier *= 0.85
        if "not actually reachable" not in con:
            con.append("low availability (stale + unresponsive)")

    return SignalResult(modifier=round(modifier, 4), quality=round(quality, 4),
                        facts=facts, positives=pos, concerns=con)
