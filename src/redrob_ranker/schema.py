"""
schema.py — candidate data model and (streaming) loader.

The dataset is 100K records (~465 MB uncompressed JSONL). We parse each line
into a compact `Candidate` with a few precomputed convenience fields, then keep
the lightweight objects in memory (well under the 16 GB budget). Parsing is
defensive: any malformed/partial record is captured as `parse_ok = False` so the
pipeline can route it to the bottom rather than crash mid-run.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterator


def _to_date(s: Any) -> date | None:
    if not s or not isinstance(s, str):
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


@dataclass(slots=True)
class Candidate:
    """Normalised view of one candidate record."""

    candidate_id: str
    raw: dict[str, Any]

    # profile
    headline: str = ""
    summary: str = ""
    current_title: str = ""
    current_company: str = ""
    current_industry: str = ""
    current_company_size: str = ""
    location: str = ""
    country: str = ""
    years_of_experience: float = 0.0

    # structured sub-records (kept as-is)
    career_history: list[dict[str, Any]] = field(default_factory=list)
    education: list[dict[str, Any]] = field(default_factory=list)
    skills: list[dict[str, Any]] = field(default_factory=list)
    certifications: list[dict[str, Any]] = field(default_factory=list)
    languages: list[dict[str, Any]] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)

    # derived convenience fields
    skill_names_lc: set[str] = field(default_factory=set)
    blob_lc: str = ""          # lower-cased concatenation of all text (for lexical/keyword scan)
    career_text_lc: str = ""   # lower-cased career descriptions + titles only
    parse_ok: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Candidate":
        try:
            prof = d.get("profile", {}) or {}
            sig = d.get("redrob_signals", {}) or {}
            ch = d.get("career_history", []) or []
            skills = d.get("skills", []) or []

            skill_names = {
                str(s.get("name", "")).strip().lower()
                for s in skills
                if isinstance(s, dict) and s.get("name")
            }

            # Career text = titles + role descriptions (this is where "plain
            # language" Tier-5 evidence lives, e.g. "built a recommendation
            # system serving millions" with no buzzwords in the skills list).
            career_chunks: list[str] = []
            for r in ch:
                if not isinstance(r, dict):
                    continue
                career_chunks.append(str(r.get("title", "")))
                career_chunks.append(str(r.get("company", "")))
                career_chunks.append(str(r.get("industry", "")))
                career_chunks.append(str(r.get("description", "")))
            career_text = " ".join(career_chunks).lower()

            # Full blob for lexical retrieval / keyword scanning.
            blob_parts = [
                str(prof.get("headline", "")),
                str(prof.get("summary", "")),
                str(prof.get("current_title", "")),
                str(prof.get("current_industry", "")),
                " ".join(sorted(skill_names)),
                " ".join(str(c.get("name", "")) for c in d.get("certifications", []) or []),
                career_text,
            ]
            blob = " ".join(blob_parts).lower()

            return cls(
                candidate_id=str(d.get("candidate_id", "")),
                raw=d,
                headline=str(prof.get("headline", "")),
                summary=str(prof.get("summary", "")),
                current_title=str(prof.get("current_title", "")),
                current_company=str(prof.get("current_company", "")),
                current_industry=str(prof.get("current_industry", "")),
                current_company_size=str(prof.get("current_company_size", "")),
                location=str(prof.get("location", "")),
                country=str(prof.get("country", "")),
                years_of_experience=float(prof.get("years_of_experience", 0) or 0),
                career_history=ch,
                education=d.get("education", []) or [],
                skills=skills,
                certifications=d.get("certifications", []) or [],
                languages=d.get("languages", []) or [],
                signals=sig,
                skill_names_lc=skill_names,
                blob_lc=blob,
                career_text_lc=career_text,
                parse_ok=True,
            )
        except Exception:
            return cls(
                candidate_id=str(d.get("candidate_id", "")) if isinstance(d, dict) else "",
                raw=d if isinstance(d, dict) else {},
                parse_ok=False,
            )

    # ---- small helpers used across scoring modules -------------------------

    def career_start_date(self) -> date | None:
        starts = [_to_date(r.get("start_date")) for r in self.career_history if isinstance(r, dict)]
        starts = [s for s in starts if s]
        return min(starts) if starts else None

    def total_career_months(self) -> int:
        tot = 0
        for r in self.career_history:
            if isinstance(r, dict):
                try:
                    tot += int(r.get("duration_months", 0) or 0)
                except (TypeError, ValueError):
                    pass
        return tot

    def last_active(self) -> date | None:
        return _to_date(self.signals.get("last_active_date"))


def _open_any(path: str):
    """Open .jsonl or .jsonl.gz transparently in text mode."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def stream_candidates(path: str) -> Iterator[Candidate]:
    """Yield Candidate objects one line at a time (constant memory)."""
    with _open_any(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield Candidate.from_dict(d)


def load_candidates(path: str) -> list[Candidate]:
    """Load all candidates into a list (used by the ranker)."""
    return list(stream_candidates(path))
