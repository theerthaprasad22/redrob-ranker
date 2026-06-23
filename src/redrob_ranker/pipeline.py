"""
pipeline.py — the ranking funnel.

    100K candidates
        │  parse (schema.py)
        ▼
    integrity / honeypot filter (integrity.py)      ← drop impossible profiles
        │
        ▼
    cheap prefilter score (scoring.py)              ← 100K → top-K (recall)
        │
        ▼
    semantic encode shortlist (embeddings.py)       ← LSA default / ST optional
        │
        ▼
    full multi-dimensional scoring (scoring.py)     ← the recruiter judgment
        │
        ▼
    behavioral-signal modifier (signals.py)         ← availability / quality
        │
        ▼
    final = base_fit × modifier × integrity_soft
        │  sort desc, normalise, tie-break by candidate_id
        ▼
    top-100  →  grounded reasoning (reasoning.py)

Everything here is CPU-only, in-memory, and (with the default LSA encoder)
needs no network and no model download — so it satisfies the 5-min / 16 GB /
CPU-only / no-network ranking constraints.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

from .embeddings import cosine, make_encoder
from .integrity import assess_integrity, estimate_reference_date
from .reasoning import generate_reasoning
from .role_spec import RoleSpec
from .schema import Candidate
from .scoring import ScoreBreakdown, cheap_prefilter_score, score_candidate
from .signals import behavioral_modifier


def _semantic_text(c: Candidate) -> str:
    """Compact text used for embedding (focused, not the whole blob)."""
    titles = " ".join(str(r.get("title", "")) for r in c.career_history[:3] if isinstance(r, dict))
    descs = " ".join(str(r.get("description", "")) for r in c.career_history[:2] if isinstance(r, dict))
    skills = " ".join(sorted(c.skill_names_lc))
    return f"{c.headline}. {c.current_title}. {c.summary} {titles}. {descs} Skills: {skills}"[:2000]


@dataclass
class RankedRow:
    candidate_id: str
    rank: int
    score: float
    reasoning: str
    final_raw: float = 0.0
    breakdown: Optional[ScoreBreakdown] = None
    signal_facts: dict[str, Any] = field(default_factory=dict)


def rank_candidates(
    candidates: list[Candidate],
    spec: RoleSpec,
    *,
    encoder_kind: str = "lsa",
    shortlist_k: int = 6000,
    n_results: int = 100,
    llm_client=None,
    reference_date: Optional[date] = None,
    verbose: bool = False,
) -> list[RankedRow]:
    t0 = time.time()

    def log(msg: str):
        if verbose:
            print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    ref = reference_date or estimate_reference_date(candidates)
    log(f"loaded {len(candidates)} candidates; reference date = {ref}")

    # --- Stage 1: integrity / honeypot filter -------------------------------
    clean: list[tuple[Candidate, float]] = []   # (candidate, integrity_soft_penalty)
    n_honeypot = 0
    for c in candidates:
        integ = assess_integrity(c, ref)
        if integ.is_honeypot:
            n_honeypot += 1
            continue
        clean.append((c, integ.soft_penalty))
    log(f"integrity filter: dropped {n_honeypot} impossible/honeypot profiles; "
        f"{len(clean)} remain")

    # --- Stage 2: cheap prefilter (recall funnel) ---------------------------
    scored_cheap = [(c, soft, cheap_prefilter_score(c, spec)) for (c, soft) in clean]
    scored_cheap.sort(key=lambda x: x[2], reverse=True)
    shortlist = scored_cheap[: max(shortlist_k, n_results * 5)]
    log(f"prefilter: kept top {len(shortlist)} for deep scoring")

    # --- Stage 3: semantic encoding of the shortlist ------------------------
    texts = [_semantic_text(c) for (c, _, _) in shortlist]
    encoder = make_encoder(encoder_kind)
    encoder.fit(texts + [spec.query_text()])
    cand_vecs = encoder.encode(texts)
    query_vec = encoder.encode([spec.query_text()])[0]
    sem = cosine(cand_vecs, query_vec)   # [0,1] per candidate
    log(f"semantic encode done ({encoder_kind}, dim≈{cand_vecs.shape[1]})")

    # --- Stage 4: full scoring + behavioral modifier ------------------------
    results: list[RankedRow] = []
    for i, (c, soft, _) in enumerate(shortlist):
        bd = score_candidate(c, spec, semantic_fit=float(sem[i]))
        sig = behavioral_modifier(c, spec.behavioral, ref)
        final = bd.base_fit * sig.modifier * soft
        results.append(RankedRow(
            candidate_id=c.candidate_id,
            rank=0,
            score=0.0,
            reasoning="",
            final_raw=final,
            breakdown=bd,
            signal_facts=sig.facts,
        ))
    log("full scoring + behavioral modifier done")

    # --- Stage 5: sort, take top-N, normalise, enforce monotonic + tiebreak -
    max_final = max((r.final_raw for r in results), default=1.0) or 1.0
    for r in results:
        r.score = r.final_raw / max_final     # normalise to [0,1]
    # round to 6 dp, then sort by (score desc, candidate_id asc) so the
    # validator's "ties broken by candidate_id ascending" rule always holds.
    for r in results:
        r.score = round(r.score, 6)
    results.sort(key=lambda r: (-r.score, r.candidate_id))
    top = results[:n_results]
    for idx, r in enumerate(top):
        r.rank = idx + 1
    log(f"selected top {len(top)}")

    # --- Stage 6: grounded reasoning for the top-N --------------------------
    # Re-derive the per-candidate objects we need (breakdown + signal facts are
    # already attached). We need the Candidate object for reasoning facts.
    cand_by_id = {c.candidate_id: c for (c, _, _) in shortlist}
    for r in top:
        c = cand_by_id.get(r.candidate_id)
        if c is None or r.breakdown is None:
            r.reasoning = "Included in shortlist."
            continue
        # rebuild a minimal SignalResult-like for reasoning (facts/positives/concerns)
        sig = behavioral_modifier(c, spec.behavioral, ref)
        r.reasoning = generate_reasoning(c, r.breakdown, sig, r.rank, client=llm_client)
    log(f"reasoning generated; total pipeline time {time.time() - t0:.1f}s")

    return top
