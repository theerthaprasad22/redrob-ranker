"""
scoring.py — the recruiter judgment.

Turns a candidate + RoleSpec into a transparent `ScoreBreakdown`. The design
principle (straight from the JD's participant note): career/role evidence and
semantic understanding dominate; raw skill-list overlap is deliberately NOT the
biggest term and is trust-weighted so keyword stuffing is punished, not rewarded.

Two entry points:
  * cheap_prefilter_score(): O(1)-ish per candidate, used to shrink 100K -> top-K
    before the expensive semantic + full-scoring stage.
  * score_candidate(): the full multi-component breakdown for the shortlist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .role_spec import RoleSpec
from .schema import Candidate

_PROF_BASE = {"beginner": 0.30, "intermediate": 0.55, "advanced": 0.80, "expert": 1.00}


def _hits(text: str, terms: list[str]) -> int:
    return sum(1 for t in terms if t in text)


def _any(text: str, terms: list[str]) -> bool:
    return any(t in text for t in terms)


# ---------------------------------------------------------------------------
# Title scoring
# ---------------------------------------------------------------------------
def _title_score(title: str, spec: RoleSpec) -> float:
    t = title.lower()
    ce = spec.career_evidence
    # Off-role titles (the keyword-stuffer signature) crush the score even if
    # the skills list is full of AI terms.
    if _any(t, ce.get("offrole_titles", [])):
        # but if an engineering word also appears (e.g. "ML Engineer, Marketing
        # Analytics") give a little credit
        if any(k in t for k in ("ai engineer", "ml engineer", "machine learning",
                                 "data scientist", "nlp", "software engineer")):
            return 0.35
        return 0.08
    if any(k in t for k in ("ai engineer", "machine learning engineer", "ml engineer",
                            "machine learning", "applied scientist", "applied ml",
                            "nlp engineer", "ml scientist", "deep learning engineer")):
        return 1.0
    if any(k in t for k in ("data scientist", "research engineer", "search engineer",
                            "ai/ml", "ai ", "ml ")):
        return 0.85
    if any(k in t for k in ("software engineer", "backend engineer", "platform engineer",
                            "sde", "staff engineer", "principal engineer",
                            "data engineer", "computer scientist", "engineer")):
        return 0.55
    return 0.25


def role_title_fit(c: Candidate, spec: RoleSpec) -> float:
    cur = _title_score(c.current_title, spec)
    recent = 0.0
    for r in c.career_history[:3]:
        if isinstance(r, dict):
            recent = max(recent, _title_score(str(r.get("title", "")), spec))
    return round(0.7 * cur + 0.3 * recent, 4)


# ---------------------------------------------------------------------------
# Career evidence (shipped ranking/search/recsys at a product company)
# ---------------------------------------------------------------------------
def _relevant_nouns(spec: RoleSpec) -> list[str]:
    out: list[str] = []
    for key in ("ranking_search_recsys", "embeddings_retrieval", "vector_db_hybrid_search"):
        out.extend(spec.must_have.get(key, {}).get("terms", []))
    return out


def career_evidence_fit(c: Candidate, spec: RoleSpec) -> tuple[float, bool]:
    ce = spec.career_evidence
    txt = c.career_text_lc
    nouns = _relevant_nouns(spec)
    verbs = ce.get("shipped_systems_terms", [])

    noun_hits = _hits(txt, nouns)
    has_verb = _any(txt, verbs)

    if noun_hits >= 2 and has_verb:
        base = 1.0
    elif noun_hits >= 1 and has_verb:
        base = 0.75
    elif noun_hits >= 1:
        base = 0.5
    else:
        base = 0.05

    # product vs services
    services = spec.disqualifiers.get("services_only", {})
    svc_companies = services.get("companies", [])
    svc_inds = services.get("services_industries", [])
    is_services_only = _is_services_only(c, svc_companies, svc_inds)

    if _any(txt, ce.get("product_signal_terms", [])):
        base = min(1.0, base + 0.15)
    if is_services_only:
        base *= 0.6

    return round(base, 4), is_services_only


def _is_services_only(c: Candidate, svc_companies: list[str], svc_inds: list[str]) -> bool:
    """True only if EVERY role is at a services firm (JD allows services-now +
    product-before)."""
    roles = [r for r in c.career_history if isinstance(r, dict)]
    if not roles:
        return False
    for r in roles:
        comp = str(r.get("company", "")).lower()
        ind = str(r.get("industry", "")).lower()
        is_svc = any(s in comp for s in svc_companies) or any(s in ind for s in svc_inds)
        if not is_svc:
            return False  # found a non-services role -> not services-only
    return True


# ---------------------------------------------------------------------------
# Skills trust (overlap weighted by proficiency / endorsements / duration /
# Redrob assessment scores) — the anti-keyword-stuffing component.
# ---------------------------------------------------------------------------
def _skill_trust(skill: dict[str, Any], assessments: dict[str, Any]) -> float:
    prof = _PROF_BASE.get(str(skill.get("proficiency", "")).lower(), 0.4)
    dur = int(skill.get("duration_months", 0) or 0)
    endo = int(skill.get("endorsements", 0) or 0)
    dur_f = 0.30 + 0.70 * min(1.0, dur / 24.0)
    endo_f = 0.60 + 0.40 * min(1.0, endo / 20.0)
    trust = prof * dur_f * endo_f
    name = str(skill.get("name", "")).lower()
    # assessment confirms or undercuts the claim
    for k, v in assessments.items():
        if str(k).lower() == name:
            try:
                trust *= 0.5 + 0.5 * (float(v) / 100.0)
            except (TypeError, ValueError):
                pass
            break
    return max(0.0, min(1.0, trust))


def skills_trust_fit(c: Candidate, spec: RoleSpec) -> tuple[float, list[str]]:
    assessments = c.signals.get("skill_assessment_scores", {}) or {}
    skills = [s for s in c.skills if isinstance(s, dict)]

    def concept_cov(concepts: dict[str, Any]) -> tuple[float, float, list[str]]:
        total_w, got = 0.0, 0.0
        matched: list[str] = []
        for cname, cfg in concepts.items():
            w = float(cfg.get("weight", 1.0))
            total_w += w
            terms = cfg.get("terms", [])
            best = 0.0
            best_name = ""
            for s in skills:
                nm = str(s.get("name", "")).lower()
                if any(t in nm or nm in t for t in terms):
                    tr = _skill_trust(s, assessments)
                    if tr > best:
                        best, best_name = tr, str(s.get("name", ""))
            if best == 0.0 and _any(c.career_text_lc, terms):
                best = 0.4  # plain-language career evidence, no skill listed
            if best > 0.0:
                got += w * best
                if best_name:
                    matched.append(best_name)
            # require minimal trust to count as "matched" for reasoning
        return (got / total_w if total_w else 0.0), total_w, matched

    must_cov, _, must_matched = concept_cov(spec.must_have)
    nice_cov, _, _ = concept_cov(spec.nice_to_have)
    score = min(1.0, must_cov + 0.15 * nice_cov)
    # dedupe matched skills while preserving order (same skill can be the best
    # match for two concepts)
    seen: set[str] = set()
    deduped: list[str] = []
    for m in must_matched:
        k = m.lower()
        if k not in seen:
            seen.add(k)
            deduped.append(m)
    return round(score, 4), deduped


# ---------------------------------------------------------------------------
# Experience fit
# ---------------------------------------------------------------------------
def experience_fit(c: Candidate, spec: RoleSpec) -> float:
    e = spec.experience
    y = c.years_of_experience
    lo, hi = e["ideal_low"], e["ideal_high"]
    mn, mx = e["min_years"], e["max_years"]
    fl, ce = e["soft_floor"], e["soft_ceiling"]
    if lo <= y <= hi:
        return 1.0
    if mn <= y < lo:
        return 0.85 + 0.15 * (y - mn) / max(1e-9, (lo - mn))
    if hi < y <= mx:
        return 0.85 + 0.15 * (mx - y) / max(1e-9, (mx - hi))
    if fl <= y < mn:
        return 0.45 + 0.40 * (y - fl) / max(1e-9, (mn - fl))
    if mx < y <= ce:
        return 0.45 + 0.40 * (ce - y) / max(1e-9, (ce - mx))
    return 0.20


# ---------------------------------------------------------------------------
# Domain fit (NLP/IR positive vs CV/speech/robotics negative)
# ---------------------------------------------------------------------------
def domain_fit(c: Candidate, spec: RoleSpec) -> tuple[float, bool]:
    pos = spec.domain.get("positive_terms", [])
    neg = spec.domain.get("negative_terms", [])
    has_pos = _any(c.blob_lc, pos)
    neg_hits = _hits(c.blob_lc, neg)
    score = 0.6
    if has_pos:
        score = min(1.0, 0.6 + 0.4)
    wrong_domain = False
    if neg_hits >= 2 and not has_pos:
        score = 0.2
        wrong_domain = True
    elif neg_hits >= 1 and not has_pos:
        score = 0.4
    return round(score, 4), wrong_domain


# ---------------------------------------------------------------------------
# Location fit
# ---------------------------------------------------------------------------
def location_fit(c: Candidate, spec: RoleSpec) -> float:
    loc = (c.location or "").lower()
    country = (c.country or "").lower()
    L = spec.location
    if _any(loc, L.get("best", [])):
        return 1.0
    if _any(loc, L.get("good", [])):
        return 0.85
    if any(ok in country for ok in L.get("ok_country", [])) or "india" in loc:
        base = 0.55
        if c.signals.get("willing_to_relocate"):
            base = 0.70
        return base
    # outside India: no visa sponsorship
    base = 0.20
    if c.signals.get("willing_to_relocate"):
        base = 0.30
    return base


# ---------------------------------------------------------------------------
# Education fit (minor)
# ---------------------------------------------------------------------------
def education_fit(c: Candidate) -> float:
    if not c.education:
        return 0.5
    tier_map = {"tier_1": 1.0, "tier_2": 0.8, "tier_3": 0.6, "tier_4": 0.45, "unknown": 0.5}
    best = 0.0
    for e in c.education:
        if not isinstance(e, dict):
            continue
        t = tier_map.get(str(e.get("tier", "unknown")).lower(), 0.5)
        field_l = str(e.get("field_of_study", "")).lower()
        if any(k in field_l for k in ("computer", "data", "machine", "artificial",
                                      "statistics", "mathematics", "software")):
            t = min(1.0, t + 0.1)
        best = max(best, t)
    return round(best, 4)


# ---------------------------------------------------------------------------
# Disqualifier factor (multiplicative, floored)
# ---------------------------------------------------------------------------
def disqualifier_factor(c: Candidate, spec: RoleSpec,
                        is_services_only: bool, wrong_domain: bool) -> tuple[float, list[str]]:
    dq = spec.disqualifiers
    factor = 1.0
    notes: list[str] = []
    blob = c.blob_lc

    if is_services_only:
        factor *= dq.get("services_only", {}).get("penalty", 0.45)
        notes.append("career entirely at IT-services/consulting firms")

    # research-only without production
    ro = dq.get("research_only", {})
    if _any(blob, ro.get("terms", [])) and not _any(blob, ro.get("production_terms", [])):
        factor *= ro.get("penalty", 0.55)
        notes.append("research/academic background with no production-deployment signal")

    # recent-framework-only (LangChain wrapper) without prior ML
    fo = dq.get("framework_only_recent", {})
    if _any(blob, fo.get("framework_terms", [])) and not _any(blob, fo.get("legacy_ml_terms", [])):
        factor *= fo.get("penalty", 0.6)
        notes.append("AI experience appears limited to recent LLM-wrapper work")

    # title-chaser (many short stints)
    tc = dq.get("title_chaser", {})
    roles = [r for r in c.career_history if isinstance(r, dict)]
    if len(roles) >= tc.get("min_jobs_for_flag", 3):
        durs = [int(r.get("duration_months", 0) or 0) for r in roles]
        avg = sum(durs) / len(durs) if durs else 0
        if 0 < avg < tc.get("max_avg_tenure_months", 19):
            factor *= tc.get("penalty", 0.7)
            notes.append(f"short average tenure (~{avg:.0f} months/role)")

    if wrong_domain:
        factor *= dq.get("wrong_domain", {}).get("penalty", 0.55)
        notes.append("primary expertise is CV/speech/robotics, not NLP/IR")

    return max(0.15, round(factor, 4)), notes


# ---------------------------------------------------------------------------
# Full breakdown
# ---------------------------------------------------------------------------
@dataclass
class ScoreBreakdown:
    components: dict[str, float] = field(default_factory=dict)
    disqualifier_factor: float = 1.0
    base_fit: float = 0.0           # weighted components * disqualifier_factor
    matched_skills: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)


def score_candidate(c: Candidate, spec: RoleSpec, semantic_fit: float) -> ScoreBreakdown:
    rtf = role_title_fit(c, spec)
    cev, services_only = career_evidence_fit(c, spec)
    skt, matched = skills_trust_fit(c, spec)
    exp = experience_fit(c, spec)
    dom, wrong_domain = domain_fit(c, spec)
    locf = location_fit(c, spec)
    edu = education_fit(c)

    comps = {
        "semantic_fit": round(semantic_fit, 4),
        "role_title_fit": rtf,
        "career_evidence": cev,
        "skills_trust": skt,
        "experience_fit": round(exp, 4),
        "domain_fit": dom,
        "location_fit": round(locf, 4),
        "education_fit": edu,
    }
    w = spec.weights
    weighted = sum(comps[k] * float(w.get(k, 0.0)) for k in comps)

    dqf, dq_notes = disqualifier_factor(c, spec, services_only, wrong_domain)
    base_fit = weighted * dqf

    # collect human-readable concerns for the reasoning module
    concerns = list(dq_notes)
    if locf <= 0.30:
        concerns.append("located outside India (no visa sponsorship)")
    if comps["role_title_fit"] <= 0.2:
        concerns.append(f"current title ('{c.current_title}') is off-target for the role")

    return ScoreBreakdown(
        components=comps,
        disqualifier_factor=dqf,
        base_fit=round(base_fit, 6),
        matched_skills=matched,
        concerns=concerns,
    )


# ---------------------------------------------------------------------------
# Cheap prefilter (100K -> top-K) — no embeddings, minimal substring work.
# ---------------------------------------------------------------------------
_ANCHOR_SKILL_TOKENS = {
    "machine learning", "deep learning", "nlp", "information retrieval", "retrieval",
    "embeddings", "sentence-transformers", "transformers", "search", "ranking",
    "recommendation", "recommender", "pytorch", "tensorflow", "faiss", "elasticsearch",
    "pinecone", "weaviate", "qdrant", "vector", "llm", "fine-tuning", "rag",
    "learning to rank", "xgboost", "semantic search", "python",
}


def cheap_prefilter_score(c: Candidate, spec: RoleSpec) -> float:
    """Fast, recall-oriented score for the funnel's first stage."""
    if not c.parse_ok:
        return -1.0
    # title signal (cheap: current title only)
    t = c.current_title.lower()
    title = _title_score(c.current_title, spec)
    # skill-token overlap (set intersection — very fast)
    overlap = 0
    for tok in _ANCHOR_SKILL_TOKENS:
        if tok in c.skill_names_lc or tok in c.career_text_lc:
            overlap += 1
    overlap_score = min(1.0, overlap / 6.0)
    # relevant career nouns (short text scan)
    nouns = _relevant_nouns(spec)
    noun = 1.0 if _any(c.career_text_lc, nouns) else 0.0
    # experience in/near band
    exp = experience_fit(c, spec)
    # combine (title + evidence dominate so off-role keyword stuffers still
    # surface for proper scoring but pure-irrelevant roles are dropped)
    return 0.4 * title + 0.3 * overlap_score + 0.2 * noun + 0.1 * exp
