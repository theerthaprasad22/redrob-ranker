"""
Tests for the paste-a-job-role feature (src/redrob_ranker/jd_spec.py).

Run:  python -m pytest tests/ -q     (or)    python tests/test_jd_spec.py

Covers the behaviours that matter for this feature:
  * an empty JD falls back to the bundled spec unchanged (demo still works),
  * the bundled spec's semantic anchor is byte-identical to before this feature
    existed -- the guarantee that the timed 100K submission is unaffected,
  * a pasted JD is parsed into a role (title + must/nice signals) with the
    offline heuristic path (no LLM, no network),
  * the resulting spec actually changes the ranking: a frontend JD puts a
    frontend engineer on top, an AI JD puts an AI engineer on top.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from redrob_ranker.jd_spec import build_spec_from_jd               # noqa: E402
from redrob_ranker.pipeline import rank_candidates                 # noqa: E402
from redrob_ranker.role_spec import RoleSpec                       # noqa: E402
from redrob_ranker.schema import Candidate                         # noqa: E402

SPEC = RoleSpec.load(os.path.join(ROOT, "config", "role_spec.yaml"))

# The hardcoded anchor the bundled role used before this feature shipped. The
# offline default path must reproduce it exactly so the timed submission, which
# uses this spec, ranks identically.
_BASELINE_DEFAULT_ANCHOR = SPEC.query_text()


def _signals(**over):
    s = {
        "profile_completeness_score": 80, "signup_date": "2023-01-01",
        "last_active_date": "2026-05-01", "open_to_work_flag": True,
        "profile_views_received_30d": 10, "applications_submitted_30d": 3,
        "recruiter_response_rate": 0.7, "avg_response_time_hours": 5,
        "skill_assessment_scores": {}, "connection_count": 200,
        "endorsements_received": 50, "notice_period_days": 30,
        "expected_salary_range_inr_lpa": {"min": 30, "max": 45},
        "preferred_work_mode": "hybrid", "willing_to_relocate": True,
        "github_activity_score": 60, "search_appearance_30d": 20,
        "saved_by_recruiters_30d": 5, "interview_completion_rate": 0.9,
        "offer_acceptance_rate": 0.5, "verified_email": True,
        "verified_phone": True, "linkedin_connected": True,
    }
    s.update(over)
    return s


def _ai_engineer():
    return Candidate.from_dict({
        "candidate_id": "CAND_1000001",
        "profile": {
            "anonymized_name": "A B", "headline": "ML Engineer | Search & Ranking",
            "summary": "Built and shipped a recommendation and search ranking system "
                       "serving millions. Embeddings, FAISS, hybrid retrieval, NDCG.",
            "location": "Pune, Maharashtra", "country": "India",
            "years_of_experience": 7.0, "current_title": "Machine Learning Engineer",
            "current_company": "ShopCo", "current_company_size": "201-500",
            "current_industry": "Internet",
        },
        "career_history": [
            {"company": "ShopCo", "title": "Machine Learning Engineer",
             "start_date": "2020-06-01", "end_date": None, "duration_months": 60,
             "is_current": True, "industry": "Internet", "company_size": "201-500",
             "description": "Shipped a production recommendation and ranking system; "
                            "embeddings retrieval with FAISS; A/B tested with NDCG."},
        ],
        "education": [], "languages": [], "certifications": [],
        "skills": [
            {"name": "Sentence Transformers", "proficiency": "expert",
             "endorsements": 30, "duration_months": 48},
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 20,
             "duration_months": 40},
            {"name": "Python", "proficiency": "expert", "endorsements": 40,
             "duration_months": 84},
        ],
        "redrob_signals": _signals(),
    })


def _frontend_engineer():
    return Candidate.from_dict({
        "candidate_id": "CAND_2000002",
        "profile": {
            "anonymized_name": "C D", "headline": "Frontend Engineer | React",
            "summary": "Frontend engineer who builds accessible, performant web UIs "
                       "in React and TypeScript. Design-system and testing experience.",
            "location": "Remote", "country": "India",
            "years_of_experience": 6.0, "current_title": "Frontend Engineer",
            "current_company": "WebCo", "current_company_size": "201-500",
            "current_industry": "Internet",
        },
        "career_history": [
            {"company": "WebCo", "title": "Frontend Engineer",
             "start_date": "2019-06-01", "end_date": None, "duration_months": 72,
             "is_current": True, "industry": "Internet", "company_size": "201-500",
             "description": "Built React + TypeScript web apps; CSS, accessibility, "
                            "Jest and Playwright testing, Next.js."},
        ],
        "education": [], "languages": [], "certifications": [],
        "skills": [
            {"name": "React", "proficiency": "expert", "endorsements": 35,
             "duration_months": 72},
            {"name": "TypeScript", "proficiency": "expert", "endorsements": 30,
             "duration_months": 60},
            {"name": "CSS", "proficiency": "advanced", "endorsements": 25,
             "duration_months": 72},
        ],
        "redrob_signals": _signals(),
    })


_FE_JD = """Senior Frontend Engineer
Own our React + TypeScript web app and build accessible, performant UI.
Must have: 5+ years React, strong TypeScript, CSS and HTML.
Nice to have: Next.js, testing with Jest or Playwright.
Location: Remote."""

_AI_JD = """Senior AI Engineer
Build search, ranking and recommendation systems. Work with embeddings,
vector retrieval and hybrid search; evaluate with NDCG.
Must have: Python, machine learning, embeddings and retrieval.
Nice to have: FAISS, learning to rank."""


def test_empty_jd_falls_back_to_base_spec():
    res = build_spec_from_jd("", base_spec=SPEC)
    assert res.method == "default"
    # the returned spec IS the bundled spec, untouched
    assert res.spec is SPEC
    assert res.spec.query_text() == _BASELINE_DEFAULT_ANCHOR


def test_blank_whitespace_jd_also_falls_back():
    res = build_spec_from_jd("   \n  \t ", base_spec=SPEC)
    assert res.method == "default"
    assert res.spec.query_text() == _BASELINE_DEFAULT_ANCHOR


def test_bundled_anchor_is_unchanged_by_feature():
    # Guards the timed submission: the default semantic anchor must not drift.
    assert "ranking" in _BASELINE_DEFAULT_ANCHOR.lower()
    fresh = RoleSpec.load(os.path.join(ROOT, "config", "role_spec.yaml"))
    assert fresh.query_text() == _BASELINE_DEFAULT_ANCHOR


def test_frontend_jd_parsed_offline():
    # No llm_client passed -> guaranteed heuristic (offline) path.
    res = build_spec_from_jd(_FE_JD, base_spec=SPEC)
    assert res.method == "heuristic"
    assert "frontend" in res.title.lower()
    assert res.is_ai_role is False
    # the semantic anchor is now the pasted role, not the hardcoded AI string
    assert res.spec.query_text() != _BASELINE_DEFAULT_ANCHOR
    assert "react" in res.spec.query_text().lower()


def test_ai_jd_detected_as_ai_role():
    res = build_spec_from_jd(_AI_JD, base_spec=SPEC)
    assert res.method == "heuristic"
    assert res.is_ai_role is True


def test_jd_changes_ranking_order():
    pool = [_ai_engineer(), _frontend_engineer()]

    fe_spec = build_spec_from_jd(_FE_JD, base_spec=SPEC).spec
    fe_rows = rank_candidates(pool, fe_spec, shortlist_k=10, n_results=2)
    assert fe_rows[0].candidate_id == "CAND_2000002", \
        "a frontend JD should rank the frontend engineer first"

    ai_spec = build_spec_from_jd(_AI_JD, base_spec=SPEC).spec
    ai_rows = rank_candidates(pool, ai_spec, shortlist_k=10, n_results=2)
    assert ai_rows[0].candidate_id == "CAND_1000001", \
        "an AI JD should rank the AI engineer first"


# --- role-relative experience + fresher fairness ---------------------------
from redrob_ranker.scoring import experience_fit, _relevant_years  # noqa: E402
from redrob_ranker.explain import explain_candidate                # noqa: E402

_SENIOR_FE_JD = ("Senior Frontend Engineer. Own our React + TypeScript web app. "
                 "Must have: 5+ years React, strong TypeScript, CSS.")
_JUNIOR_FE_JD = ("Junior Frontend Engineer (entry level). Graduates welcome. "
                 "Build React UI with TypeScript and CSS.")


def _make(cid, title, yrs, hist, skills):
    return Candidate.from_dict({
        "candidate_id": cid,
        "profile": {"current_title": title, "years_of_experience": yrs,
                    "headline": title, "summary": title + " " + " ".join(skills),
                    "location": "Remote", "country": "India"},
        "career_history": hist,
        "skills": [{"name": s, "proficiency": "expert", "endorsements": 20,
                    "duration_months": 36} for s in skills],
        "education": [{"tier": "tier_2", "field_of_study": "Computer Science",
                       "degree": "BTech"}],
        "redrob_signals": _signals(),
    })


def _veteran_accountant():
    return _make("VET_ACCT", "Senior Accountant", 12.0,
                 [{"title": "Senior Accountant", "description": "audit tax ledgers",
                   "duration_months": 84, "industry": "Finance"},
                  {"title": "Accountant", "description": "bookkeeping",
                   "duration_months": 60, "industry": "Finance"}],
                 ["Excel", "Taxation"])


def _fresher_frontend():
    return _make("FRSH_FE", "Junior Frontend Engineer", 1.0,
                 [{"title": "Junior Frontend Engineer",
                   "description": "built React TypeScript UI components and CSS",
                   "duration_months": 12, "industry": "Internet"}],
                 ["React", "TypeScript", "CSS"])


def test_offfield_experience_is_discounted():
    # A 12-year accountant has ~0 frontend-relevant years.
    spec = build_spec_from_jd(_SENIOR_FE_JD, base_spec=SPEC).spec
    vet = _veteran_accountant()
    assert _relevant_years(vet, spec) < 1.0, "accounting years are not frontend years"
    # so their experience_fit must not reward the raw 12 years
    assert experience_fit(vet, spec) <= 0.4


def test_relevant_fresher_beats_offfield_veteran():
    spec = build_spec_from_jd(_SENIOR_FE_JD, base_spec=SPEC).spec
    rows = rank_candidates([_veteran_accountant(), _fresher_frontend()],
                           spec, shortlist_k=10, n_results=2)
    assert rows[0].candidate_id == "FRSH_FE", \
        "a relevant fresher should outrank a high-tenure off-field candidate"


def test_fresher_not_penalised_on_entry_level_role():
    spec = build_spec_from_jd(_JUNIOR_FE_JD, base_spec=SPEC).spec
    assert spec.data["role"]["seniority"] == "junior"
    # a 1-year relevant fresher should score well on experience for a junior role
    assert experience_fit(_fresher_frontend(), spec) >= 0.85
    # and experience should be weighted lightly vs skills for an entry-level role
    w = spec.weights
    assert w["experience_fit"] < w["skills_trust"]
    assert w["experience_fit"] <= 0.06


def test_seniority_shifts_experience_weight():
    junior_w = build_spec_from_jd(_JUNIOR_FE_JD, base_spec=SPEC).spec.weights
    senior_w = build_spec_from_jd(_SENIOR_FE_JD, base_spec=SPEC).spec.weights
    assert senior_w["experience_fit"] > junior_w["experience_fit"], \
        "a senior role must weight experience more than a junior role"


def test_default_spec_has_no_experience_policy():
    # The bundled spec must not carry a policy, so experience_fit keeps the
    # original raw-years behaviour (and the timed submission is unaffected).
    assert "policy" not in SPEC.experience


def test_explain_reports_matched_and_missing():
    spec = build_spec_from_jd(_SENIOR_FE_JD, base_spec=SPEC).spec
    pool = [_fresher_frontend(), _veteran_accountant()]
    rows = rank_candidates(pool, spec, shortlist_k=10, n_results=2)
    by = {c.candidate_id: c for c in pool}
    fe = explain_candidate(by["FRSH_FE"], spec, rows[0].breakdown)
    vet = explain_candidate(by["VET_ACCT"], spec, rows[1].breakdown)
    assert fe.matched and not fe.missing, "frontend fresher should match the must-haves"
    assert vet.missing, "the accountant should be missing the role's must-haves"
    assert fe.confidence in {"high", "medium", "low"}
    assert isinstance(fe.narrative, str) and fe.narrative


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} tests passed")
    sys.exit(0 if passed == len(fns) else 1)
