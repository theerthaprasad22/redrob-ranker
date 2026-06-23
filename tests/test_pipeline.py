"""
Tests for the Redrob ranker.

Run:  python -m pytest tests/ -q     (or)    python tests/test_pipeline.py

Covers the behaviours that matter for the challenge:
  * honeypots with the documented signatures are detected,
  * a real candidate is NOT flagged as a honeypot,
  * keyword-stuffers (off-role title + many low-trust skills) rank below real
    engineers,
  * output is format-valid (unique ranks 1..N, scores non-increasing, ties
    broken by candidate_id ascending).
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from redrob_ranker.integrity import assess_integrity            # noqa: E402
from redrob_ranker.pipeline import rank_candidates              # noqa: E402
from redrob_ranker.role_spec import RoleSpec                    # noqa: E402
from redrob_ranker.schema import Candidate                      # noqa: E402

SPEC = RoleSpec.load(os.path.join(ROOT, "config", "role_spec.yaml"))


def _base_signals(**over):
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


def _real_engineer():
    return Candidate.from_dict({
        "candidate_id": "CAND_1000001",
        "profile": {
            "anonymized_name": "A B", "headline": "ML Engineer | Search & Ranking",
            "summary": "Built and shipped a recommendation and search ranking system "
                       "serving millions of users at a product company. Embeddings, "
                       "FAISS, hybrid retrieval, NDCG-based evaluation.",
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
                            "embeddings retrieval with FAISS; A/B tested with NDCG/MAP."},
            {"company": "DataCo", "title": "Data Scientist",
             "start_date": "2017-06-01", "end_date": "2020-05-01", "duration_months": 36,
             "is_current": False, "industry": "Software", "company_size": "51-200",
             "description": "Search relevance and learning to rank."},
        ],
        "education": [{"institution": "IIT", "degree": "BTech",
                       "field_of_study": "Computer Science", "start_year": 2013,
                       "end_year": 2017, "grade": "8.5", "tier": "tier_1"}],
        "skills": [
            {"name": "Sentence Transformers", "proficiency": "expert",
             "endorsements": 30, "duration_months": 48},
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 20,
             "duration_months": 40},
            {"name": "Learning to Rank", "proficiency": "advanced",
             "endorsements": 15, "duration_months": 36},
            {"name": "Python", "proficiency": "expert", "endorsements": 40,
             "duration_months": 84},
        ],
        "certifications": [], "languages": [],
        "redrob_signals": _base_signals(),
    })


def _keyword_stuffer():
    # off-role title, AI skills with LOW trust (short duration, few endorsements)
    # -- a "lazy stuffer", not an impossible profile, so it should rank LOW
    # rather than be removed as a honeypot.
    skills = [{"name": n, "proficiency": "expert", "endorsements": 2,
               "duration_months": 2}
              for n in ["NLP", "RAG", "Pinecone", "Embeddings", "Fine-tuning LLMs",
                        "Transformers", "Vector Search"]]
    return Candidate.from_dict({
        "candidate_id": "CAND_2000002",
        "profile": {
            "anonymized_name": "C D", "headline": "Marketing Manager",
            "summary": "Marketing manager. Also know AI, NLP, RAG, embeddings.",
            "location": "Noida, Uttar Pradesh", "country": "India",
            "years_of_experience": 7.0, "current_title": "Marketing Manager",
            "current_company": "AdCo", "current_company_size": "201-500",
            "current_industry": "Marketing",
        },
        "career_history": [
            {"company": "AdCo", "title": "Marketing Manager",
             "start_date": "2019-01-01", "end_date": None, "duration_months": 84,
             "is_current": True, "industry": "Marketing", "company_size": "201-500",
             "description": "Managed marketing campaigns and brand strategy."},
        ],
        "education": [], "skills": skills, "certifications": [], "languages": [],
        "redrob_signals": _base_signals(),
    })


def _honeypot_exp():
    # 8y experience but career started ~3y ago
    return Candidate.from_dict({
        "candidate_id": "CAND_3000003",
        "profile": {
            "anonymized_name": "E F", "headline": "AI Engineer",
            "summary": "AI engineer with ranking and retrieval experience.",
            "location": "Pune, Maharashtra", "country": "India",
            "years_of_experience": 8.0, "current_title": "AI Engineer",
            "current_company": "X", "current_company_size": "51-200",
            "current_industry": "Internet",
        },
        "career_history": [
            {"company": "X", "title": "AI Engineer", "start_date": "2023-06-01",
             "end_date": None, "duration_months": 35, "is_current": True,
             "industry": "Internet", "company_size": "51-200",
             "description": "Ranking, retrieval, embeddings."},
        ],
        "education": [], "skills": [], "certifications": [], "languages": [],
        "redrob_signals": _base_signals(),
    })


def _honeypot_skills():
    # expert in many skills with 0 months used and 0 endorsements
    skills = [{"name": n, "proficiency": "expert", "endorsements": 0,
               "duration_months": 0}
              for n in ["A", "B", "C", "D", "E", "F", "G"]]
    return Candidate.from_dict({
        "candidate_id": "CAND_4000004",
        "profile": {
            "anonymized_name": "G H", "headline": "AI Engineer",
            "summary": "Expert in everything.", "location": "Pune", "country": "India",
            "years_of_experience": 6.0, "current_title": "AI Engineer",
            "current_company": "Y", "current_company_size": "51-200",
            "current_industry": "Internet",
        },
        "career_history": [
            {"company": "Y", "title": "AI Engineer", "start_date": "2019-06-01",
             "end_date": None, "duration_months": 60, "is_current": True,
             "industry": "Internet", "company_size": "51-200", "description": "ML."},
        ],
        "education": [], "skills": skills, "certifications": [], "languages": [],
        "redrob_signals": _base_signals(),
    })


def test_honeypot_experience_detected():
    integ = assess_integrity(_honeypot_exp())
    assert integ.is_honeypot, "experience-exceeds-career honeypot should be flagged"


def test_honeypot_skills_detected():
    integ = assess_integrity(_honeypot_skills())
    assert integ.is_honeypot, "expert-in-many-zero-duration-skills should be flagged"


def test_real_candidate_not_flagged():
    integ = assess_integrity(_real_engineer())
    assert not integ.is_honeypot, "a real engineer must not be flagged as a honeypot"


def test_stuffer_ranks_below_real_engineer():
    pool = [_real_engineer(), _keyword_stuffer()]
    rows = rank_candidates(pool, SPEC, shortlist_k=10, n_results=2)
    order = [r.candidate_id for r in rows]
    assert order[0] == "CAND_1000001", "real engineer should outrank the keyword stuffer"
    assert rows[0].score >= rows[1].score


def test_output_format_valid():
    # build a small pool with the real + stuffer duplicated with distinct ids
    pool = []
    base_real = _real_engineer()
    base_stf = _keyword_stuffer()
    for i in range(60):
        d = dict(base_real.raw)
        d = {**d, "candidate_id": f"CAND_50{i:05d}"}
        pool.append(Candidate.from_dict(d))
    for i in range(60):
        d = {**base_stf.raw, "candidate_id": f"CAND_60{i:05d}"}
        pool.append(Candidate.from_dict(d))
    rows = rank_candidates(pool, SPEC, shortlist_k=200, n_results=100)
    assert len(rows) == 100
    ranks = [r.rank for r in rows]
    assert ranks == list(range(1, 101)), "ranks must be 1..100 unique and ordered"
    scores = [r.score for r in rows]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)), \
        "scores must be non-increasing"
    # ties broken by candidate_id ascending
    for i in range(len(rows) - 1):
        if rows[i].score == rows[i + 1].score:
            assert rows[i].candidate_id <= rows[i + 1].candidate_id


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
