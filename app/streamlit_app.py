"""
Streamlit sandbox/demo for the Redrob ranker.

Satisfies the challenge's sandbox requirement: accepts a small candidate sample
(<=100), runs the full ranking pipeline end-to-end on CPU within the compute
budget, and shows the ranked shortlist with scores and grounded reasoning.

Run locally:   streamlit run app/streamlit_app.py
Deploy free:   Streamlit Community Cloud or HuggingFace Spaces (point at this repo)
"""

from __future__ import annotations

import io
import json
import os
import sys
import time

import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from redrob_ranker.pipeline import rank_candidates      # noqa: E402
from redrob_ranker.role_spec import RoleSpec             # noqa: E402
from redrob_ranker.schema import Candidate               # noqa: E402

st.set_page_config(page_title="Redrob Candidate Ranker", page_icon="🧭", layout="wide")

st.title("🧭 Redrob Intelligent Candidate Ranker")
st.caption(
    "Ranks candidates the way a recruiter would — reading career history, "
    "trust-weighted skills, and behavioral signals, not just keywords. "
    "Runs fully on CPU, no network, no API keys."
)


@st.cache_resource
def _load_spec():
    return RoleSpec.load(os.path.join(ROOT, "config", "role_spec.yaml"))


def _read_candidates(raw_text: str) -> list[Candidate]:
    cands: list[Candidate] = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            cands.append(Candidate.from_dict(json.loads(line)))
        except json.JSONDecodeError:
            continue
    return cands


spec = _load_spec()

with st.sidebar:
    st.header("Input")
    st.write("Upload a `.jsonl` of candidate records (one JSON object per line), "
             "or use the bundled sample.")
    uploaded = st.file_uploader("Candidate JSONL", type=["jsonl", "json", "txt"])
    use_sample = st.checkbox("Use bundled demo sample (150 candidates)", value=uploaded is None)
    top_n = st.slider("How many to rank", 5, 100, 25)
    st.divider()
    

raw_text = ""
if uploaded is not None and not use_sample:
    raw_text = io.TextIOWrapper(uploaded, encoding="utf-8").read()
elif use_sample:
    sample_path = os.path.join(ROOT, "sample_data", "demo_candidates.jsonl")
    if os.path.exists(sample_path):
        raw_text = open(sample_path, "r", encoding="utf-8").read()
    else:
        st.warning("Bundled sample not found; please upload a file.")

candidates = _read_candidates(raw_text) if raw_text else []

col_a, col_b = st.columns([1, 3])
with col_a:
    st.metric("Candidates loaded", len(candidates))
    run = st.button("Rank candidates", type="primary", disabled=not candidates)

if candidates and (len(candidates) > 100):
    st.info(f"{len(candidates)} candidates provided. The sandbox is intended for "
            f"<=100; ranking the top {top_n} of the first 100.")
    candidates = candidates[:100]

if run and candidates:
    t0 = time.time()
    with st.spinner("Ranking (parse → integrity → prefilter → semantic → score → signals → reason)…"):
        rows = rank_candidates(candidates, spec, shortlist_k=max(200, len(candidates)),
                               n_results=min(top_n, len(candidates)))
    elapsed = time.time() - t0
    st.success(f"Ranked {len(rows)} candidates in {elapsed:.2f}s on CPU.")

    by_id = {c.candidate_id: c for c in candidates}
    table = []
    for r in rows:
        c = by_id.get(r.candidate_id)
        p = c.raw.get("profile", {}) if c else {}
        table.append({
            "rank": r.rank,
            "score": round(r.score, 4),
            "candidate_id": r.candidate_id,
            "current_title": p.get("current_title", ""),
            "years": p.get("years_of_experience", ""),
            "location": p.get("location", ""),
            "reasoning": r.reasoning,
        })
    st.dataframe(table, use_container_width=True, hide_index=True)

    # detail expander for the top pick
    if rows and rows[0].breakdown is not None:
        with st.expander("Why #1? — score breakdown"):
            comps = rows[0].breakdown.components
            st.write({k: round(v, 3) for k, v in comps.items()})
            st.write(f"disqualifier_factor: {rows[0].breakdown.disqualifier_factor}")
            if rows[0].breakdown.concerns:
                st.write("concerns:", rows[0].breakdown.concerns)

    # download
    import csv as _csv
    buf = io.StringIO()
    w = _csv.writer(buf)
    w.writerow(["candidate_id", "rank", "score", "reasoning"])
    for r in rows:
        w.writerow([r.candidate_id, r.rank, f"{r.score:.6f}", r.reasoning])
    st.download_button("Download ranked CSV", buf.getvalue(),
                       file_name="submission_sample.csv", mime="text/csv")
else:
    st.write("Load candidates and click **Rank candidates** to see the shortlist.")
