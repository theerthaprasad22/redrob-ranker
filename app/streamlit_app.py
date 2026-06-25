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
from redrob_ranker.ingest import load_any                # noqa: E402

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


spec = _load_spec()

# Sidebar label -> ingest fmt code (None = auto-detect / sniff).
_FORMAT_CHOICES = {
    "Auto-detect": None,
    "JSON / JSONL": "json",
    "CSV": "csv",
    "TSV": "tsv",
    "Excel (.xlsx)": "xlsx",
    "Plain text / résumé": "text",
}

with st.sidebar:
    st.header("Input")
    st.write("Upload candidates as **JSON/JSONL, CSV, Excel (.xlsx), or plain text** "
             "(one record per line/row/block), paste them directly. ")
    uploaded = st.file_uploader(
        "Candidate file",
        type=["jsonl", "json", "txt", "md", "csv", "tsv", "xlsx", "xlsm"],
    )
    pasted = st.text_area(
        "…or paste candidates here",
        height=140,
        placeholder='JSON array, one-JSON-object-per-line, CSV rows, or résumé text.',
    )
    fmt_label = st.selectbox(
        "Format",
        list(_FORMAT_CHOICES),
        index=0,
        help="Auto-detect uses the file extension (or sniffs pasted text). "
             "Override it if a file has a missing/misleading extension, or to "
             "tell the parser how to read pasted text. Excel can't be pasted — "
             "upload an .xlsx file for that.",
    )
    fmt_code = _FORMAT_CHOICES[fmt_label]

    has_manual_input = (uploaded is not None) or bool(pasted.strip())
    use_sample = st.checkbox(
        "Use bundled demo sample (150 candidates)", value=not has_manual_input
    )
    top_n = st.slider("How many to rank", 5, 100, 25)
    st.divider()


candidates: list[Candidate] = []
load_error = ""
try:
    if not use_sample and uploaded is not None:
        # Raw bytes + filename so binary formats (.xlsx) work; fmt overrides detection.
        candidates = load_any(uploaded.getvalue(), filename=uploaded.name, fmt=fmt_code)
        if pasted.strip():
            st.info("Both a file and pasted text were provided — using the uploaded file.")
    elif not use_sample and pasted.strip():
        if fmt_code == "xlsx":
            raise ValueError("Excel can't be pasted as text. Upload an .xlsx file, "
                             "or pick a text-based format for pasted content.")
        candidates = load_any(pasted, fmt=fmt_code)  # no filename → sniff if auto
    elif use_sample:
        sample_path = os.path.join(ROOT, "sample_data", "demo_candidates.jsonl")
        if os.path.exists(sample_path):
            candidates = load_any(sample_path)
        else:
            st.warning("Bundled sample not found; please upload or paste a file.")
except ImportError as e:
    load_error = str(e)  # e.g. .xlsx given without openpyxl installed
except Exception as e:  # noqa: BLE001 - surface any parse error to the user
    load_error = f"Could not parse the input: {e}"

if load_error:
    st.error(load_error)

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
