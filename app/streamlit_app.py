"""
Streamlit sandbox/demo for the Redrob ranker.

Satisfies the challenge's sandbox requirement: accepts a small candidate sample
(<=100), runs the full ranking pipeline end-to-end on CPU within the compute
budget, and shows the ranked shortlist with scores and grounded reasoning.

The role is no longer hard-coded: paste a job description into the "job role"
box and candidates are ranked against *that* role. With no JD pasted, the app
falls back to the bundled Senior-AI-Engineer spec so the demo still works
out of the box.

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
from redrob_ranker.jd_spec import build_spec_from_jd     # noqa: E402
from redrob_ranker.llm import LLMClient                  # noqa: E402
from redrob_ranker.explain import explain_candidate      # noqa: E402

st.set_page_config(page_title="Redrob Candidate Ranker", page_icon="🧭", layout="wide")

st.title("🧭 Redrob Intelligent Candidate Ranker")
st.caption(
    "Paste a job role, drop in candidates, and get a ranked shortlist — scored "
    "the way a recruiter would by reading career history, trust-weighted skills, "
    "and behavioral signals, not just keywords. Runs on CPU, no network needed."
)


@st.cache_resource
def _load_default_spec() -> RoleSpec:
    return RoleSpec.load(os.path.join(ROOT, "config", "role_spec.yaml"))


@st.cache_resource
def _llm_client() -> LLMClient:
    # Cheap to build; .enabled is False unless LLM_PROVIDER etc. are configured.
    return LLMClient()


@st.cache_data(show_spinner=False)
def _build_spec_cached(jd_text: str):
    """Build a spec from the pasted JD. Cached on the JD text so we don't
    re-run the (possibly LLM-backed) build on every Streamlit rerun."""
    base = _load_default_spec()
    client = _llm_client()
    return build_spec_from_jd(jd_text, llm_client=client, base_spec=base)


# Sidebar label -> ingest fmt code (None = auto-detect / sniff).
_FORMAT_CHOICES = {
    "Auto-detect": None,
    "JSON / JSONL": "json",
    "CSV": "csv",
    "TSV": "tsv",
    "Excel (.xlsx)": "xlsx",
    "Plain text / résumé": "text",
}

# ---------------------------------------------------------------------------
# Step 1 — the job role.  This is the context everything is ranked against.
# ---------------------------------------------------------------------------
st.subheader("Paste the job role")
jd_text = st.text_area(
    "Job title + description — what are you hiring for?",
    height=200,
    placeholder=(
        "e.g.\n\n"
        "Senior Frontend Engineer\n\n"
        "We're looking for a frontend engineer to own our React + TypeScript web app. "
        "You'll build accessible, performant UI and work closely with design.\n\n"
        "Must have: 5+ years with React, strong TypeScript, CSS/HTML.\n"
        "Nice to have: Next.js, testing (Jest/Playwright), design-system experience.\n"
        "Location: Remote (EU time zones)."
    ),
    help="Candidates are ranked against this role. Leave it empty to use the "
         "bundled Senior-AI-Engineer demo role.",
)

# Build the spec from whatever is (or isn't) pasted.
spec_result = _build_spec_cached(jd_text)
spec = spec_result.spec

# ---------------------------------------------------------------------------
# Show what the app understood the role to be (transparency for the recruiter).
# ---------------------------------------------------------------------------
_METHOD_BADGE = {
    "llm": "🤖 parsed by LLM",
    "heuristic": "⚙️ parsed offline (no LLM)",
    "default": "📌 bundled demo role (no JD pasted)",
}
with st.container():
    if spec_result.method == "default":
        st.info(
            f"**No job role pasted — using the bundled demo role:** "
            f"{spec_result.title or 'Senior AI Engineer'}. "
            f"Paste a description above to rank against your own role."
        )
    else:
        role_kind = "AI/ML role" if spec_result.is_ai_role else "general role"
        st.success(
            f"**Role understood:** {spec_result.title or '(untitled role)'}  "
            f"· {_METHOD_BADGE.get(spec_result.method, spec_result.method)} · {role_kind}"
        )
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**Must-have signals**")
            if spec_result.must_haves:
                st.write(", ".join(spec_result.must_haves))
            else:
                st.write("_none detected_")
        with cc2:
            st.markdown("**Nice-to-have signals**")
            if spec_result.nice_to_haves:
                st.write(", ".join(spec_result.nice_to_haves))
            else:
                st.write("_none detected_")
        for note in spec_result.notes:
            st.caption(f"ℹ️ {note}")

    # Transparency: show the component weights this role resolved to. They shift
    # by seniority (e.g. a junior role leans on skills/education over tenure).
    _w = {k: v for k, v in (spec.weights or {}).items() if v and v > 0}
    if _w:
        _pretty = {
            "semantic_fit": "Semantic fit to the role",
            "role_title_fit": "Title / role match",
            "career_evidence": "Relevant career evidence",
            "skills_trust": "Trusted skills",
            "experience_fit": "Experience (role-relevant)",
            "domain_fit": "Domain match",
            "location_fit": "Location",
            "education_fit": "Education",
        }
        with st.expander("How this role is weighted"):
            st.caption("Experience is scored on role-relevant years, so time in an "
                       "unrelated field counts only partially and strong freshers "
                       "aren't penalised on entry-level roles.")
            st.write({_pretty.get(k, k): round(v, 3)
                      for k, v in sorted(_w.items(), key=lambda kv: kv[1], reverse=True)})

st.divider()

# ---------------------------------------------------------------------------
# Step 2 — candidates.  Kept in the sidebar so the role stays front-and-centre.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.write("Upload candidates as **JSON/JSONL, CSV, Excel (.xlsx), or plain text** "
             "(one record per line/row/block), or paste them directly.")
    uploaded = st.file_uploader(
        "Candidate file",
        type=["jsonl", "json", "txt", "md", "csv", "tsv", "xlsx", "xlsm"],
    )
    pasted = st.text_area(
        "…or paste candidates here",
        height=140,
        placeholder='JSON array, one-JSON-object-per-line, CSV rows, or resume text.',
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

st.subheader("Rank")
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
    ranked_for = spec_result.title or "the role"
    st.success(f"Ranked {len(rows)} candidates for **{ranked_for}** in {elapsed:.2f}s on CPU.")

    by_id = {c.candidate_id: c for c in candidates}
    # Per-row trust layer (computed only for the displayed shortlist).
    explanations = {
        r.candidate_id: explain_candidate(by_id[r.candidate_id], spec, r.breakdown)
        for r in rows if by_id.get(r.candidate_id)
    }
    table = []
    for r in rows:
        c = by_id.get(r.candidate_id)
        p = c.raw.get("profile", {}) if c else {}
        ex = explanations.get(r.candidate_id)
        table.append({
            "rank": r.rank,
            "score": round(r.score, 4),
            "candidate_id": r.candidate_id,
            "current_title": p.get("current_title", ""),
            "years": p.get("years_of_experience", ""),
            "confidence": ex.confidence if ex else "",
            "missing must-haves": ", ".join(ex.missing) if ex else "",
            "location": p.get("location", ""),
            "reasoning": r.reasoning,
        })
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption("‘Confidence’ reflects how complete and consistent a profile is — "
               "treat low-confidence rows as needing a closer look, not as ranked facts.")

    # Recruiter-readable summary of the very top picks.
    top_for_summary = rows[: min(3, len(rows))]
    if top_for_summary:
        with st.expander("Recruiter view — top picks at a glance", expanded=True):
            for r in top_for_summary:
                ex = explanations.get(r.candidate_id)
                if not ex:
                    continue
                st.markdown(f"**#{r.rank} · {r.candidate_id}** — {ex.narrative}")
                if ex.confidence_reason:
                    st.caption(f"Confidence: {ex.confidence} — {ex.confidence_reason}")

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
