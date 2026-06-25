# Redrob Intelligent Candidate Ranker

A candidate-ranking system for the **Intelligent Candidate Discovery & Ranking
Challenge**. It ranks the top 100 candidates for the *Senior AI Engineer —
Founding Team* role the way a thoughtful recruiter would: by reading the **whole
profile** — career history, what they actually shipped, skills (and whether
those skills are real), and platform behaviour — not by counting keywords.

> The hard part of this challenge isn't matching keywords — it's *not* being
> fooled by them. The dataset is full of HR Managers with nine AI skills,
> "experts" who have never used the skill, and ~80 impossible honeypot profiles.
> A keyword/embedding-only ranker walks straight into all three traps. This
> system is built specifically to avoid them.

---

## What it does (and why it's built this way)

The JD's own "note for participants" tells you the rules of the game:

- The right answer is **not** "most AI keywords." That's an explicit trap.
- Reason about the **gap between what the JD says and what it means.** A strong
  candidate may never write "RAG" or "Pinecone" but their career shows they
  built a recommendation system at a product company — that's a fit.
- AI keywords + title "Marketing Manager" → **not** a fit.
- **Down-weight** perfect-on-paper candidates who aren't actually available
  (stale logins, 5% recruiter response rate).

So the ranker is a **hybrid funnel** with five ideas doing the work:

1. **Semantic understanding** (embeddings) reads plain-language career history,
   so a Tier-5 candidate with no buzzwords still surfaces.
2. **Career-evidence scoring** rewards *shipping* ranking / search / recsys /
   retrieval systems at *product* companies — the thing the JD actually wants.
3. **Trust-weighted skills** counter keyword stuffing: a skill claimed "expert"
   with 0 months used, 0 endorsements, and a low Redrob assessment score counts
   for almost nothing.
4. **Behavioral-signal modifier** turns the 23 Redrob signals into an
   availability/quality multiplier, pulling down the stale-and-unresponsive.
5. **Integrity / honeypot detection** filters subtly impossible profiles before
   they can reach the top 100 (the challenge disqualifies you at >10%).

A small structured **role spec** (`config/role_spec.yaml`), distilled from the
JD, encodes the must-haves, nice-to-haves, and the explicit "do NOT want" list
(services-only careers, pure research, recent-LangChain-only, title-chasers,
CV/speech/robotics-only) as graduated penalties.

### Where the LLMs fit (and why they're not in the hot loop)

The challenge forbids network/hosted-LLM calls **during ranking** (5 min, 16 GB,
CPU-only, no network) — because a system that calls an LLM per candidate can't
scale in production. So LLMs are used **offline, around the run**, exactly where
a production system would put them:

- **JD understanding** — `precompute.py` can (re)generate `config/role_spec.yaml`
  from the JD text using a free LLM.
- **Reasoning** — `--reasoning-llm` can polish the top-100 explanations into
  natural prose, fed *only* profile-derived facts so it cannot hallucinate.

Both are **optional**. The default pipeline uses a fully local TF-IDF + SVD
(latent semantic analysis) encoder and deterministic, grounded reasoning, so the
whole thing runs with **no API keys and no network**. Free LLM providers
supported: **Ollama** (local, no key), **Google Gemini** (free tier), **Groq**
(free tier).

---

## Quickstart

```bash
# 1. Install (core deps only — numpy / scipy / scikit-learn / pyyaml)
pip install -r requirements.txt

# 2. Produce the submission from the candidate pool (the single Stage-3 command)
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# 3. Validate the format with the official validator
python validate_submission.py submission.csv
```

On the full 100,000-candidate pool this runs in **well under a minute** on a
CPU (≈28 s on an 8-core box), inside the 5-min / 16 GB / CPU-only / no-network
budget. `candidates.jsonl.gz` is also accepted directly.

> **Other input formats.** `--candidates` also accepts **CSV, TSV, Excel
> (`.xlsx`), JSON, and plain text** — the format is auto-detected from the
> extension and converted to the candidate schema internally (the ranking engine
> is unchanged). Excel needs `pip install openpyxl`; the rest need nothing extra.
> Column conventions and copy-paste templates are in
> [`INPUT_FORMATS.md`](INPUT_FORMATS.md) (`sample_data/template_candidates.csv`,
> `…template_candidates.xlsx`, `…template_resume.txt`). To recover the skills /
> career dimensions from plain résumés, an optional offline LLM pre-step,
> `parse_resumes.py`, extracts structured records before ranking (see
> `INPUT_FORMATS.md`).

### Options

```bash
# Higher-quality semantic encoder (needs a locally cached sentence-transformers
# model; see "Optional: better embeddings" below). Still offline at run time.
python rank.py --candidates ./candidates.jsonl --out ./submission.csv --encoder st

# Polish the top-100 reasoning with a free LLM (OFFLINE step, 100 rows only)
LLM_PROVIDER=ollama LLM_MODEL=llama3.1 \
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv --reasoning-llm
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--candidates` | (required) | Path to `candidates.jsonl` / `.jsonl.gz` |
| `--out` | `submission.csv` | Output CSV |
| `--encoder` | `lsa` | `lsa` (local) or `st` (sentence-transformers) |
| `--shortlist` | `6000` | Deep-scoring shortlist size (recall funnel) |
| `--top` | `100` | Rows to output |
| `--reasoning-llm` | off | Polish reasoning via a free LLM (offline) |

### Configuring a free LLM (optional, for `--reasoning-llm`)

The LLM settings can be real environment variables or, more conveniently, a
local `.env` file in the repo root (it's gitignored, so your key never gets
committed). Copy the template and fill it in:

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

```
# .env  (Gemini example — free tier, no local model)
LLM_PROVIDER=gemini
GEMINI_API_KEY=AIza...your_key_here
LLM_MODEL=gemini-2.5-flash
```

Then just run `python rank.py … --reasoning-llm` — the key is picked up
automatically. Get a free Gemini key at aistudio.google.com (no card). The free
tier is rate-limited (~10 requests/min), so 100 rows take a couple of minutes and
some may fall back to the deterministic reasoning — that's expected. For no
limits and no network, use Ollama instead (`LLM_PROVIDER=ollama`).

---

## How it works (architecture)

```
100K candidates
   │  parse + normalise                         src/redrob_ranker/schema.py
   ▼
integrity / honeypot filter  ── drop impossible src/redrob_ranker/integrity.py
   │
   ▼
cheap prefilter score  ── 100K → top-K (recall) src/redrob_ranker/scoring.py
   │
   ▼
semantic encode shortlist  ── LSA / ST          src/redrob_ranker/embeddings.py
   │
   ▼
full multi-dimensional scoring                  src/redrob_ranker/scoring.py
   │   semantic · role/title · career-evidence ·
   │   trust-weighted skills · experience ·
   │   domain · location · education · disqualifiers
   ▼
behavioral-signal modifier                      src/redrob_ranker/signals.py
   │
   ▼
final = base_fit × modifier × integrity_soft
   │   sort, normalise, tie-break by candidate_id
   ▼
top-100  →  grounded reasoning                  src/redrob_ranker/reasoning.py
```

The funnel is what keeps it fast *and* good: a cheap, recall-oriented prefilter
removes the ~94% of the pool that is plainly irrelevant (the dataset is mostly
HR / Sales / Mechanical-Engineer decoys), and the expensive semantic + full
scoring only runs on a few thousand genuine contenders.

See [`APPROACH.md`](APPROACH.md) for the full methodology write-up (this is also
the source for the submission deck).

---

## Repository layout

```
redrob-ranker/
├── rank.py                      # MAIN entrypoint (candidates.jsonl -> submission.csv)
├── precompute.py                # OPTIONAL offline: LLM-regenerate role spec / cache ST model
├── validate_submission.py       # official format validator (for local checks)
├── requirements.txt
├── submission_metadata.yaml     # portal metadata mirror
├── APPROACH.md                  # methodology (deck source)
├── config/
│   ├── role_spec.yaml           # the JD distilled into structured scoring rules
│   └── candidate_schema.json    # dataset schema (reference)
├── src/redrob_ranker/
│   ├── schema.py                # Candidate model + streaming JSONL loader
│   ├── role_spec.py             # RoleSpec loader + semantic query
│   ├── embeddings.py            # pluggable encoder: LSA (default) / sentence-transformers
│   ├── integrity.py             # honeypot / impossible-profile detection
│   ├── scoring.py               # recruiter-style multi-dimensional scoring
│   ├── signals.py               # behavioral-signal modifier
│   ├── reasoning.py             # grounded reasoning (+ optional LLM polish)
│   ├── llm.py                   # free LLM providers (ollama / gemini / groq / none)
│   └── pipeline.py              # orchestration funnel
├── app/streamlit_app.py         # sandbox / demo UI
├── scripts/make_sample.py       # build a small candidate sample
├── tests/test_pipeline.py       # honeypot, anti-stuffing, format tests
└── sample_data/                 # small samples for the demo / tests
```

---

## Sandbox / demo

The challenge requires a hosted sandbox that runs the ranker on ≤100 candidates.
A Streamlit app is included.

```bash
pip install streamlit
streamlit run app/streamlit_app.py
```

Upload a small `.jsonl` (or use the bundled sample), and it returns the ranked
top candidates with scores and reasoning, well within the compute budget.

**Deploying for free:** see [`DEPLOY.md`](DEPLOY.md) for exact click-by-click steps. In short:
- **Streamlit Community Cloud** — point it at this repo and `app/streamlit_app.py`.
- **Hugging Face Spaces** — create a Streamlit Space with this repo.
- **Docker** — `docker build -t redrob-ranker . && docker run -p 8501:8501 redrob-ranker`.

---

## Optional: better embeddings (sentence-transformers)

The default LSA encoder needs no downloads and is the proven, offline-safe path.
For higher semantic quality you can switch to sentence-transformers. Because the
ranking step has **no network**, the model must be cached first:

```bash
pip install sentence-transformers
python precompute.py --cache-model         # downloads + caches the model once
python rank.py --candidates ./candidates.jsonl --out ./submission.csv --encoder st
```

To keep within 5 minutes, the ST path only encodes the retrieval shortlist
(a few thousand candidates), not the full pool.

---

## Reproducibility & compute

- Deterministic by default (fixed SVD seed; deterministic tie-breaks).
- Default run: CPU-only, no network, no model download, < 1 min on 100K, peak
  RAM ≈ 2.5 GB.
- Tests: `python tests/test_pipeline.py` (honeypot detection, anti-stuffing,
  format validity).

## Limitations / honest notes

- The role spec is tuned from the JD by hand (and optionally an LLM); a few
  weights (e.g. how hard to penalise sub-band experience or non-India location)
  are deliberate, defensible judgement calls and are easy to adjust in
  `config/role_spec.yaml`.
- Without labelled hiring outcomes, scoring is a principled heuristic, not a
  trained learning-to-rank model. The architecture is built so that a trained
  reranker could drop into the "full scoring" stage if labels become available.
