# Input formats

`rank.py` and the Streamlit app now accept several input formats, not just
`.jsonl`. The format is **auto-detected from the file extension**, converted
into the candidate schema internally, and fed to the exact same ranking engine.
Nothing about scoring changes — only how candidates get *in*.

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv   # original
python rank.py --candidates ./candidates.csv   --out ./submission.csv   # NEW
python rank.py --candidates ./candidates.xlsx  --out ./submission.csv   # NEW
python rank.py --candidates ./resumes.txt      --out ./submission.csv   # NEW
```

| Extension | Format | Notes |
| --- | --- | --- |
| `.jsonl`, `.ndjson`, `.jsonl.gz` | one JSON object per line | original path, unchanged |
| `.json` | a JSON array, a single object, or `{"candidates": [...]}` | |
| `.csv`, `.tsv` | one candidate per row | flat / dotted / delimited columns (below) |
| `.xlsx`, `.xlsm` | flat single sheet **or** multi-sheet workbook | needs `pip install openpyxl` |
| `.txt`, `.md`, other | free text; one candidate per block | best-effort; whole block becomes the summary |

> Templates you can copy are in `sample_data/`:
> `template_candidates.csv`, `template_candidates.xlsx`, `template_resume.txt`.

---

## The one thing to understand first

A candidate record is **deeply nested**: a `profile` object, a ~23-field
`redrob_signals` object, and *variable-length arrays* (`career_history`,
`education`, `skills`, `certifications`, `languages`). A spreadsheet row is
flat. So the only real question for CSV/Excel is **how the flat cells map onto
that nested shape.** There are three ways, and you can mix them freely.

### 1. Bare columns (the easy 90%)

These column names nest automatically — just use them as headers:

* **Profile:** `headline`, `summary`, `current_title`, `current_company`,
  `current_industry`, `current_company_size`, `location`, `country`,
  `years_of_experience`, `anonymized_name` (alias: `name`).
* **Signals:** `recruiter_response_rate`, `last_active_date`,
  `open_to_work_flag`, `github_activity_score`, `profile_completeness_score`,
  `notice_period_days`, `preferred_work_mode`, `willing_to_relocate`,
  `verified_email`, … (any field from `redrob_signals`).
* **Top-level:** `candidate_id` (auto-generated as `CAND_AUTO_000000N` if you
  omit it).

### 2. Delimited list columns (arrays in one cell)

To carry the arrays in a single flat row, put multiple entries in one cell
separated by `;`, with fields inside each entry separated by `|` in this fixed
order:

| Column | Field order (separated by `\|`) |
| --- | --- |
| `skills` | `name \| proficiency \| endorsements \| duration_months` |
| `career_history` | `title \| company \| industry \| duration_months \| description` |
| `education` | `institution \| degree \| field_of_study \| start_year \| end_year` |
| `certifications` | `name \| issuer \| year` |
| `languages` | `language \| proficiency` |

Example `skills` cell:

```
Python|expert|40|90; PyTorch|advanced|18|48; RAG|advanced|6|12
```

### 3. Dotted / indexed columns (full control)

Any header with dots builds nesting directly; an integer segment (1-based) means
a list position. Useful for signal sub-objects and per-row career entries:

```
profile.headline
redrob_signals.recruiter_response_rate
redrob_signals.expected_salary_range_inr_lpa.min
career_history.1.title      career_history.1.company
career_history.2.title      career_history.2.company
```

---

## Excel: multi-sheet (the lossless option)

For full fidelity without cramming arrays into single cells, use a workbook with
a **main sheet** (`candidates`, or just the first sheet) holding one flat row per
candidate, plus **child sheets keyed by `candidate_id`**:

* sheets named `career_history`, `education`, `skills`, `certifications`,
  `languages` — each row is one entry, attached to the matching candidate;
* an optional `signals` sheet (one row per candidate) merged into
  `redrob_signals`.

`sample_data/template_candidates.xlsx` is exactly this layout. If a workbook has
none of those child sheets, the first sheet is read as a flat table (rules 1–3
above).

---

## Free text (`.txt` / `.md`)

Plain text has no structure, so this is best-effort. Separate candidates with a
divider line of dashes (`---`) or a double-blank-line gap. Optional leading
`key: value` lines are recognised (`candidate_id`, `name`, `title`, `company`,
`location`, `country`, `years`); the **entire block is also stored as the
summary**, which is what the semantic encoder reads — so a plain résumé still
ranks on career evidence.

What text *can't* give you: trust-weighted skills, behavioral signals, and
integrity checks have little structured data to work with, so scores from text
lean almost entirely on the semantic/career-evidence side. If you need the full
signal-aware ranking from résumés, parse them into JSON/Excel first — see the
LLM pre-step below.

---

## Optional: LLM résumé parsing (`parse_resumes.py`)

To get the *skills* and *career-evidence* dimensions back from plain résumés,
run the offline parser first. It uses a free LLM (Ollama / Gemini / Groq —
configured exactly like the rest of the project, via env vars or `.env`) to
extract a structured record per candidate, then writes a `candidates.jsonl` that
`rank.py` consumes normally:

```bash
# 1) résumés -> structured JSONL  (offline-capable with Ollama; no network needed)
LLM_PROVIDER=ollama LLM_MODEL=llama3.1 \
    python parse_resumes.py --in resumes.txt --out candidates.jsonl

# 2) rank as usual (no network, no LLM in this step)
python rank.py --candidates candidates.jsonl --out submission.csv
```

`--in` accepts a single file (résumés separated by `---` or blank-line gaps) or
a **directory** of `.txt`/`.md` files (one résumé per file). Useful flags:
`--provider` / `--model` (override the env config) and `--allow-textfallback`
(write minimal text-only records if no LLM is configured, instead of exiting).

**This is a pre-step, never part of the timed ranking run** (the challenge
forbids network/LLM calls during ranking). Two honest limits by design:

* The prompt forbids inventing anything — the LLM extracts only what the résumé
  states or clearly implies.
* It **cannot** produce the Redrob behavioral signals (recruiter response rate,
  last-active date, GitHub score, …) — that platform data isn't in a résumé. So
  `redrob_signals` is left empty and the signal modifier treats every parsed
  candidate neutrally; résumé scores rest on the semantic / skills / career
  dimensions. If any single résumé fails to parse, it falls back to a minimal
  text-only record so no candidate is lost.

> Configure the LLM in a local `.env` (gitignored) — see `.env.example`. **Never
> commit or share a `.env` with a real API key.**

---

## Sandbox UI: format override + paste

In the Streamlit app you can also **paste** candidates directly instead of
uploading, and use the **Format** dropdown to override auto-detection — handy for
a file with a missing/misleading extension, or to tell the parser how to read
pasted text (JSON, CSV rows, or a résumé). Excel must be uploaded as a file
(it can't be pasted as text).

---

## Type handling

* Numbers and booleans are coerced automatically (`true/false/yes/no/1/0` →
  bool for known flag fields).
* Dates stay as strings in `YYYY-MM-DD` (the schema and date parser expect that).
* `candidate_id` and any `*_date` field are never numeric-coerced.
* Unknown **bare** columns are ignored; use a dotted header
  (e.g. `profile.my_field`) if you need to force an unrecognised field in.
