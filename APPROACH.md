# Approach — Redrob Intelligent Candidate Ranker

*Methodology write-up. This is also the narrative source for the submission deck.*

---

## 1. The real problem

Recruiters miss great people not because the talent isn't there, but because
keyword filters can't see what matters. The challenge bakes this into the data
on purpose. The 100K pool contains four things designed to break naive systems:

| Trap | What it looks like | What breaks on it |
| --- | --- | --- |
| **Keyword stuffers** | HR Manager / Content Writer with 8–9 AI skills listed | keyword & embedding matchers rank them #1 |
| **Plain-language Tier-5s** | strong career, *no* buzzwords ("built a recommendation system serving millions") | keyword matchers miss them entirely |
| **Behavioral twins** | identical-looking profiles, very different engagement | profile-only scoring can't separate them |
| **~80 honeypots** | subtly impossible (8 yrs at a 3-yr-old company; "expert" in 10 unused skills) | embedding similarity ranks them like real experts → **disqualification at >10%** |

The brief is explicit: rank candidates **the way a great recruiter would** — by
understanding who fits, not by matching words. So our design goal was never
"maximise similarity to the JD"; it was "**reproduce recruiter judgement, and be
robust to all four traps.**"

## 2. Reading the JD the way it's meant to be read

The JD is unusually candid, and its "note for participants" is essentially the
grading rubric:

- Not "most AI keywords" — that's a trap.
- Reason about the **gap between what the JD says and means.**
- Career evidence (shipped a recsys at a product company) beats a buzzword list.
- AI skills + an off-role title (Marketing Manager) is **not** a fit.
- Down-weight perfect-on-paper-but-unavailable candidates.

We distilled the JD once into a structured **role spec** (`config/role_spec.yaml`):
the must-haves (embeddings retrieval, vector DB / hybrid search, ranking
evaluation, strong Python), the nice-to-haves (LoRA/PEFT, learning-to-rank,
HR-tech, OSS), the ideal profile (6–8 yrs, 4–5 in applied ML at *product*
companies, shipped an end-to-end ranking/search/recsys system, Noida/Pune), and
— most importantly — the explicit **"do NOT want"** list as graduated penalties.
Doing this once (offline) means the 100K-candidate ranking step stays fast and
network-free.

## 3. Architecture: a hybrid funnel

No single technique survives all four traps, so we combine several, each
covering the others' blind spots, in a funnel that is also what keeps us inside
the 5-minute budget.

```
parse → integrity filter → cheap prefilter (100K→K) → semantic encode
      → full multi-dimensional scoring → behavioral modifier → rank → reason
```

**Why a funnel.** The pool is ~94% obvious decoys (HR, Sales, Mechanical
Engineer, Accountant…). A cheap, recall-oriented prefilter removes them in
seconds, so the expensive semantic + full scoring only runs on a few thousand
genuine contenders. This is the production latency-vs-quality tradeoff the JD
says it cares about.

### 3.1 Semantic understanding (catches the plain-language Tier-5s)

We embed a focused view of each candidate (headline, summary, recent titles and
role descriptions, skills) and compare it to a **plain-language description of
the role's meaning** — deliberately not a keyword dump. This is what lets a
candidate who wrote "built and shipped a personalised ranking system" match a
role asking for "retrieval and ranking experience," with zero shared keywords.

The encoder is pluggable. The default is **TF-IDF + Truncated SVD (latent
semantic analysis)** — fully local, no downloads, milliseconds per 1K docs,
which is why the whole system runs offline on CPU. An optional
**sentence-transformers** encoder (e.g. BGE-small) gives higher-quality
embeddings when a model is cached locally beforehand.

### 3.2 Career-evidence scoring (rewards what the JD actually wants)

Embedding similarity alone is gameable (stuffers paste keywords into summaries).
So the heaviest single component is **career evidence**: does the *career
history* show the person *shipped* ranking / search / recsys / retrieval systems
(shipping verbs co-occurring with relevant nouns), and at a **product** company
rather than pure IT-services? This is where genuine talent separates from
profile decoration.

### 3.3 Trust-weighted skills (defuses the keyword stuffer)

We do use the skills list — but every matched skill is weighted by a **trust**
factor built from proficiency, **endorsements**, **months actually used**, and
the candidate's **Redrob skill-assessment score**. A skill claimed "expert" with
0 months, 0 endorsements, and no assessment contributes almost nothing; a real
expert with years of use, endorsements, and an 85/100 assessment contributes
fully. Plain-language career evidence earns partial credit even when the skill
isn't explicitly listed. The result: stuffing keywords doesn't move the score.

### 3.4 Behavioral-signal modifier (separates the behavioral twins)

The 23 Redrob signals describe whether someone is *actually hireable now*. We
fold them into a multiplier on the fit score (≈0.60–1.15), weighting **recency
of activity** and **recruiter response rate** most heavily, with contributions
from open-to-work, recruiter saves, interview-completion, verification, GitHub
activity, and notice period. A strong-on-paper candidate who's been silent for
six months with a 5% response rate gets pulled down — exactly as the JD asks.

### 3.5 Integrity / honeypot detection (insurance against disqualification)

The spec says a good ranker should *naturally* avoid honeypots — and ours mostly
does, because their fake skills carry no trust and their thin careers carry no
evidence. But >10% honeypots in the top 100 is an automatic disqualification, so
we add explicit, **conservative** consistency checks that encode the documented
signatures: experience exceeding the span the career actually covers, roles
whose duration exceeds their date range, roles starting in the future, and
many advanced/expert skills with 0 months used and 0 endorsements. Conservative
by design — we only hard-drop clear impossibilities, so a real candidate is
never penalised. On the full pool this removes dozens of impossible profiles
before scoring.

### 3.6 The disqualifier penalties (the "do NOT want" list)

Applied as graduated multiplicative factors, traceable line-by-line to the JD:
services-only careers (unless there's prior product experience), pure research
with no production signal, "AI experience" that's only recent LLM-wrapper work,
title-chasers (many short stints), and CV/speech/robotics-only profiles.

## 4. Putting it together

```
base_fit = Σ wᵢ · componentᵢ        (semantic, role/title, career-evidence,
                                     trust-weighted skills, experience, domain,
                                     location, education)
base_fit × = disqualifier_factor    (the "do NOT want" penalties, floored)
final     = base_fit × behavioral_modifier × integrity_soft_penalty
```

Weights put career evidence and semantic understanding first and **deliberately
keep raw skill-list overlap from being the largest term** — the whole point is
that keywords aren't king. Final scores are normalised to [0, 1], sorted
descending, and tie-broken by `candidate_id` ascending so the output always
satisfies the validator.

## 5. Reasoning that survives manual review

Stage 4 samples 10 rows and checks for specific facts, JD connection, honest
concerns, **no hallucination**, variation, and tone-matching-rank. Our reasoning
is built from each candidate's own facts (title, years, the skills that actually
passed the trust filter, real signal numbers) plus the concerns surfaced by
scoring — so it is specific, honest about gaps, varied across rows, and tone is
selected by rank tier. An optional free-LLM polish rewrites the *same facts*
into smoother prose under a prompt that forbids adding anything not given, so it
can't invent skills.

## 6. Where the LLMs live

Hosted-LLM calls are banned during ranking, for good reason — they don't scale
to per-candidate inference. So LLMs sit **offline, around the run**, where a
production team would actually put them: regenerating the structured role spec
from the JD, and polishing the 100 final explanations. Free providers (Ollama,
Gemini free tier, Groq free tier) are supported; the default needs none, so the
system reproduces with zero keys and zero network.

## 7. Results & cost

- Full **100,000-candidate** pool ranked in **≈28 s** on CPU (8 cores), peak RAM
  ≈2.5 GB — comfortably inside 5 min / 16 GB / CPU-only / no-network.
- Top picks are Staff/Senior/Lead ML & AI Engineers, 6–8 yrs, active GitHub,
  Indian metros (incl. Noida) — no HR-manager stuffers, no honeypots.
- Dozens of impossible honeypot profiles removed before scoring.
- Cost: **$0** — local models for ranking, free-tier LLMs (optional) for the
  offline steps.

## 8. What we'd do next with labelled data

The architecture is built so that the "full scoring" stage can be replaced by a
**trained learning-to-rank model** (e.g. LightGBM/LambdaMART) or a **fine-tuned
cross-encoder reranker** the moment hiring-outcome labels exist — the current
components become features. Until then, a principled, transparent heuristic that
mirrors recruiter judgement is the honest and defensible choice, and it is fully
explainable in the Stage-5 interview.
