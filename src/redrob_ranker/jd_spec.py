"""
jd_spec.py — turn a *pasted* job description into a RoleSpec the ranker can use.

Why this exists
---------------
The bundled ``config/role_spec.yaml`` describes ONE fixed role (the challenge's
Senior AI Engineer). That's correct for the timed 100K submission, but it means
the demo can only ever rank against that one role. A recruiter using the sandbox
wants to paste *their* job description and have candidates ranked against *it* —
otherwise there is no context to rank against at all.

This module bridges that gap. Given free-text JD, ``build_spec_from_jd`` returns
a ``RoleSpec`` shaped exactly like the YAML one, so the existing pipeline,
scoring, and reasoning code consume it unchanged.

Two paths, picked automatically:

  * LLM path (only if an LLM provider is configured via env / .env): asks a free
    LLM to extract a structured spec. Best quality. Falls back silently on any
    error.
  * Heuristic path (always available, no network, no keys): extracts the years
    band, locations, and required/nice skills from the text using a built-in
    concept ontology, and uses the JD itself as the semantic anchor. This is the
    piece that makes ranking role-aware even with zero LLM — the embedding
    similarity is computed against the real role text.

If the JD box is empty we just return the caller's base spec untouched, so the
out-of-the-box demo still works.

Nothing here runs during the timed ranking step; it's part of the sandbox /
around-the-run tooling, exactly like precompute.py.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .role_spec import RoleSpec


def _term_in(term: str, text: str) -> bool:
    """Whole-word-ish containment, to avoid substring traps like 'ux' matching
    'linux' or 'go' matching 'going'. Terms containing punctuation that breaks
    \\b semantics (c++, c#, .net, node.js, ci/cd, a/b test) fall back to a plain
    substring test, which is fine for those distinctive tokens."""
    if re.fullmatch(r"[a-z0-9 ]+", term):
        return re.search(r"\b" + re.escape(term) + r"\b", text) is not None
    return term in text

# ---------------------------------------------------------------------------
# Concept ontology — "if the JD mentions any of these terms, the role needs
# this capability." Each concept ships the synonyms the scorer matches against,
# so a plain-language candidate profile still hits. Deliberately broad (tech +
# data + some general knowledge-work) so most pasted JDs map onto something;
# anything not covered is still handled by the semantic anchor (the JD text).
# `ai` flags concepts that, if present, mark this as an AI/ML role and unlock
# the stricter anti-keyword disqualifiers.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Concept:
    key: str
    weight: float
    terms: tuple[str, ...]
    ai: bool = False


_ONTOLOGY: tuple[_Concept, ...] = (
    # --- programming languages -------------------------------------------------
    _Concept("python", 0.8, ("python", "pandas", "numpy", "fastapi", "django", "flask")),
    _Concept("java", 0.8, ("java", "spring", "spring boot", "jvm")),
    _Concept("javascript_ts", 0.8, ("javascript", "typescript", "node.js", "nodejs", "node ")),
    _Concept("golang", 0.7, ("golang", "go", "go programming")),
    _Concept("cpp", 0.7, ("c++", "cpp")),
    _Concept("csharp_dotnet", 0.7, ("c#", ".net", "dotnet", "asp.net")),
    _Concept("ruby", 0.6, ("ruby", "rails", "ruby on rails")),
    _Concept("php", 0.6, ("php", "laravel")),
    _Concept("rust", 0.6, ("rust",)),
    _Concept("scala", 0.6, ("scala",)),
    # --- AI / ML ---------------------------------------------------------------
    _Concept("machine_learning", 1.0, (
        "machine learning", "ml model", "ml models", "scikit-learn", "scikit learn",
        "supervised learning", "unsupervised", "feature engineering", "model training",
    ), ai=True),
    _Concept("deep_learning", 0.9, (
        "deep learning", "pytorch", "tensorflow", "keras", "neural network",
        "neural networks", "cnn", "rnn", "transformer", "transformers",
    ), ai=True),
    _Concept("nlp", 0.9, (
        "nlp", "natural language processing", "natural language", "text classification",
        "named entity", "language model", "tokeniz",
    ), ai=True),
    _Concept("llm", 0.9, (
        "llm", "large language model", "gpt", "prompt engineering", "rag",
        "retrieval augmented", "fine-tuning", "fine tuning", "langchain", "llamaindex",
    ), ai=True),
    _Concept("embeddings_retrieval", 1.0, (
        "embedding", "embeddings", "sentence-transformers", "sentence transformers",
        "semantic search", "dense retrieval", "vector search", "bge", "e5",
        "nearest neighbor", "ann", "retrieval",
    ), ai=True),
    _Concept("vector_db_hybrid_search", 0.9, (
        "vector database", "vector db", "pinecone", "weaviate", "qdrant", "milvus",
        "faiss", "opensearch", "elasticsearch", "elastic search", "hybrid search",
        "bm25", "lucene", "solr",
    ), ai=True),
    _Concept("ranking_search_recsys", 1.0, (
        "ranking", "ranker", "learning to rank", "ltr", "recommendation",
        "recommender", "recsys", "personalization", "personalisation",
        "search relevance", "matching", "candidate generation",
    ), ai=True),
    _Concept("ranking_evaluation", 0.8, (
        "ndcg", "mrr", "mean average precision", "precision@k", "recall@k",
        "offline evaluation", "a/b test", "ab test", "a/b testing", "ranking metrics",
    ), ai=True),
    _Concept("computer_vision", 0.9, (
        "computer vision", "image classification", "object detection", "segmentation",
        "opencv", "image recognition",
    ), ai=True),
    _Concept("data_science", 0.8, (
        "data science", "data scientist", "statistical", "statistics", "hypothesis test",
        "experimentation", "regression", "predictive model",
    ), ai=True),
    # --- data engineering ------------------------------------------------------
    _Concept("sql", 0.8, ("sql", "postgres", "postgresql", "mysql", "queries", "rdbms")),
    _Concept("nosql", 0.6, ("nosql", "mongodb", "cassandra", "dynamodb", "redis")),
    _Concept("data_pipelines", 0.8, (
        "etl", "elt", "data pipeline", "airflow", "dbt", "spark", "hadoop", "kafka",
        "data warehouse", "snowflake", "databricks", "bigquery",
    )),
    # --- backend / systems -----------------------------------------------------
    _Concept("backend_apis", 0.8, (
        "rest api", "restful", "api design", "microservice", "microservices",
        "graphql", "grpc", "backend",
    )),
    _Concept("distributed_systems", 0.7, (
        "distributed systems", "scalable", "scalability", "high availability",
        "low latency", "high throughput", "message queue", "event-driven",
    )),
    # --- frontend / mobile -----------------------------------------------------
    _Concept("frontend", 0.8, (
        "react", "angular", "vue", "frontend", "front-end", "html", "css", "tailwind",
        "next.js", "redux",
    )),
    _Concept("mobile", 0.8, (
        "ios", "android", "swift", "kotlin", "react native", "flutter", "mobile app",
    )),
    # --- cloud / devops --------------------------------------------------------
    _Concept("cloud", 0.8, (
        "aws", "azure", "gcp", "google cloud", "cloud", "ec2", "s3", "lambda",
    )),
    _Concept("devops", 0.7, (
        "docker", "kubernetes", "k8s", "ci/cd", "cicd", "terraform", "jenkins",
        "github actions", "infrastructure as code", "devops",
    )),
    _Concept("mlops", 0.7, (
        "mlops", "model serving", "model deployment", "mlflow", "feature store",
        "model monitoring", "triton", "onnx",
    ), ai=True),
    # --- general knowledge-work ------------------------------------------------
    _Concept("product_design", 0.8, (
        "product design", "ux", "ui/ux", "user experience", "figma", "wireframe",
        "prototyping", "design system", "usability",
    )),
    _Concept("product_management", 0.8, (
        "product manager", "product management", "roadmap", "user stories",
        "stakeholder", "go-to-market", "prioritization",
    )),
    _Concept("project_management", 0.7, (
        "project management", "scrum", "agile", "kanban", "jira", "sprint planning",
        "pmp",
    )),
    _Concept("marketing", 0.8, (
        "marketing", "seo", "sem", "content marketing", "campaign", "growth marketing",
        "demand generation", "brand",
    )),
    _Concept("sales", 0.8, (
        "sales", "quota", "crm", "salesforce", "account executive",
        "business development", "lead generation",
    )),
    _Concept("finance_accounting", 0.8, (
        "accounting", "financial reporting", "gaap", "audit", "financial modeling",
        "fp&a", "bookkeeping", "tax",
    )),
    _Concept("hr_recruiting", 0.8, (
        "recruiting", "recruitment", "talent acquisition", "hr", "human resources",
        "onboarding", "ats", "sourcing",
    )),
)

# Concepts whose presence means "treat this as an AI/ML role" and apply the
# stricter anti-keyword-stuffing disqualifiers.
_AI_KEYS = frozenset(c.key for c in _ONTOLOGY if c.ai)


# ---------------------------------------------------------------------------
# Built-in location vocabulary (extends the JD/spec one). Anything not here is
# still picked up if it appears verbatim in the JD's own "location:" style line;
# but mostly we want to recognise common hubs so location_fit means something.
# ---------------------------------------------------------------------------
_KNOWN_CITIES = {
    # India (matches the bundled spec)
    "pune", "noida", "hyderabad", "mumbai", "delhi", "new delhi", "gurgaon",
    "gurugram", "bengaluru", "bangalore", "ncr", "navi mumbai", "ghaziabad",
    "faridabad", "chennai", "kolkata", "ahmedabad", "jaipur",
    # global hubs
    "san francisco", "new york", "seattle", "austin", "boston", "london",
    "berlin", "amsterdam", "paris", "dublin", "toronto", "singapore", "sydney",
    "tokyo", "tel aviv", "dubai",
}
_KNOWN_COUNTRIES = {
    "india", "united states", "usa", "us", "uk", "united kingdom", "canada",
    "germany", "france", "netherlands", "ireland", "singapore", "australia",
    "uae", "israel",
}
_REMOTE_CUES = ("remote", "work from home", "wfh", "distributed team", "anywhere")


# ---------------------------------------------------------------------------
# Section / strength cues for must-have vs nice-to-have classification.
# ---------------------------------------------------------------------------
_NICE_CUES = (
    "nice to have", "nice-to-have", "good to have", "good-to-have", "bonus",
    "plus", "preferred", "preferably", "desirable", "ideally", "advantage",
    "appreciated", "optional", "would be great", "a plus",
)
_MUST_CUES = (
    "must have", "must-have", "required", "requirement", "requirements",
    "you have", "we expect", "essential", "need to have", "should have",
    "qualifications", "responsibilities", "what you'll do", "what you will do",
    "you'll be", "key skills", "minimum",
)


@dataclass
class SpecBuildResult:
    spec: RoleSpec
    method: str                       # "llm" | "heuristic" | "default"
    title: str = ""
    notes: list[str] = field(default_factory=list)
    must_haves: list[str] = field(default_factory=list)
    nice_to_haves: list[str] = field(default_factory=list)
    is_ai_role: bool = False


# ===========================================================================
# Public entry point
# ===========================================================================
def build_spec_from_jd(
    jd_text: str,
    *,
    llm_client=None,
    base_spec: Optional[RoleSpec] = None,
) -> SpecBuildResult:
    """Build a RoleSpec from pasted JD text.

    `base_spec` is returned unchanged when `jd_text` is empty (keeps the demo
    working with no JD). `llm_client` (a redrob_ranker.llm.LLMClient) is used
    only if it is enabled; any failure falls back to the heuristic path.
    """
    text = (jd_text or "").strip()
    if not text:
        spec = base_spec or RoleSpec.load()
        return SpecBuildResult(spec=spec, method="default",
                               title=str(spec.data.get("role", {}).get("title", "")),
                               notes=["No job description provided — using the bundled default role."])

    # 1) LLM path (best quality) when a provider is configured.
    if llm_client is not None and getattr(llm_client, "enabled", False):
        try:
            res = _build_with_llm(text, llm_client)
            if res is not None:
                return res
        except Exception:  # noqa: BLE001 — never let the LLM path break the app
            pass

    # 2) Deterministic, offline heuristic path.
    return _build_heuristic(text)


# ===========================================================================
# Heuristic builder (no network)
# ===========================================================================
def _build_heuristic(text: str) -> SpecBuildResult:
    low = text.lower()
    notes: list[str] = []

    title = _extract_title(text)
    must, nice = _classify_concepts(low)
    is_ai = bool({c.key for c in (must + nice)} & _AI_KEYS)

    exp_block, exp_found = _extract_experience(low)
    if not exp_found:
        notes.append("No explicit experience range found — experience is weighted lightly.")
    loc_block, loc_kind = _extract_locations(low)
    if loc_kind == "remote":
        notes.append("Role looks remote/location-flexible — location is not used for ranking.")
    elif loc_kind == "none":
        notes.append("No location found in the JD — location is not used for ranking.")

    notes.extend(_emphasis_notes(_seniority(low)))

    data = _assemble_spec_data(
        title=title, must=must, nice=nice, is_ai=is_ai,
        exp_block=exp_block, exp_found=exp_found,
        loc_block=loc_block, loc_kind=loc_kind, low=low,
    )
    spec = RoleSpec(data=data, query_text_override=_clean_anchor(text, title))

    return SpecBuildResult(
        spec=spec,
        method="heuristic",
        title=title,
        notes=notes,
        must_haves=[c.key for c in must],
        nice_to_haves=[c.key for c in nice],
        is_ai_role=is_ai,
    )


def _assemble_spec_data(*, title, must, nice, is_ai, exp_block, exp_found,
                        loc_block, loc_kind, low) -> dict[str, Any]:
    must_concepts = {c.key: {"weight": c.weight, "terms": list(c.terms)} for c in must}
    nice_concepts = {c.key: {"weight": c.weight, "terms": list(c.terms)} for c in nice}

    # Domain orientation: for an AI role keep the rich NLP/IR-vs-CV defaults so
    # the strong anti-keyword behaviour survives; for a general role derive
    # positives from the must-haves and leave negatives empty (don't mislabel).
    if is_ai:
        domain = {
            "positive_terms": ["nlp", "natural language", "information retrieval", "ir",
                               "search", "ranking", "recommendation", "embeddings", "llm",
                               "language model", "text", "semantic"],
            "negative_terms": [],  # CV/speech only counts against an *NLP* role; we
                                   # only add these back if NLP/IR is the actual ask.
        }
        # If the role is specifically NLP/IR/search/recsys, penalise CV/speech-only.
        if {c.key for c in must} & {"nlp", "embeddings_retrieval", "ranking_search_recsys",
                                    "vector_db_hybrid_search", "llm"}:
            if not ({c.key for c in must} & {"computer_vision"}):
                domain["negative_terms"] = ["computer vision", "image classification",
                                            "object detection", "segmentation",
                                            "speech recognition", "asr", "text-to-speech",
                                            "tts", "robotics", "slam", "lidar"]
    else:
        pos = sorted({t for c in must for t in c.terms} | {t for c in nice for t in c.terms})
        domain = {"positive_terms": pos[:40], "negative_terms": []}

    # Disqualifiers: always flag job-hopping (role-agnostic). Apply the
    # AI-specific anti-keyword penalties only when this is an AI/ML role.
    disq: dict[str, Any] = {
        "title_chaser": {"penalty": 0.8, "max_avg_tenure_months": 14, "min_jobs_for_flag": 4},
    }
    if is_ai:
        disq.update({
            "services_only": {
                "penalty": 0.6,
                "companies": ["tcs", "tata consultancy", "infosys", "wipro", "accenture",
                              "cognizant", "capgemini", "tech mahindra", "hcl", "hcltech",
                              "ltimindtree", "mindtree", "mphasis", "dxc", "genpact"],
                "services_industries": ["it services", "consulting", "outsourcing", "bpo",
                                        "staffing", "information technology & services"],
            },
            "research_only": {
                "penalty": 0.7,
                "terms": ["phd researcher", "research scientist", "research assistant",
                          "postdoc", "post-doc", "professor", "lecturer", "academia"],
                "production_terms": ["production", "deployed", "shipped", "launched",
                                     "users", "in production", "serving"],
            },
            "framework_only_recent": {
                "penalty": 0.7,
                "framework_terms": ["langchain", "llamaindex", "llama-index", "auto-gpt",
                                    "autogen", "openai api", "prompt engineering", "wrapper"],
                "legacy_ml_terms": ["machine learning", "ml model", "scikit", "tensorflow",
                                    "pytorch", "xgboost", "recommendation", "ranking", "nlp",
                                    "deep learning", "feature engineering"],
            },
            "wrong_domain": {"penalty": 0.6},
        })

    # Off-role titles only make sense as a stuffer-signal for AI roles; for a
    # general role we must NOT punish (a real Marketing Manager applying to a
    # marketing role should not be crushed).
    offrole = (
        ["marketing", "sales", "recruiter", "accountant", "finance", "content writer",
         "graphic designer", "customer success", "operations manager"]
        if is_ai else []
    )

    seniority = _seniority(low)
    exp_block = _experience_policy(exp_block, exp_found=exp_found, seniority=seniority)
    weights = _finalize_weights(seniority=seniority, exp_found=exp_found,
                                loc_kind=loc_kind, n_must=len(must))

    return {
        "role": {"title": title or "Custom role (from pasted JD)", "seniority": seniority},
        "experience": exp_block,
        "location": loc_block,
        "must_have_concepts": must_concepts,
        "nice_to_have_concepts": nice_concepts,
        "domain": domain,
        "disqualifiers": disq,
        "career_evidence": {
            "shipped_systems_terms": ["built", "shipped", "launched", "deployed", "designed",
                                      "led", "owned", "architected", "scaled", "delivered",
                                      "managed", "production"],
            "product_signal_terms": ["product", "saas", "platform", "consumer", "b2c", "b2b",
                                     "startup", "scale", "millions", "users", "customers",
                                     "real-time"],
            "target_title_terms": _title_terms(title),
            "role_noun": _role_noun(title),
            "offrole_titles": offrole,
            "positive_titles": [],
        },
        "weights": weights,
        "behavioral": {
            "modifier_min": 0.60, "modifier_max": 1.15, "stale_days": 120,
            "good_response_rate": 0.5, "good_notice_days": 30,
        },
    }


# ---------------------------------------------------------------------------
# Field extractors
# ---------------------------------------------------------------------------
def _classify_concepts(low: str) -> tuple[list[_Concept], list[_Concept]]:
    """Return (must_have, nice_to_have) concepts present in the JD.

    A concept goes to 'nice' if the segment that first mentions it sits under a
    nice-to-have cue; otherwise 'must'. Capped so weights stay meaningful.
    """
    segments = _segments_with_bucket(low)
    must: list[_Concept] = []
    nice: list[_Concept] = []
    for c in _ONTOLOGY:
        bucket = None
        for seg_text, seg_bucket in segments:
            if any(_term_in(t, seg_text) for t in c.terms):
                bucket = seg_bucket
                break
        if bucket is None:
            continue
        (nice if bucket == "nice" else must).append(c)

    # Keep the most relevant; AI concepts and higher weights first.
    must.sort(key=lambda c: (c.ai, c.weight), reverse=True)
    nice.sort(key=lambda c: (c.ai, c.weight), reverse=True)
    must = must[:9]
    nice = [c for c in nice if c.key not in {m.key for m in must}][:8]

    # A role with zero detected skills still needs *something* to score skills
    # against; fall back to a generic "core skills" bag from the JD's own nouns.
    if not must and not nice:
        must = [_Concept("core_role_skills", 1.0, tuple(_top_keywords(low)) or ("experience",))]
    return must, nice


def _segments_with_bucket(low: str) -> list[tuple[str, str]]:
    """Split the JD into small segments, tagging each must/nice by the nearest
    section header or inline cue. Sticky: a 'Nice to have:' header makes the
    following lines 'nice' until a must-cue header appears."""
    out: list[tuple[str, str]] = []
    current = "must"
    # Insert split points just before known section cues so an inline
    # 'Nice to have: X' at the end of a sentence becomes its own segment instead
    # of dragging the rest of the line into the wrong bucket.
    marked = low
    for cue in _NICE_CUES + _MUST_CUES:
        marked = marked.replace(cue, "\n" + cue)
    # split on newlines, bullets, sentence-enders, and semicolons
    raw_lines = re.split(r"[\n\r]+|[.;]\s+|(?:^|\s)[•\-\*\u2022]\s+", marked)
    for line in raw_lines:
        seg = line.strip()
        if not seg:
            continue
        has_nice = any(cue in seg for cue in _NICE_CUES)
        has_must = any(cue in seg for cue in _MUST_CUES)
        # Header-like (short line, often ending ':') flips the sticky section.
        is_headerish = len(seg) <= 60 and (seg.endswith(":") or len(seg.split()) <= 5)
        if has_nice and (is_headerish or not has_must):
            current = "nice"
        elif has_must and is_headerish:
            current = "must"
        seg_bucket = "nice" if has_nice and not has_must else current
        out.append((seg, seg_bucket))
    return out


def _extract_experience(low: str) -> tuple[dict[str, Any], bool]:
    """Pull a years-of-experience band out of the JD."""
    # range: "5-9 years", "5 to 9 years", "6–8 yrs"
    m = re.search(r"(\d{1,2})\s*(?:-|–|—|to)\s*(\d{1,2})\s*\+?\s*(?:years|yrs|yr)", low)
    if m:
        lo, hi = sorted((int(m.group(1)), int(m.group(2))))
        return {
            "min_years": lo, "max_years": hi,
            "ideal_low": lo + max(0, (hi - lo) // 4),
            "ideal_high": hi - max(0, (hi - lo) // 4),
            "soft_floor": max(0, lo - 2), "soft_ceiling": hi + 4,
        }, True
    # single: "5+ years", "minimum 5 years", "at least 5 years", "5 years"
    m = re.search(r"(?:minimum\s+(?:of\s+)?|at\s+least\s+)?(\d{1,2})\s*\+?\s*(?:years|yrs|yr)", low)
    if m:
        n = int(m.group(1))
        return {
            "min_years": n, "max_years": n + 5,
            "ideal_low": n, "ideal_high": n + 3,
            "soft_floor": max(0, n - 2), "soft_ceiling": n + 9,
        }, True
    # none: neutral wide band that barely penalises anyone
    return {
        "min_years": 0, "max_years": 40, "ideal_low": 2, "ideal_high": 20,
        "soft_floor": 0, "soft_ceiling": 45,
    }, False


def _extract_locations(low: str) -> tuple[dict[str, Any], str]:
    """Return (location_block, kind) where kind ∈ {best, remote, none}."""
    if any(_term_in(cue, low) for cue in _REMOTE_CUES):
        return ({"best": [], "good": [], "ok_country": []}, "remote")
    best = sorted({city for city in _KNOWN_CITIES if _term_in(city, low)})
    countries = sorted({c for c in _KNOWN_COUNTRIES if _term_in(c, low)})
    if best or countries:
        return ({"best": best, "good": [], "ok_country": countries}, "best")
    return ({"best": [], "good": [], "ok_country": []}, "none")


def _extract_title(text: str) -> str:
    """Best-effort role title: an explicit 'Role/Title/Position:' line, else the
    first short, title-cased-ish line near the top."""
    for line in text.splitlines()[:12]:
        m = re.match(r"\s*(?:role|title|position|job\s*title)\s*[:\-]\s*(.+)", line, re.I)
        if m:
            return m.group(1).strip()[:80]
    for line in text.splitlines()[:6]:
        s = line.strip()
        if 3 <= len(s) <= 70 and not s.endswith(".") and len(s.split()) <= 9:
            # avoid grabbing a sentence; favour something that looks like a title
            if not re.search(r"\b(we|you|the|our|is|are|seeking|looking)\b", s.lower()):
                return s[:80]
    # fallback: scan for "<seniority>? ... engineer/manager/designer/..."
    m = re.search(r"((?:senior|lead|staff|principal|junior|sr\.?|jr\.?)?\s*[\w/ ]{0,30}?"
                  r"(?:engineer|developer|scientist|manager|designer|analyst|architect|"
                  r"specialist|lead|consultant|administrator))", text, re.I)
    return m.group(1).strip()[:80] if m else ""


_PROF_NOUNS = (
    "engineer", "developer", "scientist", "manager", "designer", "analyst",
    "architect", "specialist", "consultant", "researcher", "administrator",
    "recruiter", "marketer", "accountant", "lead", "programmer",
)
_ROLE_PHRASES = (
    "product designer", "ux designer", "ui designer", "data scientist",
    "data engineer", "data analyst", "product manager", "project manager",
    "program manager", "backend engineer", "frontend engineer", "full stack",
    "full-stack", "machine learning engineer", "ml engineer", "ai engineer",
    "software engineer", "devops engineer", "site reliability", "sre",
    "cloud engineer", "platform engineer", "qa engineer", "mobile developer",
    "android developer", "ios developer", "marketing manager", "sales manager",
    "account executive", "business analyst", "research engineer", "nlp engineer",
    "security engineer", "solutions architect",
)


def _role_noun(title: str) -> str:
    """The role's primary profession noun (the last prof-noun in the title),
    used to grant 'same family, wrong specialty' partial title credit."""
    t = title.lower()
    found = [n for n in _PROF_NOUNS if _term_in(n, t)]
    if not found:
        return ""
    # pick the one that appears last (titles read 'Senior Backend Engineer')
    return max(found, key=lambda n: t.rfind(n))


def _title_terms(title: str) -> list[str]:
    """Tokens scoring._title_score uses to recognise the role's OWN title.

    Deliberately specific: we include the full title, the title minus seniority,
    the role's distinguishing token (e.g. 'devops', 'frontend', 'machine
    learning'), and known multi-word role phrases present in the title — but NOT
    bare nouns like 'engineer'/'manager', which would make every engineer match
    every engineering role.
    """
    if not title:
        return []
    t = title.lower().strip()
    terms = {t}
    bare = re.sub(r"\b(senior|sr\.?|lead|staff|principal|junior|jr\.?|head\s+of|"
                  r"entry[- ]level|graduate)\b", "", t).strip()
    bare = re.sub(r"\s{2,}", " ", bare)
    if len(bare) >= 4:
        terms.add(bare)
        # distinguishing token = bare minus the trailing profession noun,
        # e.g. 'devops engineer' -> 'devops', 'machine learning engineer' ->
        # 'machine learning'. Skip if it leaves nothing meaningful.
        core = bare
        for n in _PROF_NOUNS:
            core = re.sub(r"\s*\b" + re.escape(n) + r"\b\s*", " ", core).strip()
        core = re.sub(r"\s{2,}", " ", core)
        if len(core) >= 3 and core != bare:
            terms.add(core)
    for phrase in _ROLE_PHRASES:
        if phrase in t:
            terms.add(phrase)
    # keep only specific tokens (>=3 chars, not a bare profession noun alone)
    return sorted({x for x in terms if len(x) >= 3 and x not in _PROF_NOUNS})


def _seniority(low: str) -> str:
    if any(w in low for w in ("principal", "staff", "head of", "director")):
        return "staff+"
    if any(w in low for w in ("senior", "sr.", "lead")):
        return "senior"
    if any(w in low for w in ("junior", "entry", "graduate", "intern")):
        return "junior"
    return "mid"


def _top_keywords(low: str, k: int = 12) -> list[str]:
    """Cheap fallback 'skills' when nothing in the ontology matches: the most
    frequent meaningful words in the JD."""
    words = re.findall(r"[a-zA-Z][a-zA-Z+#./-]{2,}", low)
    stop = {
        "the", "and", "for", "with", "you", "our", "are", "will", "have", "this",
        "that", "your", "from", "who", "all", "can", "but", "not", "their", "they",
        "work", "team", "role", "job", "company", "experience", "years", "skills",
        "ability", "strong", "good", "great", "well", "looking", "seeking", "join",
        "build", "building", "across", "within", "including", "etc", "such", "into",
    }
    freq: dict[str, int] = {}
    for w in words:
        if w in stop or len(w) > 24:
            continue
        freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:k]]


_BASE_WEIGHTS = {
    "semantic_fit": 0.20, "role_title_fit": 0.18, "career_evidence": 0.22,
    "skills_trust": 0.16, "experience_fit": 0.12, "domain_fit": 0.07,
    "location_fit": 0.04, "education_fit": 0.01,
}

# What a role *needs* differs by seniority. Freshers are judged on demonstrated
# skills, education and role fit rather than tenure or a long track record;
# senior roles lean on relevant, role-specific experience and career evidence.
_SENIORITY_WEIGHTS = {
    "junior": {
        "semantic_fit": 0.22, "role_title_fit": 0.15, "career_evidence": 0.13,
        "skills_trust": 0.24, "experience_fit": 0.05, "domain_fit": 0.07,
        "location_fit": 0.04, "education_fit": 0.10,
    },
    "mid": dict(_BASE_WEIGHTS),
    "senior": {
        "semantic_fit": 0.18, "role_title_fit": 0.17, "career_evidence": 0.26,
        "skills_trust": 0.14, "experience_fit": 0.16, "domain_fit": 0.07,
        "location_fit": 0.02, "education_fit": 0.00,
    },
}
_SENIORITY_WEIGHTS["staff+"] = dict(_SENIORITY_WEIGHTS["senior"])


def _finalize_weights(*, seniority: str = "mid", exp_found: bool,
                      loc_kind: str, n_must: int = 0) -> dict[str, float]:
    """Pick a weighting profile for the role and renormalise to ~1.0.

    Starts from the seniority-appropriate mix, then: leans a little more on
    skills for skills-heavy JDs, keeps experience tiny when the JD never stated
    one (we can't fairly assess it), and drops location when there's nothing to
    match. Renormalising lets the remaining components absorb freed weight while
    keeping their relative proportions.
    """
    base = dict(_SENIORITY_WEIGHTS.get(seniority, _BASE_WEIGHTS))
    if n_must >= 5:  # a concrete, skills-heavy JD -> trust the skills signal more
        base["skills_trust"] += 0.03
        base["semantic_fit"] = max(0.0, base["semantic_fit"] - 0.03)
    if not exp_found:  # never told an experience range -> don't lean on it
        base["experience_fit"] = min(base["experience_fit"], 0.04)
    if loc_kind != "best":  # no usable location -> don't rank on it
        base["location_fit"] = 0.0
    s = sum(base.values()) or 1.0
    return {k: round(v / s, 4) for k, v in base.items()}


def _experience_policy(exp_block: dict[str, Any], *, exp_found: bool,
                       seniority: str) -> dict[str, Any]:
    """Attach a role-aware experience policy and, for entry-level roles with no
    stated range, recentre the band on 0–2 years so freshers are a clean fit."""
    junior = seniority == "junior"
    senior = seniority in ("senior", "staff+")
    low_req = (not exp_found) or int(exp_block.get("min_years", 0)) <= 1
    fresher_friendly = junior or (low_req and not senior)
    eb = dict(exp_block)
    if junior and not exp_found:
        eb.update({"min_years": 0, "max_years": 3, "ideal_low": 0, "ideal_high": 2,
                   "soft_floor": 0, "soft_ceiling": 6})
    eb["policy"] = {"relevance_aware": True, "fresher_friendly": fresher_friendly}
    return eb


def _emphasis_notes(seniority: str) -> list[str]:
    """Plain-language notes telling the recruiter how the role reshaped ranking."""
    notes = ["Experience is measured against this role — time spent in unrelated "
             "roles counts only partially, so a real fit isn't outranked by raw tenure."]
    if seniority == "junior":
        notes.append("Entry-level role — ranking favours demonstrated skills and "
                     "education over years, so strong freshers are not penalised.")
    elif seniority in ("senior", "staff+"):
        notes.append("Senior role — ranking leans on relevant, role-specific "
                     "experience and a demonstrated track record.")
    else:
        notes.append("Mid-level role — balanced weighting across skills, relevant "
                     "experience and track record.")
    return notes


def _clean_anchor(text: str, title: str) -> str:
    """The semantic anchor: a compact, readable version of the JD. Prepending
    the title sharpens the embedding."""
    collapsed = re.sub(r"\s+", " ", text).strip()
    head = f"{title}. " if title else ""
    return (head + collapsed)[:3000]


# ===========================================================================
# LLM builder (optional, best quality)
# ===========================================================================
_LLM_SYSTEM = (
    "You convert a job description into a STRICT JSON spec for a candidate "
    "ranking engine. Infer the recruiter's real intent, not just keywords. "
    "Output ONLY a single JSON object, no markdown, no prose."
)

_LLM_SCHEMA_HINT = """Return JSON with exactly these keys:
{
  "title": "short role title",
  "seniority": "junior|mid|senior|staff+",
  "query_text": "2-4 sentence description of the IDEAL candidate in plain language",
  "experience": {"min_years": int, "max_years": int, "ideal_low": int, "ideal_high": int},
  "must_have": [{"name": "snake_case", "weight": 0.5-1.0, "terms": ["synonyms","tools"]}],
  "nice_to_have": [{"name": "snake_case", "weight": 0.3-0.6, "terms": ["synonyms"]}],
  "locations": {"cities": ["lowercase"], "countries": ["lowercase"], "remote": true|false},
  "domain_positive": ["lowercase terms that indicate a good domain fit"],
  "domain_negative": ["lowercase terms that indicate the WRONG domain, or empty"],
  "avoid": ["short phrases describing candidates that are NOT a fit"]
}
Keep lists tight (<=8 items). terms must be lowercase."""


def _build_with_llm(text: str, client) -> Optional[SpecBuildResult]:
    raw = client.complete(
        f"{_LLM_SCHEMA_HINT}\n\nJob description:\n{text[:8000]}",
        system=_LLM_SYSTEM, max_tokens=1100, temperature=0.1,
    )
    if not raw:
        return None
    obj = _parse_json_blob(raw)
    if not isinstance(obj, dict):
        return None

    title = str(obj.get("title", "")).strip()
    must_raw = obj.get("must_have") or []
    nice_raw = obj.get("nice_to_have") or []

    def _concepts(items) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for it in items:
            if not isinstance(it, dict):
                continue
            name = re.sub(r"[^a-z0-9_]+", "_", str(it.get("name", "")).lower()).strip("_")
            terms = [str(t).lower() for t in (it.get("terms") or []) if str(t).strip()]
            if not name or not terms:
                continue
            try:
                w = float(it.get("weight", 0.7))
            except (TypeError, ValueError):
                w = 0.7
            out[name] = {"weight": max(0.1, min(1.0, w)), "terms": terms}
        return out

    must = _concepts(must_raw)
    nice = _concepts(nice_raw)
    if not must:  # an LLM spec with no must-haves is useless — fall back
        return None

    exp = obj.get("experience") or {}
    try:
        mn = int(exp.get("min_years", 0)); mx = int(exp.get("max_years", mn + 5))
        lo = int(exp.get("ideal_low", mn)); hi = int(exp.get("ideal_high", mx))
        exp_found = bool(exp)
    except (TypeError, ValueError):
        mn, mx, lo, hi, exp_found = 0, 40, 2, 20, False
    exp_block = {
        "min_years": mn, "max_years": mx, "ideal_low": lo, "ideal_high": hi,
        "soft_floor": max(0, mn - 2), "soft_ceiling": mx + 5,
    }
    sen = str(obj.get("seniority", _seniority(text.lower())))
    exp_block = _experience_policy(exp_block, exp_found=exp_found, seniority=sen)

    locs = obj.get("locations") or {}
    remote = bool(locs.get("remote"))
    cities = sorted({str(c).lower() for c in (locs.get("cities") or [])})
    countries = sorted({str(c).lower() for c in (locs.get("countries") or [])})
    if remote:
        loc_kind, loc_block = "remote", {"best": [], "good": [], "ok_country": []}
    elif cities or countries:
        loc_kind = "best"
        loc_block = {"best": cities, "good": [], "ok_country": countries}
    else:
        loc_kind, loc_block = "none", {"best": [], "good": [], "ok_country": []}

    is_ai = bool(set(must) & _AI_KEYS) or bool(
        {"machine learning", "ml", "nlp", "llm", "embedding", "ranking", "deep learning"}
        & {t for v in must.values() for t in v["terms"]}
    )

    dom_pos = [str(t).lower() for t in (obj.get("domain_positive") or [])]
    dom_neg = [str(t).lower() for t in (obj.get("domain_negative") or [])]

    disq: dict[str, Any] = {
        "title_chaser": {"penalty": 0.8, "max_avg_tenure_months": 14, "min_jobs_for_flag": 4},
    }
    if is_ai:
        disq.update({
            "research_only": {
                "penalty": 0.7,
                "terms": ["phd researcher", "research scientist", "postdoc", "professor",
                          "academia", "lecturer"],
                "production_terms": ["production", "deployed", "shipped", "launched", "users"],
            },
            "wrong_domain": {"penalty": 0.6},
        })

    data = {
        "role": {"title": title or "Custom role (from pasted JD)",
                 "seniority": sen},
        "query_text": str(obj.get("query_text", "")).strip() or _clean_anchor(text, title),
        "experience": exp_block,
        "location": loc_block,
        "must_have_concepts": must,
        "nice_to_have_concepts": nice,
        "domain": {"positive_terms": dom_pos or sorted({t for v in must.values() for t in v["terms"]})[:30],
                   "negative_terms": dom_neg},
        "disqualifiers": disq,
        "career_evidence": {
            "shipped_systems_terms": ["built", "shipped", "launched", "deployed", "designed",
                                      "led", "owned", "architected", "scaled", "delivered",
                                      "managed", "production"],
            "product_signal_terms": ["product", "saas", "platform", "consumer", "b2c", "b2b",
                                     "startup", "scale", "millions", "users", "customers"],
            "target_title_terms": _title_terms(title),
            "role_noun": _role_noun(title),
            "offrole_titles": [],
            "positive_titles": [],
        },
        "weights": _finalize_weights(seniority=sen, exp_found=exp_found,
                                     loc_kind=loc_kind, n_must=len(must)),
        "behavioral": {"modifier_min": 0.60, "modifier_max": 1.15, "stale_days": 120,
                       "good_response_rate": 0.5, "good_notice_days": 30},
    }

    notes = ["Spec extracted by the configured LLM from your job description."]
    avoid = [str(a) for a in (obj.get("avoid") or []) if str(a).strip()]
    if avoid:
        notes.append("LLM-identified non-fit signals: " + "; ".join(avoid[:5]))
    notes.extend(_emphasis_notes(sen))

    spec = RoleSpec(data=data)  # query_text lives in data; override not needed
    return SpecBuildResult(
        spec=spec, method="llm", title=title, notes=notes,
        must_haves=list(must), nice_to_haves=list(nice), is_ai_role=is_ai,
    )


def _parse_json_blob(s: str) -> Any:
    """Parse JSON that may be wrapped in ```fences``` or have leading prose."""
    s = s.strip()
    s = re.sub(r"^```(?:json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # grab the outermost {...}
    start, end = s.find("{"), s.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(s[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None
