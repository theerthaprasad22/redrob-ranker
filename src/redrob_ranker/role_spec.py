"""
role_spec.py — load and expose the structured job-description understanding.

This is the bridge between the JD (a human document) and the scoring code. All
recruiter "intent" lives in config/role_spec.yaml; this module just loads it and
offers a couple of convenience views (e.g. a flattened set of all must-have
terms used to build the retrieval query).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "role_spec.yaml"


@dataclass
class RoleSpec:
    data: dict[str, Any] = field(default_factory=dict)
    # Optional natural-language anchor for semantic similarity. When a spec is
    # built dynamically from a pasted job description (see jd_spec.py) we set
    # this to the role's own text, so candidates are scored against THAT role
    # rather than the hard-coded default below. Left as None for the bundled
    # config/role_spec.yaml, which keeps its original behaviour exactly.
    query_text_override: str | None = None

    # ---- raw section accessors --------------------------------------------
    @property
    def experience(self) -> dict[str, Any]:
        return self.data.get("experience", {})

    @property
    def location(self) -> dict[str, Any]:
        return self.data.get("location", {})

    @property
    def must_have(self) -> dict[str, Any]:
        return self.data.get("must_have_concepts", {})

    @property
    def nice_to_have(self) -> dict[str, Any]:
        return self.data.get("nice_to_have_concepts", {})

    @property
    def domain(self) -> dict[str, Any]:
        return self.data.get("domain", {})

    @property
    def disqualifiers(self) -> dict[str, Any]:
        return self.data.get("disqualifiers", {})

    @property
    def career_evidence(self) -> dict[str, Any]:
        return self.data.get("career_evidence", {})

    @property
    def weights(self) -> dict[str, float]:
        return self.data.get("weights", {})

    @property
    def behavioral(self) -> dict[str, Any]:
        return self.data.get("behavioral", {})

    # ---- derived views -----------------------------------------------------
    def all_must_have_terms(self) -> list[str]:
        terms: list[str] = []
        for c in self.must_have.values():
            terms.extend(c.get("terms", []))
        return terms

    def query_text(self) -> str:
        """A natural-language query that represents the *meaning* of the role.

        Used as the anchor for semantic similarity. Written in plain language
        (not a keyword dump) so it aligns with how strong candidates actually
        describe their work.

        Resolution order:
          1. an explicit ``query_text_override`` (set when the spec is built
             from a pasted JD) — this is what makes ranking role-aware;
          2. a ``query_text`` key inside the loaded data (e.g. an LLM-generated
             spec may carry its own anchor);
          3. the hard-coded default for the challenge's Senior-AI-Engineer role.
        """
        if self.query_text_override and self.query_text_override.strip():
            return self.query_text_override.strip()
        anchor = self.data.get("query_text")
        if isinstance(anchor, str) and anchor.strip():
            return anchor.strip()
        return (
            "Senior AI / machine learning engineer for a product company. "
            "Has built and shipped end-to-end ranking, search, retrieval, and "
            "recommendation systems to real users at scale. Strong with "
            "embeddings-based semantic retrieval (sentence-transformers, BGE, E5), "
            "vector databases and hybrid search (FAISS, Pinecone, Elasticsearch), "
            "and rigorous ranking evaluation (NDCG, MRR, MAP, A/B testing). "
            "Production applied-ML and information-retrieval experience, strong "
            "Python, writes code. Not a pure researcher, not a keyword-stuffer, "
            "not a title-chaser, not primarily computer vision or speech."
        )

    @classmethod
    def load(cls, path: str | Path | None = None) -> "RoleSpec":
        p = Path(path) if path else DEFAULT_PATH
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(data=data)
