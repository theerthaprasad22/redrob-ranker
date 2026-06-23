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
        """
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
