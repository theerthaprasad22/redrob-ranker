"""Redrob candidate ranker — a hybrid, offline-capable, recruiter-style ranker."""

from .pipeline import RankedRow, rank_candidates
from .role_spec import RoleSpec
from .schema import Candidate, load_candidates, stream_candidates

__all__ = [
    "Candidate",
    "RoleSpec",
    "RankedRow",
    "rank_candidates",
    "load_candidates",
    "stream_candidates",
]

__version__ = "1.0.0"
