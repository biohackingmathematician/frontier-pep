"""
Query API for the Peptide Atlas.

Provides programmatic access to the knowledge graph and embeddings.

REMINDER: This project is for research and education only.
"""

from peptide_atlas.api.atlas import PeptideAtlas
from peptide_atlas.api.queries import (
    query_by_class,
    query_by_evidence,
    query_by_pathway,
    query_by_target,
    find_similar,
)

__all__ = [
    "PeptideAtlas",
    "query_by_class",
    "query_by_evidence",
    "query_by_pathway",
    "query_by_target",
    "find_similar",
]

