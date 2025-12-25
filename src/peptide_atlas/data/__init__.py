"""
Data module for the Peptide Atlas.

Contains data schemas, loaders, validators, and the curated peptide catalog.

REMINDER: This project is for research and education only.
No dosing, no protocols, no therapeutic recommendations.
"""

from peptide_atlas.data.schemas import (
    PeptideNode,
    TargetNode,
    PathwayNode,
    EffectDomainNode,
    RiskNode,
    EvidenceSourceNode,
    BindsEdge,
    ModulatesEdge,
    AssociatedWithEffectEdge,
    AssociatedWithRiskEdge,
    KnowledgeGraph,
)
from peptide_atlas.data.peptide_catalog import (
    get_curated_peptides,
    get_peptide_count_by_class,
    get_peptide_count_by_evidence_tier,
)

__all__ = [
    # Node schemas
    "PeptideNode",
    "TargetNode",
    "PathwayNode",
    "EffectDomainNode",
    "RiskNode",
    "EvidenceSourceNode",
    # Edge schemas
    "BindsEdge",
    "ModulatesEdge",
    "AssociatedWithEffectEdge",
    "AssociatedWithRiskEdge",
    # Container
    "KnowledgeGraph",
    # Catalog functions
    "get_curated_peptides",
    "get_peptide_count_by_class",
    "get_peptide_count_by_evidence_tier",
]

