"""
Data module for the Peptide Atlas.

REMINDER: This project is for research and education only.
"""

from peptide_atlas.data.schemas import (
    PeptideNode,
    TargetNode,
    PathwayNode,
    EffectDomainNode,
    RiskNode,
    KnowledgeGraph,
)
from peptide_atlas.data.peptide_catalog import (
    get_curated_peptides,
    get_peptide_count_by_class,
    get_peptide_count_by_evidence_tier,
)
from peptide_atlas.data.loaders import (
    load_knowledge_graph,
    save_knowledge_graph,
    load_yaml_config,
    load_embeddings,
    save_embeddings,
)

__all__ = [
    # Schemas
    "PeptideNode",
    "TargetNode",
    "PathwayNode",
    "EffectDomainNode",
    "RiskNode",
    "KnowledgeGraph",
    # Catalog
    "get_curated_peptides",
    "get_peptide_count_by_class",
    "get_peptide_count_by_evidence_tier",
    # Loaders
    "load_knowledge_graph",
    "save_knowledge_graph",
    "load_yaml_config",
    "load_embeddings",
    "save_embeddings",
]
