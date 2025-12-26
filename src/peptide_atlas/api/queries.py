"""
Standalone query functions for the Peptide Atlas.

These can be used without instantiating PeptideAtlas.

REMINDER: This project is for research and education only.
"""

from typing import List, Optional

import numpy as np

from peptide_atlas.data.schemas import KnowledgeGraph, PeptideNode
from peptide_atlas.constants import EvidenceTier, PeptideClass


def query_by_class(
    kg: KnowledgeGraph,
    peptide_class: str,
) -> List[PeptideNode]:
    """Get peptides by class."""
    target_class = PeptideClass(peptide_class)
    return [p for p in kg.peptides if p.peptide_class == target_class]


def query_by_evidence(
    kg: KnowledgeGraph,
    min_tier: int = 4,
) -> List[PeptideNode]:
    """Get peptides with evidence tier <= min_tier (lower is better)."""
    tier_order = {
        EvidenceTier.TIER_1_APPROVED: 1,
        EvidenceTier.TIER_2_LATE_CLINICAL: 2,
        EvidenceTier.TIER_3_EARLY_CLINICAL: 3,
        EvidenceTier.TIER_4_PRECLINICAL: 4,
        EvidenceTier.TIER_5_MECHANISTIC: 5,
        EvidenceTier.TIER_6_ANECDOTAL: 6,
        EvidenceTier.TIER_UNKNOWN: 7,
    }
    
    return [
        p for p in kg.peptides
        if tier_order.get(p.evidence_tier, 7) <= min_tier
    ]


def query_by_pathway(
    kg: KnowledgeGraph,
    pathway_name: str,
) -> List[PeptideNode]:
    """Get peptides modulating a pathway."""
    pathway_name_lower = pathway_name.lower()
    
    pathway_ids = {
        str(p.id) for p in kg.pathways
        if pathway_name_lower in p.name.lower()
    }
    
    peptide_ids = {
        str(e.source_id) for e in kg.modulates_edges
        if str(e.target_id) in pathway_ids
    }
    
    return [p for p in kg.peptides if str(p.id) in peptide_ids]


def query_by_target(
    kg: KnowledgeGraph,
    target_name: str,
) -> List[PeptideNode]:
    """Get peptides binding a target."""
    target_name_lower = target_name.lower()
    
    target_ids = {
        str(t.id) for t in kg.targets
        if target_name_lower in t.name.lower()
    }
    
    peptide_ids = {
        str(e.source_id) for e in kg.binds_edges
        if str(e.target_id) in target_ids
    }
    
    return [p for p in kg.peptides if str(p.id) in peptide_ids]


def find_similar(
    embeddings: np.ndarray,
    peptide_names: List[str],
    query_name: str,
    k: int = 5,
) -> List[tuple[str, float]]:
    """
    Find similar peptides by embedding distance.
    
    Returns list of (name, distance) tuples.
    """
    name_to_idx = {n.lower(): i for i, n in enumerate(peptide_names)}
    query_idx = name_to_idx.get(query_name.lower())
    
    if query_idx is None:
        return []
    
    query_emb = embeddings[query_idx]
    distances = np.linalg.norm(embeddings - query_emb, axis=1)
    
    sorted_indices = np.argsort(distances)
    
    results = []
    for idx in sorted_indices:
        if idx == query_idx:
            continue
        if len(results) >= k:
            break
        results.append((peptide_names[idx], float(distances[idx])))
    
    return results

