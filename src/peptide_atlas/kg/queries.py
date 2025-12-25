"""
Knowledge Graph query utilities.

REMINDER: This project is for research and education only.
No dosing, no protocols, no therapeutic recommendations.
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from peptide_atlas.constants import EffectDomain, EvidenceTier, PeptideClass
from peptide_atlas.data.schemas import (
    AssociatedWithEffectEdge,
    AssociatedWithRiskEdge,
    KnowledgeGraph,
    PeptideNode,
)


def get_peptides_by_class(
    kg: KnowledgeGraph, 
    peptide_class: PeptideClass
) -> list[PeptideNode]:
    """Get all peptides of a specific class."""
    return [p for p in kg.peptides if p.peptide_class == peptide_class]


def get_peptides_by_evidence_tier(
    kg: KnowledgeGraph, 
    min_tier: EvidenceTier
) -> list[PeptideNode]:
    """Get peptides with evidence tier at or above the specified level."""
    min_score = min_tier.confidence_score
    return [p for p in kg.peptides if p.evidence_tier.confidence_score >= min_score]


def get_peptide_effects(
    kg: KnowledgeGraph, 
    peptide_id: UUID
) -> list[AssociatedWithEffectEdge]:
    """Get all effect associations for a peptide."""
    return [e for e in kg.effect_edges if e.source_id == peptide_id]


def get_peptide_risks(
    kg: KnowledgeGraph, 
    peptide_id: UUID
) -> list[AssociatedWithRiskEdge]:
    """Get all risk associations for a peptide."""
    return [e for e in kg.risk_edges if e.source_id == peptide_id]


def get_peptides_for_effect(
    kg: KnowledgeGraph, 
    effect_domain: EffectDomain,
    min_evidence_tier: Optional[EvidenceTier] = None
) -> list[tuple[PeptideNode, AssociatedWithEffectEdge]]:
    """
    Get all peptides associated with an effect domain.
    
    Returns tuples of (peptide, effect_edge) for inspection.
    """
    # Find the effect domain node
    effect_node = None
    for e in kg.effect_domains:
        if e.category == effect_domain:
            effect_node = e
            break
    
    if effect_node is None:
        return []
    
    results = []
    for edge in kg.effect_edges:
        if edge.target_id != effect_node.id:
            continue
        
        if min_evidence_tier is not None:
            if edge.evidence_tier.confidence_score < min_evidence_tier.confidence_score:
                continue
        
        peptide = kg.get_node_by_id(edge.source_id)
        if peptide and isinstance(peptide, PeptideNode):
            results.append((peptide, edge))
    
    return results


def get_peptides_sharing_target(
    kg: KnowledgeGraph,
    target_name: str
) -> list[PeptideNode]:
    """Get all peptides that bind to a specific target."""
    # Find target
    target_id = None
    for t in kg.targets:
        if t.name.lower() == target_name.lower():
            target_id = t.id
            break
    
    if target_id is None:
        return []
    
    peptide_ids = {e.source_id for e in kg.binds_edges if e.target_id == target_id}
    return [p for p in kg.peptides if p.id in peptide_ids]


def get_peptides_sharing_pathway(
    kg: KnowledgeGraph,
    pathway_name: str
) -> list[PeptideNode]:
    """Get all peptides that modulate a specific pathway."""
    # Find pathway
    pathway_id = None
    for p in kg.pathways:
        if p.name.lower() == pathway_name.lower():
            pathway_id = p.id
            break
    
    if pathway_id is None:
        return []
    
    peptide_ids = {e.source_id for e in kg.modulates_edges if e.target_id == pathway_id}
    return [p for p in kg.peptides if p.id in peptide_ids]


def compute_peptide_similarity(
    kg: KnowledgeGraph,
    peptide_a: PeptideNode,
    peptide_b: PeptideNode
) -> float:
    """
    Compute similarity between two peptides based on shared relationships.
    
    Returns Jaccard similarity of target/pathway/effect/risk sets.
    """
    def get_relationship_set(peptide_id: UUID) -> set[str]:
        relations = set()
        
        for e in kg.binds_edges:
            if e.source_id == peptide_id:
                relations.add(f"binds:{e.target_id}")
        
        for e in kg.modulates_edges:
            if e.source_id == peptide_id:
                relations.add(f"modulates:{e.target_id}")
        
        for e in kg.effect_edges:
            if e.source_id == peptide_id:
                relations.add(f"effect:{e.target_id}")
        
        for e in kg.risk_edges:
            if e.source_id == peptide_id:
                relations.add(f"risk:{e.target_id}")
        
        return relations
    
    set_a = get_relationship_set(peptide_a.id)
    set_b = get_relationship_set(peptide_b.id)
    
    if not set_a and not set_b:
        return 0.0
    
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    return intersection / union if union > 0 else 0.0


def summarize_knowledge_graph(kg: KnowledgeGraph) -> dict:
    """Generate a summary of the knowledge graph contents."""
    return {
        "nodes": {
            "peptides": len(kg.peptides),
            "targets": len(kg.targets),
            "pathways": len(kg.pathways),
            "effect_domains": len(kg.effect_domains),
            "risks": len(kg.risks),
            "total": kg.node_count,
        },
        "edges": {
            "binds": len(kg.binds_edges),
            "modulates": len(kg.modulates_edges),
            "effects": len(kg.effect_edges),
            "risks": len(kg.risk_edges),
            "total": kg.edge_count,
        },
        "peptides_by_class": {
            pc.value: len([p for p in kg.peptides if p.peptide_class == pc])
            for pc in PeptideClass
        },
        "peptides_by_evidence": {
            et.value: len([p for p in kg.peptides if p.evidence_tier == et])
            for et in EvidenceTier
        },
    }

