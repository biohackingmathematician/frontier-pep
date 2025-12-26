"""
Knowledge graph export utilities.

Converts the internal KnowledgeGraph format to formats suitable for
PyTorch Geometric, NetworkX, and other graph libraries.

REMINDER: This project is for research and education only.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from loguru import logger

from peptide_atlas.data.schemas import KnowledgeGraph


def export_for_pytorch_geometric(kg: KnowledgeGraph) -> dict[str, Any]:
    """
    Convert KnowledgeGraph to PyTorch Geometric format.
    
    Args:
        kg: KnowledgeGraph instance
        
    Returns:
        Dictionary containing:
            - node_types: list[str] - Type of each node
            - node_ids: list[str] - UUID of each node
            - node_names: list[str] - Human-readable name
            - edge_index: list[list[int]] - [2, num_edges] source/target indices
            - edge_types: list[str] - Type of each edge
            - peptide_mask: list[bool] - True for peptide nodes
            - num_nodes: int - Total node count
            - num_edges: int - Total edge count
    """
    logger.info("Exporting knowledge graph for PyTorch Geometric")
    
    # Build node index mapping
    node_id_to_idx: dict[str, int] = {}
    node_types: list[str] = []
    node_ids: list[str] = []
    node_names: list[str] = []
    peptide_mask: list[bool] = []
    
    idx = 0
    
    # Add peptide nodes first (important for masking)
    for peptide in kg.peptides:
        node_id_to_idx[str(peptide.id)] = idx
        node_types.append("peptide")
        node_ids.append(str(peptide.id))
        node_names.append(peptide.canonical_name)
        peptide_mask.append(True)
        idx += 1
    
    # Add target nodes
    for target in kg.targets:
        node_id_to_idx[str(target.id)] = idx
        node_types.append("target")
        node_ids.append(str(target.id))
        node_names.append(target.name)
        peptide_mask.append(False)
        idx += 1
    
    # Add pathway nodes
    for pathway in kg.pathways:
        node_id_to_idx[str(pathway.id)] = idx
        node_types.append("pathway")
        node_ids.append(str(pathway.id))
        node_names.append(pathway.name)
        peptide_mask.append(False)
        idx += 1
    
    # Add effect domain nodes
    for effect in kg.effect_domains:
        node_id_to_idx[str(effect.id)] = idx
        node_types.append("effect_domain")
        node_ids.append(str(effect.id))
        node_names.append(effect.name)
        peptide_mask.append(False)
        idx += 1
    
    # Add risk nodes
    for risk in kg.risks:
        node_id_to_idx[str(risk.id)] = idx
        node_types.append("risk")
        node_ids.append(str(risk.id))
        node_names.append(risk.name)
        peptide_mask.append(False)
        idx += 1
    
    # Build edge lists
    edge_sources: list[int] = []
    edge_targets: list[int] = []
    edge_types: list[str] = []
    
    # BINDS edges
    for edge in kg.binds_edges:
        src = node_id_to_idx.get(str(edge.source_id))
        dst = node_id_to_idx.get(str(edge.target_id))
        if src is not None and dst is not None:
            edge_sources.append(src)
            edge_targets.append(dst)
            edge_types.append("binds")
            # Add reverse edge for message passing
            edge_sources.append(dst)
            edge_targets.append(src)
            edge_types.append("binds_rev")
    
    # MODULATES edges
    for edge in kg.modulates_edges:
        src = node_id_to_idx.get(str(edge.source_id))
        dst = node_id_to_idx.get(str(edge.target_id))
        if src is not None and dst is not None:
            edge_sources.append(src)
            edge_targets.append(dst)
            edge_types.append("modulates")
            edge_sources.append(dst)
            edge_targets.append(src)
            edge_types.append("modulates_rev")
    
    # ASSOCIATED_WITH_EFFECT edges
    for edge in kg.effect_edges:
        src = node_id_to_idx.get(str(edge.source_id))
        dst = node_id_to_idx.get(str(edge.target_id))
        if src is not None and dst is not None:
            edge_sources.append(src)
            edge_targets.append(dst)
            edge_types.append("associated_with_effect")
            edge_sources.append(dst)
            edge_targets.append(src)
            edge_types.append("associated_with_effect_rev")
    
    # ASSOCIATED_WITH_RISK edges
    for edge in kg.risk_edges:
        src = node_id_to_idx.get(str(edge.source_id))
        dst = node_id_to_idx.get(str(edge.target_id))
        if src is not None and dst is not None:
            edge_sources.append(src)
            edge_targets.append(dst)
            edge_types.append("associated_with_risk")
            edge_sources.append(dst)
            edge_targets.append(src)
            edge_types.append("associated_with_risk_rev")
    
    logger.info(f"Exported: {len(node_ids)} nodes, {len(edge_sources)} edges")
    
    return {
        "node_types": node_types,
        "node_ids": node_ids,
        "node_names": node_names,
        "edge_index": [edge_sources, edge_targets],
        "edge_types": edge_types,
        "peptide_mask": peptide_mask,
        "num_nodes": len(node_ids),
        "num_edges": len(edge_sources),
    }


def get_node_type_mapping(graph_data: dict[str, Any]) -> dict[str, int]:
    """
    Get mapping from node type string to integer index.
    
    Args:
        graph_data: Output from export_for_pytorch_geometric
        
    Returns:
        Dictionary mapping type name to integer
    """
    unique_types = sorted(set(graph_data["node_types"]))
    return {t: i for i, t in enumerate(unique_types)}


def get_edge_type_mapping(graph_data: dict[str, Any]) -> dict[str, int]:
    """
    Get mapping from edge type string to integer index.
    
    Args:
        graph_data: Output from export_for_pytorch_geometric
        
    Returns:
        Dictionary mapping type name to integer
    """
    unique_types = sorted(set(graph_data["edge_types"]))
    return {t: i for i, t in enumerate(unique_types)}


def to_torch_tensors(graph_data: dict[str, Any]) -> dict[str, Any]:
    """
    Convert graph data to PyTorch tensors.
    
    Args:
        graph_data: Output from export_for_pytorch_geometric
        
    Returns:
        Dictionary with torch tensors
    """
    import torch
    
    node_type_map = get_node_type_mapping(graph_data)
    edge_type_map = get_edge_type_mapping(graph_data)
    
    node_types = torch.tensor(
        [node_type_map[t] for t in graph_data["node_types"]],
        dtype=torch.long
    )
    
    edge_index = torch.tensor(
        graph_data["edge_index"],
        dtype=torch.long
    )
    
    edge_types = torch.tensor(
        [edge_type_map[t] for t in graph_data["edge_types"]],
        dtype=torch.long
    )
    
    peptide_mask = torch.tensor(
        graph_data["peptide_mask"],
        dtype=torch.bool
    )
    
    return {
        "node_types": node_types,
        "edge_index": edge_index,
        "edge_types": edge_types,
        "peptide_mask": peptide_mask,
        "num_node_types": len(node_type_map),
        "num_edge_types": len(edge_type_map),
        "node_type_map": node_type_map,
        "edge_type_map": edge_type_map,
    }


def export_peptide_features(kg: KnowledgeGraph) -> dict[str, np.ndarray]:
    """
    Extract peptide features for node initialization.
    
    Args:
        kg: KnowledgeGraph instance
        
    Returns:
        Dictionary with feature arrays
    """
    from peptide_atlas.constants import PeptideClass, EvidenceTier, RegulatoryStatus
    
    # Get enum mappings
    class_map = {c: i for i, c in enumerate(PeptideClass)}
    tier_map = {t: i for i, t in enumerate(EvidenceTier)}
    status_map = {s: i for i, s in enumerate(RegulatoryStatus)}
    
    peptide_classes = []
    evidence_tiers = []
    regulatory_statuses = []
    
    for peptide in kg.peptides:
        peptide_classes.append(class_map[peptide.peptide_class])
        evidence_tiers.append(tier_map[peptide.evidence_tier])
        regulatory_statuses.append(status_map[peptide.regulatory_status])
    
    return {
        "peptide_class": np.array(peptide_classes),
        "evidence_tier": np.array(evidence_tiers),
        "regulatory_status": np.array(regulatory_statuses),
        "num_classes": len(class_map),
        "num_tiers": len(tier_map),
        "num_statuses": len(status_map),
    }


def export_to_networkx(kg: KnowledgeGraph) -> "nx.MultiDiGraph":
    """
    Export KnowledgeGraph to NetworkX format.
    
    Args:
        kg: KnowledgeGraph instance
        
    Returns:
        NetworkX MultiDiGraph
    """
    import networkx as nx
    
    G = nx.MultiDiGraph()
    
    # Add peptide nodes
    for p in kg.peptides:
        G.add_node(
            str(p.id),
            name=p.canonical_name,
            node_type="peptide",
            peptide_class=p.peptide_class.value,
            evidence_tier=p.evidence_tier.value,
        )
    
    # Add target nodes
    for t in kg.targets:
        G.add_node(
            str(t.id),
            name=t.name,
            node_type="target",
            target_type=t.target_type.value,
        )
    
    # Add pathway nodes
    for p in kg.pathways:
        G.add_node(
            str(p.id),
            name=p.name,
            node_type="pathway",
            category=p.category.value,
        )
    
    # Add effect domain nodes
    for e in kg.effect_domains:
        G.add_node(
            str(e.id),
            name=e.name,
            node_type="effect_domain",
        )
    
    # Add risk nodes
    for r in kg.risks:
        G.add_node(
            str(r.id),
            name=r.name,
            node_type="risk",
            severity=r.severity.value,
        )
    
    # Add edges
    for e in kg.binds_edges:
        G.add_edge(str(e.source_id), str(e.target_id), edge_type="binds")
    
    for e in kg.modulates_edges:
        G.add_edge(str(e.source_id), str(e.target_id), edge_type="modulates")
    
    for e in kg.effect_edges:
        G.add_edge(str(e.source_id), str(e.target_id), edge_type="associated_with_effect")
    
    for e in kg.risk_edges:
        G.add_edge(str(e.source_id), str(e.target_id), edge_type="associated_with_risk")
    
    return G
