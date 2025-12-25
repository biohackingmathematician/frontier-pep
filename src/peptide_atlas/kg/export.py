"""
Knowledge Graph export utilities.

Supports export to various formats for analysis and visualization.

REMINDER: This project is for research and education only.
No dosing, no protocols, no therapeutic recommendations.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import networkx as nx
from loguru import logger

from peptide_atlas.data.schemas import KnowledgeGraph
from peptide_atlas.kg.builder import KnowledgeGraphBuilder


def export_to_json(kg: KnowledgeGraph, path: Path) -> None:
    """Export knowledge graph to JSON format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(kg.model_dump(mode="json"), f, indent=2, default=str)
    
    logger.info(f"Exported knowledge graph to JSON: {path}")


def export_to_networkx(kg: KnowledgeGraph) -> nx.MultiDiGraph:
    """Export knowledge graph to NetworkX format."""
    builder = KnowledgeGraphBuilder()
    builder.kg = kg
    return builder.to_networkx()


def export_to_gexf(kg: KnowledgeGraph, path: Path) -> None:
    """Export knowledge graph to GEXF format for Gephi."""
    G = export_to_networkx(kg)
    
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(G, str(path))
    
    logger.info(f"Exported knowledge graph to GEXF: {path}")


def export_to_graphml(kg: KnowledgeGraph, path: Path) -> None:
    """Export knowledge graph to GraphML format."""
    G = export_to_networkx(kg)
    
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, str(path))
    
    logger.info(f"Exported knowledge graph to GraphML: {path}")


def export_nodes_to_csv(kg: KnowledgeGraph, output_dir: Path) -> None:
    """Export all nodes to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Peptides
    peptide_path = output_dir / "peptides.csv"
    with open(peptide_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "canonical_name", "synonyms", "peptide_class", 
            "subclass", "regulatory_status", "evidence_tier", "description"
        ])
        for p in kg.peptides:
            writer.writerow([
                str(p.id),
                p.canonical_name,
                "|".join(p.synonyms),
                p.peptide_class.value,
                p.subclass or "",
                p.regulatory_status.value,
                p.evidence_tier.value,
                p.description or "",
            ])
    
    # Targets
    target_path = output_dir / "targets.csv"
    with open(target_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name", "target_type", "description"])
        for t in kg.targets:
            writer.writerow([
                str(t.id),
                t.name,
                t.target_type.value,
                t.description or "",
            ])
    
    # Pathways
    pathway_path = output_dir / "pathways.csv"
    with open(pathway_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name", "category", "description"])
        for p in kg.pathways:
            writer.writerow([
                str(p.id),
                p.name,
                p.category.value,
                p.description or "",
            ])
    
    # Effect domains
    effect_path = output_dir / "effect_domains.csv"
    with open(effect_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name", "category", "description"])
        for e in kg.effect_domains:
            writer.writerow([
                str(e.id),
                e.name,
                e.category.value,
                e.description or "",
            ])
    
    # Risks
    risk_path = output_dir / "risks.csv"
    with open(risk_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name", "category", "severity", "description"])
        for r in kg.risks:
            writer.writerow([
                str(r.id),
                r.name,
                r.category.value,
                r.severity_typical.value,
                r.description or "",
            ])
    
    logger.info(f"Exported nodes to CSV files in: {output_dir}")


def export_edges_to_csv(kg: KnowledgeGraph, output_dir: Path) -> None:
    """Export all edges to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Binds edges
    binds_path = output_dir / "binds_edges.csv"
    with open(binds_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_id", "target_id", "binding_type", "confidence"])
        for e in kg.binds_edges:
            writer.writerow([
                str(e.source_id),
                str(e.target_id),
                e.binding_type.value,
                e.confidence.value,
            ])
    
    # Modulates edges
    modulates_path = output_dir / "modulates_edges.csv"
    with open(modulates_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_id", "target_id", "direction", "magnitude", "confidence"])
        for e in kg.modulates_edges:
            writer.writerow([
                str(e.source_id),
                str(e.target_id),
                e.direction,
                e.magnitude or "",
                e.confidence.value,
            ])
    
    # Effect edges
    effect_path = output_dir / "effect_edges.csv"
    with open(effect_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source_id", "target_id", "direction", "evidence_tier", "confidence"
        ])
        for e in kg.effect_edges:
            writer.writerow([
                str(e.source_id),
                str(e.target_id),
                e.direction.value,
                e.evidence_tier.value,
                e.confidence.value,
            ])
    
    # Risk edges
    risk_path = output_dir / "risk_edges.csv"
    with open(risk_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source_id", "target_id", "frequency", "evidence_tier", "confidence"
        ])
        for e in kg.risk_edges:
            writer.writerow([
                str(e.source_id),
                str(e.target_id),
                e.frequency or "",
                e.evidence_tier.value,
                e.confidence.value,
            ])
    
    logger.info(f"Exported edges to CSV files in: {output_dir}")


def export_for_pytorch_geometric(kg: KnowledgeGraph) -> dict[str, Any]:
    """
    Export knowledge graph in format suitable for PyTorch Geometric.
    
    Returns dictionary with node features, edge indices, and edge types.
    """
    # Create node type mappings
    node_id_to_idx: dict[str, int] = {}
    node_types: list[str] = []
    node_features: list[dict] = []
    
    idx = 0
    
    # Add peptides
    for p in kg.peptides:
        node_id_to_idx[str(p.id)] = idx
        node_types.append("peptide")
        node_features.append({
            "name": p.canonical_name,
            "peptide_class": p.peptide_class.value,
            "evidence_tier": p.evidence_tier.value,
        })
        idx += 1
    
    # Add targets
    for t in kg.targets:
        node_id_to_idx[str(t.id)] = idx
        node_types.append("target")
        node_features.append({
            "name": t.name,
            "target_type": t.target_type.value,
        })
        idx += 1
    
    # Add pathways
    for p in kg.pathways:
        node_id_to_idx[str(p.id)] = idx
        node_types.append("pathway")
        node_features.append({
            "name": p.name,
            "category": p.category.value,
        })
        idx += 1
    
    # Add effect domains
    for e in kg.effect_domains:
        node_id_to_idx[str(e.id)] = idx
        node_types.append("effect_domain")
        node_features.append({
            "name": e.name,
            "category": e.category.value,
        })
        idx += 1
    
    # Add risks
    for r in kg.risks:
        node_id_to_idx[str(r.id)] = idx
        node_types.append("risk")
        node_features.append({
            "name": r.name,
            "category": r.category.value,
            "severity": r.severity_typical.value,
        })
        idx += 1
    
    # Build edge indices
    edge_index: list[list[int]] = [[], []]  # [sources, targets]
    edge_types: list[str] = []
    edge_attrs: list[dict] = []
    
    for e in kg.binds_edges:
        src = node_id_to_idx.get(str(e.source_id))
        tgt = node_id_to_idx.get(str(e.target_id))
        if src is not None and tgt is not None:
            edge_index[0].append(src)
            edge_index[1].append(tgt)
            edge_types.append("binds")
            edge_attrs.append({"binding_type": e.binding_type.value})
    
    for e in kg.modulates_edges:
        src = node_id_to_idx.get(str(e.source_id))
        tgt = node_id_to_idx.get(str(e.target_id))
        if src is not None and tgt is not None:
            edge_index[0].append(src)
            edge_index[1].append(tgt)
            edge_types.append("modulates")
            edge_attrs.append({"direction": e.direction})
    
    for e in kg.effect_edges:
        src = node_id_to_idx.get(str(e.source_id))
        tgt = node_id_to_idx.get(str(e.target_id))
        if src is not None and tgt is not None:
            edge_index[0].append(src)
            edge_index[1].append(tgt)
            edge_types.append("associated_with_effect")
            edge_attrs.append({"evidence_tier": e.evidence_tier.value})
    
    for e in kg.risk_edges:
        src = node_id_to_idx.get(str(e.source_id))
        tgt = node_id_to_idx.get(str(e.target_id))
        if src is not None and tgt is not None:
            edge_index[0].append(src)
            edge_index[1].append(tgt)
            edge_types.append("associated_with_risk")
            edge_attrs.append({"evidence_tier": e.evidence_tier.value})
    
    return {
        "num_nodes": len(node_types),
        "num_edges": len(edge_types),
        "node_id_to_idx": node_id_to_idx,
        "node_types": node_types,
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_types": edge_types,
        "edge_attrs": edge_attrs,
    }

