"""
TDA result analysis and interpretation.

REMINDER: This project is for research and education only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from loguru import logger


@dataclass
class ClusterInfo:
    """Information about a Mapper cluster."""
    
    node_id: str
    member_indices: list[int]
    size: int
    
    # Computed properties
    dominant_class: Optional[str] = None
    class_distribution: Optional[dict[str, float]] = None
    mean_evidence_tier: Optional[float] = None
    
    # Connectivity
    neighbors: list[str] = None
    degree: int = 0


@dataclass
class BridgeInfo:
    """Information about a bridge between clusters."""
    
    node_a: str
    node_b: str
    
    # Shared members
    shared_indices: list[int]
    shared_count: int
    
    # Classes involved
    classes_a: Optional[list[str]] = None
    classes_b: Optional[list[str]] = None


@dataclass 
class MapperAnalysis:
    """Complete analysis of a Mapper graph."""
    
    # Cluster information
    clusters: list[ClusterInfo]
    num_clusters: int
    
    # Bridge information
    bridges: list[BridgeInfo]
    num_bridges: int
    
    # Global statistics
    total_points: int
    avg_cluster_size: float
    max_cluster_size: int
    
    # Connectivity
    num_connected_components: int
    largest_component_size: int


def analyze_mapper_graph(
    mapper_result: Any,  # MapperResult
    labels: Optional[np.ndarray] = None,
    evidence_tiers: Optional[np.ndarray] = None,
    class_names: Optional[list[str]] = None,
) -> MapperAnalysis:
    """
    Analyze a Mapper graph result.
    
    Args:
        mapper_result: Result from MapperPipeline
        labels: Optional class labels for each point
        evidence_tiers: Optional evidence tier for each point
        class_names: Optional names for class labels
        
    Returns:
        MapperAnalysis with cluster and bridge information
    """
    logger.info("Analyzing Mapper graph...")
    
    clusters = []
    all_members = set()
    
    # Build adjacency
    adjacency: dict[str, list[str]] = {nid: [] for nid in mapper_result.node_ids}
    for src, dst in mapper_result.edges:
        adjacency[src].append(dst)
        adjacency[dst].append(src)
    
    # Analyze each cluster
    for node_id in mapper_result.node_ids:
        members = mapper_result.node_members[node_id]
        all_members.update(members)
        
        cluster = ClusterInfo(
            node_id=node_id,
            member_indices=members,
            size=len(members),
            neighbors=adjacency[node_id],
            degree=len(adjacency[node_id]),
        )
        
        # Add class information if available
        if labels is not None:
            member_labels = labels[members]
            unique, counts = np.unique(member_labels, return_counts=True)
            
            # Dominant class
            dominant_idx = np.argmax(counts)
            if class_names is not None and unique[dominant_idx] < len(class_names):
                cluster.dominant_class = class_names[unique[dominant_idx]]
            else:
                cluster.dominant_class = str(unique[dominant_idx])
            
            # Distribution
            total = len(members)
            cluster.class_distribution = {
                str(u): float(c) / total for u, c in zip(unique, counts)
            }
        
        # Add evidence tier if available
        if evidence_tiers is not None:
            cluster.mean_evidence_tier = float(np.mean(evidence_tiers[members]))
        
        clusters.append(cluster)
    
    # Analyze bridges
    bridges = []
    seen_edges = set()
    
    for src, dst in mapper_result.edges:
        edge_key = tuple(sorted([src, dst]))
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        
        members_a = set(mapper_result.node_members[src])
        members_b = set(mapper_result.node_members[dst])
        shared = list(members_a & members_b)
        
        bridge = BridgeInfo(
            node_a=src,
            node_b=dst,
            shared_indices=shared,
            shared_count=len(shared),
        )
        
        if labels is not None:
            bridge.classes_a = list(set(labels[list(members_a)]))
            bridge.classes_b = list(set(labels[list(members_b)]))
        
        bridges.append(bridge)
    
    # Compute connected components (simple BFS)
    visited = set()
    components = []
    
    for start in mapper_result.node_ids:
        if start in visited:
            continue
        
        component = []
        queue = [start]
        
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            queue.extend(n for n in adjacency[node] if n not in visited)
        
        components.append(component)
    
    # Summary statistics
    cluster_sizes = [c.size for c in clusters]
    
    analysis = MapperAnalysis(
        clusters=clusters,
        num_clusters=len(clusters),
        bridges=bridges,
        num_bridges=len(bridges),
        total_points=len(all_members),
        avg_cluster_size=np.mean(cluster_sizes) if cluster_sizes else 0,
        max_cluster_size=max(cluster_sizes) if cluster_sizes else 0,
        num_connected_components=len(components),
        largest_component_size=max(len(c) for c in components) if components else 0,
    )
    
    logger.info(
        f"Analysis complete: {analysis.num_clusters} clusters, "
        f"{analysis.num_bridges} bridges, "
        f"{analysis.num_connected_components} components"
    )
    
    return analysis


def find_mechanistic_bridges(
    analysis: MapperAnalysis,
    min_shared: int = 1,
) -> list[BridgeInfo]:
    """
    Find bridges that connect different peptide classes.
    
    These bridges may represent shared mechanisms or transition regions.
    """
    cross_class_bridges = []
    
    for bridge in analysis.bridges:
        if bridge.classes_a is None or bridge.classes_b is None:
            continue
        
        # Check if classes differ
        if set(bridge.classes_a) != set(bridge.classes_b):
            if bridge.shared_count >= min_shared:
                cross_class_bridges.append(bridge)
    
    return cross_class_bridges


def summarize_cluster_by_class(
    analysis: MapperAnalysis,
) -> dict[str, list[ClusterInfo]]:
    """
    Group clusters by their dominant class.
    """
    by_class: dict[str, list[ClusterInfo]] = {}
    
    for cluster in analysis.clusters:
        if cluster.dominant_class:
            if cluster.dominant_class not in by_class:
                by_class[cluster.dominant_class] = []
            by_class[cluster.dominant_class].append(cluster)
    
    return by_class

