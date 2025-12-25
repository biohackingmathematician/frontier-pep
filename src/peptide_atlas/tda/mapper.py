"""
Mapper algorithm implementation for the Peptide Atlas.

Mapper produces a graph that captures the topological structure
of high-dimensional data.

REMINDER: This project is for research and education only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from loguru import logger

try:
    import kmapper as km
    from sklearn.cluster import DBSCAN
    HAS_KMAPPER = True
except ImportError:
    HAS_KMAPPER = False
    logger.warning("kmapper not installed. Mapper functionality will be limited.")

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False


@dataclass
class MapperConfig:
    """Configuration for the Mapper algorithm."""
    
    # Cover parameters
    n_cubes: int = 15
    overlap_perc: float = 0.5
    
    # Clustering
    clustering_algorithm: str = "hdbscan"  # or "dbscan"
    min_cluster_size: int = 3
    min_samples: int = 2
    eps: float = 0.5  # for DBSCAN
    
    # Graph construction
    min_intersection: int = 1


@dataclass
class MapperResult:
    """Result from Mapper algorithm."""
    
    graph: dict[str, Any]  # The Mapper graph
    node_ids: list[str]  # Node identifiers
    node_members: dict[str, list[int]]  # Point indices in each node
    edges: list[tuple[str, str]]  # Edge list
    
    # Metadata
    num_nodes: int = 0
    num_edges: int = 0
    
    def __post_init__(self):
        self.num_nodes = len(self.node_ids)
        self.num_edges = len(self.edges)


class MapperPipeline:
    """
    Complete Mapper pipeline for the Peptide Atlas.
    
    Applies the Mapper algorithm to peptide embeddings to reveal
    topological structure and clustering patterns.
    """
    
    def __init__(self, config: Optional[MapperConfig] = None):
        """
        Initialize Mapper pipeline.
        
        Args:
            config: Mapper configuration
        """
        if not HAS_KMAPPER:
            raise ImportError(
                "kmapper is required for Mapper functionality. "
                "Install with: pip install kmapper"
            )
        
        self.config = config or MapperConfig()
        self.mapper = km.KeplerMapper(verbose=1)
    
    def fit(
        self,
        X: np.ndarray,
        lens: Optional[np.ndarray] = None,
        lens_fn: Optional[Callable] = None,
    ) -> MapperResult:
        """
        Apply Mapper to data.
        
        Args:
            X: Data matrix [n_samples, n_features]
            lens: Pre-computed lens values [n_samples, lens_dim]
            lens_fn: Function to compute lens (if lens not provided)
            
        Returns:
            MapperResult with graph structure
        """
        # Compute lens if not provided
        if lens is None:
            if lens_fn is not None:
                lens = lens_fn(X)
            else:
                # Default: use UMAP projection
                lens = self._default_lens(X)
        
        logger.info(f"Running Mapper with lens shape {lens.shape}")
        
        # Get clustering algorithm
        clusterer = self._get_clusterer()
        
        # Build Mapper graph
        graph = self.mapper.map(
            lens,
            X,
            clusterer=clusterer,
            cover=km.Cover(
                n_cubes=self.config.n_cubes,
                perc_overlap=self.config.overlap_perc,
            ),
        )
        
        # Extract structure
        node_ids = list(graph["nodes"].keys())
        node_members = {k: v for k, v in graph["nodes"].items()}
        
        # Get edges
        edges = []
        for link in graph["links"].items():
            src = link[0]
            for dst in link[1]:
                edges.append((src, dst))
        
        result = MapperResult(
            graph=graph,
            node_ids=node_ids,
            node_members=node_members,
            edges=edges,
        )
        
        logger.info(f"Mapper complete: {result.num_nodes} nodes, {result.num_edges} edges")
        
        return result
    
    def _default_lens(self, X: np.ndarray) -> np.ndarray:
        """Compute default lens using UMAP."""
        try:
            import umap
            
            reducer = umap.UMAP(
                n_neighbors=15,
                min_dist=0.1,
                n_components=2,
                metric="cosine",
                random_state=42,
            )
            return reducer.fit_transform(X)
        except ImportError:
            logger.warning("UMAP not available, using PCA")
            from sklearn.decomposition import PCA
            return PCA(n_components=2).fit_transform(X)
    
    def _get_clusterer(self):
        """Get clustering algorithm."""
        if self.config.clustering_algorithm == "hdbscan" and HAS_HDBSCAN:
            return hdbscan.HDBSCAN(
                min_cluster_size=self.config.min_cluster_size,
                min_samples=self.config.min_samples,
            )
        else:
            return DBSCAN(
                eps=self.config.eps,
                min_samples=self.config.min_samples,
            )
    
    def visualize_html(
        self,
        result: MapperResult,
        output_path: str,
        title: str = "Peptide Atlas Mapper",
        color_values: Optional[np.ndarray] = None,
        color_function_name: str = "Evidence Tier",
        node_labels: Optional[dict[str, str]] = None,
    ) -> str:
        """
        Generate interactive HTML visualization.
        
        Args:
            result: Mapper result
            output_path: Path for output HTML
            title: Visualization title
            color_values: Values for coloring nodes
            color_function_name: Name of color function
            node_labels: Optional custom labels for nodes
            
        Returns:
            Path to generated HTML
        """
        # Add disclaimer to title
        full_title = f"{title} - RESEARCH USE ONLY"
        
        html = self.mapper.visualize(
            result.graph,
            path_html=output_path,
            title=full_title,
            color_values=color_values,
            color_function_name=color_function_name,
            custom_tooltips=node_labels,
        )
        
        logger.info(f"Saved Mapper visualization to {output_path}")
        return output_path


def create_mapper_from_embeddings(
    embeddings: np.ndarray,
    labels: Optional[list[str]] = None,
    config: Optional[MapperConfig] = None,
) -> MapperResult:
    """
    Convenience function to run Mapper on embeddings.
    
    Args:
        embeddings: Peptide embeddings [n_peptides, dim]
        labels: Optional peptide labels
        config: Mapper configuration
        
    Returns:
        MapperResult
    """
    pipeline = MapperPipeline(config)
    return pipeline.fit(embeddings)

