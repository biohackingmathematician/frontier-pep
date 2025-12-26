"""
Mapper algorithm pipeline for the Peptide Atlas.

The Mapper algorithm provides a way to visualize high-dimensional data
by creating a simplicial complex that captures its topological structure.

REMINDER: This project is for research and education only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
from loguru import logger

try:
    import kmapper as km
    HAS_KMAPPER = True
except ImportError:
    HAS_KMAPPER = False
    logger.warning("kmapper not installed. Install with: pip install kmapper")


@dataclass
class MapperConfig:
    """Configuration for the Mapper algorithm."""
    
    # Cover parameters
    n_cubes: int = 15
    overlap_perc: float = 0.5
    
    # Clustering parameters
    clustering_algorithm: str = "hdbscan"
    min_cluster_size: int = 2
    min_samples: int = 2
    
    # Filter/lens parameters
    default_filter: str = "l2norm"
    
    # Visualization
    colormap: str = "viridis"


@dataclass
class MapperResult:
    """Result from Mapper algorithm."""
    
    # The raw kmapper graph
    graph: dict[str, Any]
    
    # Extracted structure
    node_ids: list[str] = field(default_factory=list)
    node_members: dict[str, list[int]] = field(default_factory=dict)
    edges: list[tuple[str, str]] = field(default_factory=list)
    
    # Metadata
    num_points: int = 0
    filter_name: str = ""
    config: Optional[MapperConfig] = None
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes in the Mapper graph."""
        return len(self.node_ids)
    
    @property
    def num_edges(self) -> int:
        """Number of edges in the Mapper graph."""
        return len(self.edges)
    
    def get_node_size(self, node_id: str) -> int:
        """Get the size (member count) of a node."""
        return len(self.node_members.get(node_id, []))
    
    def get_cluster_sizes(self) -> list[int]:
        """Get list of cluster sizes."""
        return [len(members) for members in self.node_members.values()]


class MapperPipeline:
    """
    Complete Mapper pipeline for topological data analysis.
    
    The Mapper algorithm:
    1. Projects data to a lower dimension using a filter function (lens)
    2. Covers the filter space with overlapping intervals
    3. Clusters points within each interval
    4. Creates a graph where nodes are clusters and edges connect overlapping clusters
    """
    
    def __init__(self, config: Optional[MapperConfig] = None):
        """
        Initialize the Mapper pipeline.
        
        Args:
            config: Mapper configuration
            
        Raises:
            ImportError: If kmapper is not installed
        """
        if not HAS_KMAPPER:
            raise ImportError(
                "kmapper is required for the Mapper algorithm. "
                "Install with: pip install kmapper"
            )
        
        self.config = config or MapperConfig()
        self.mapper = km.KeplerMapper(verbose=0)
        self._last_lens: Optional[np.ndarray] = None
    
    def fit(
        self,
        X: np.ndarray,
        lens: Optional[np.ndarray] = None,
        lens_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> MapperResult:
        """
        Apply Mapper algorithm to data.
        
        Args:
            X: Data matrix [n_samples, n_features]
            lens: Pre-computed filter values [n_samples, n_components]
            lens_fn: Function to compute filter from X
            
        Returns:
            MapperResult with the computed graph
        """
        logger.info(f"Running Mapper on {X.shape[0]} points")
        
        filter_name = self.config.default_filter
        
        # Compute lens if not provided
        if lens is None:
            if lens_fn is not None:
                lens = lens_fn(X)
                filter_name = "custom"
            else:
                # Default: use L2 norm projection
                lens = self.mapper.fit_transform(X, projection="l2norm")
                filter_name = "l2norm"
        
        self._last_lens = lens
        
        # Get clustering algorithm
        clusterer = self._get_clusterer()
        
        # Build the Mapper graph
        graph = self.mapper.map(
            lens,
            X,
            cover=km.Cover(
                n_cubes=self.config.n_cubes,
                perc_overlap=self.config.overlap_perc,
            ),
            clusterer=clusterer,
        )
        
        # Extract graph structure
        node_ids = list(graph["nodes"].keys())
        node_members = {
            node_id: list(members) 
            for node_id, members in graph["nodes"].items()
        }
        
        # Extract edges from links
        edges = []
        if "links" in graph:
            for src_node, dst_nodes in graph["links"].items():
                for dst_node in dst_nodes:
                    if (dst_node, src_node) not in edges:  # Avoid duplicates
                        edges.append((src_node, dst_node))
        
        result = MapperResult(
            graph=graph,
            node_ids=node_ids,
            node_members=node_members,
            edges=edges,
            num_points=X.shape[0],
            filter_name=filter_name,
            config=self.config,
        )
        
        logger.info(f"Mapper complete: {result.num_nodes} nodes, {result.num_edges} edges")
        
        return result
    
    def _get_clusterer(self):
        """Get clustering algorithm instance."""
        if self.config.clustering_algorithm == "hdbscan":
            try:
                import hdbscan
                return hdbscan.HDBSCAN(
                    min_cluster_size=self.config.min_cluster_size,
                    min_samples=self.config.min_samples,
                )
            except ImportError:
                logger.warning("HDBSCAN not available, falling back to DBSCAN")
        
        # Fallback to DBSCAN
        from sklearn.cluster import DBSCAN
        return DBSCAN(eps=0.5, min_samples=self.config.min_samples)
    
    def visualize_html(
        self,
        result: MapperResult,
        output_path: Union[str, Path],
        title: str = "Peptide Atlas Mapper",
        color_values: Optional[np.ndarray] = None,
        color_function_name: str = "mean",
        custom_tooltips: Optional[list[str]] = None,
        **kwargs,
    ) -> str:
        """
        Generate interactive HTML visualization of Mapper graph.
        
        Args:
            result: MapperResult from fit()
            output_path: Path to save HTML file
            title: Title for the visualization
            color_values: Values to use for coloring nodes
            color_function_name: How to aggregate colors ("mean", "std", "sum")
            custom_tooltips: Custom tooltip text for each point
            **kwargs: Additional arguments to kmapper.visualize()
            
        Returns:
            Path to the saved HTML file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add disclaimer to title
        title_with_disclaimer = f"{title} (Research Use Only)"
        
        html = self.mapper.visualize(
            result.graph,
            path_html=str(output_path),
            title=title_with_disclaimer,
            color_values=color_values,
            color_function_name=color_function_name,
            custom_tooltips=custom_tooltips,
            **kwargs,
        )
        
        logger.info(f"Saved Mapper visualization to {output_path}")
        
        return str(output_path)
    
    def get_lens(self) -> Optional[np.ndarray]:
        """Return the last computed lens/filter values."""
        return self._last_lens


def create_mapper_from_config(config_dict: dict[str, Any]) -> MapperPipeline:
    """
    Create MapperPipeline from configuration dictionary.
    
    Args:
        config_dict: Dictionary with mapper configuration
        
    Returns:
        Configured MapperPipeline
    """
    mapper_config = config_dict.get("mapper", {})
    cover_config = mapper_config.get("cover", {})
    clustering_config = mapper_config.get("clustering", {})
    
    config = MapperConfig(
        n_cubes=cover_config.get("n_cubes", 15),
        overlap_perc=cover_config.get("overlap_perc", 0.5),
        clustering_algorithm=clustering_config.get("algorithm", "hdbscan"),
        min_cluster_size=clustering_config.get("min_cluster_size", 2),
        min_samples=clustering_config.get("min_samples", 2),
    )
    
    return MapperPipeline(config)
