"""
Filter functions for Mapper algorithm.

These functions project high-dimensional embeddings to lower dimensions
for use as the "lens" in the Mapper algorithm.

REMINDER: This project is for research and education only.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from loguru import logger


def pca_filter(
    X: np.ndarray,
    n_components: int = 2,
) -> np.ndarray:
    """
    PCA projection as filter function.
    
    Args:
        X: Data matrix [n_samples, n_features]
        n_components: Number of components to project to
        
    Returns:
        Projected data [n_samples, n_components]
    """
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=n_components)
    result = pca.fit_transform(X)
    
    logger.debug(f"PCA filter: explained variance = {pca.explained_variance_ratio_.sum():.3f}")
    
    return result


def umap_filter(
    X: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """
    UMAP projection as filter function.
    
    Args:
        X: Data matrix [n_samples, n_features]
        n_components: Number of dimensions
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        metric: Distance metric
        random_state: Random seed
        
    Returns:
        Projected data [n_samples, n_components]
    """
    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn required: pip install umap-learn")
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    
    return reducer.fit_transform(X)


def density_filter(
    X: np.ndarray,
    n_neighbors: int = 5,
) -> np.ndarray:
    """
    Local density as filter function.
    
    Computes density as inverse of mean distance to k nearest neighbors.
    
    Args:
        X: Data matrix [n_samples, n_features]
        n_neighbors: Number of neighbors for density estimation
        
    Returns:
        Density values [n_samples, 1]
    """
    from sklearn.neighbors import NearestNeighbors
    
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)  # +1 to exclude self
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    
    # Density = inverse of mean distance (exclude self-distance at index 0)
    mean_dist = distances[:, 1:].mean(axis=1)
    density = 1.0 / (mean_dist + 1e-8)
    
    return density.reshape(-1, 1)


def eccentricity_filter(
    X: np.ndarray,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Eccentricity (max distance to any other point) as filter.
    
    Points at the "edge" of the distribution have high eccentricity.
    
    Args:
        X: Data matrix [n_samples, n_features]
        metric: Distance metric
        
    Returns:
        Eccentricity values [n_samples, 1]
    """
    from scipy.spatial.distance import cdist
    
    dists = cdist(X, X, metric=metric)
    eccentricity = dists.max(axis=1)
    
    return eccentricity.reshape(-1, 1)


def l2norm_filter(X: np.ndarray) -> np.ndarray:
    """
    L2 norm of each point as filter.
    
    Args:
        X: Data matrix [n_samples, n_features]
        
    Returns:
        L2 norms [n_samples, 1]
    """
    norms = np.linalg.norm(X, axis=1)
    return norms.reshape(-1, 1)


def evidence_tier_filter(
    evidence_tiers: np.ndarray,
) -> np.ndarray:
    """
    Evidence tier as filter function.
    
    Maps evidence tier to numerical value for filtering.
    Lower tiers (better evidence) get lower values.
    
    Args:
        evidence_tiers: Array of evidence tier indices
        
    Returns:
        Filter values [n_samples, 1]
    """
    # Normalize to [0, 1] range
    min_tier = evidence_tiers.min()
    max_tier = evidence_tiers.max()
    
    if max_tier == min_tier:
        return np.zeros((len(evidence_tiers), 1))
    
    normalized = (evidence_tiers - min_tier) / (max_tier - min_tier)
    return normalized.reshape(-1, 1)


def combined_filter(
    X: np.ndarray,
    filters: list[str],
    weights: Optional[list[float]] = None,
    **kwargs,
) -> np.ndarray:
    """
    Combine multiple filter functions.
    
    Args:
        X: Data matrix
        filters: List of filter names to combine
        weights: Optional weights for each filter
        **kwargs: Additional arguments for individual filters
        
    Returns:
        Combined filter values [n_samples, n_filters]
    """
    if weights is None:
        weights = [1.0] * len(filters)
    
    results = []
    
    for filter_name, weight in zip(filters, weights):
        filter_fn = get_filter_function(filter_name)
        result = filter_fn(X, **kwargs.get(filter_name, {}))
        
        # Normalize each filter to [0, 1]
        result_min = result.min()
        result_max = result.max()
        if result_max > result_min:
            result = (result - result_min) / (result_max - result_min)
        results.append(result * weight)
    
    return np.hstack(results)


def get_filter_function(name: str) -> Callable:
    """
    Get filter function by name.
    
    Args:
        name: Name of the filter function
        
    Returns:
        Filter function callable
        
    Raises:
        ValueError: If filter name is unknown
    """
    filters = {
        "pca": pca_filter,
        "umap": umap_filter,
        "density": density_filter,
        "eccentricity": eccentricity_filter,
        "l2norm": l2norm_filter,
        "evidence_tier": evidence_tier_filter,
    }
    
    if name not in filters:
        available = ", ".join(filters.keys())
        raise ValueError(f"Unknown filter: {name}. Available: {available}")
    
    return filters[name]


def list_available_filters() -> list[str]:
    """Return list of available filter function names."""
    return ["pca", "umap", "density", "eccentricity", "l2norm", "evidence_tier"]
