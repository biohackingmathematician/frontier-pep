"""
Filter functions for the Mapper algorithm.

Filter functions (also called lenses) project data to lower dimensions
for the Mapper cover construction.

REMINDER: This project is for research and education only.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from loguru import logger


def get_filter_function(
    name: str,
    **kwargs,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Get a filter function by name.
    
    Args:
        name: Name of filter function
        **kwargs: Additional arguments for the filter
        
    Returns:
        Callable that maps data to filter values
    """
    filters = {
        "umap": umap_filter,
        "pca": pca_filter,
        "density": density_filter,
        "eccentricity": eccentricity_filter,
        "l2_norm": l2_norm_filter,
        "evidence_tier": evidence_tier_filter,
    }
    
    if name not in filters:
        raise ValueError(f"Unknown filter: {name}. Available: {list(filters.keys())}")
    
    def wrapped(X: np.ndarray) -> np.ndarray:
        return filters[name](X, **kwargs)
    
    return wrapped


def umap_filter(
    X: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """
    UMAP-based filter function.
    
    Projects data using UMAP for a topology-preserving lens.
    """
    try:
        import umap
        
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            random_state=random_state,
        )
        return reducer.fit_transform(X)
    except ImportError:
        logger.warning("UMAP not available, falling back to PCA")
        return pca_filter(X, n_components=n_components)


def pca_filter(
    X: np.ndarray,
    n_components: int = 2,
) -> np.ndarray:
    """
    PCA-based filter function.
    
    Projects to principal components.
    """
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)


def density_filter(
    X: np.ndarray,
    n_neighbors: int = 15,
) -> np.ndarray:
    """
    Density-based filter function.
    
    Estimates local density using k-nearest neighbors.
    """
    from sklearn.neighbors import NearestNeighbors
    
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    
    # Density estimate: inverse of average distance to k neighbors
    avg_dist = distances.mean(axis=1)
    density = 1.0 / (avg_dist + 1e-8)
    
    return density.reshape(-1, 1)


def eccentricity_filter(
    X: np.ndarray,
    p: float = 2.0,
) -> np.ndarray:
    """
    Eccentricity-based filter function.
    
    Measures how far each point is from the center of the data.
    """
    # Compute distances to mean
    center = X.mean(axis=0)
    eccentricity = np.linalg.norm(X - center, ord=p, axis=1)
    
    return eccentricity.reshape(-1, 1)


def l2_norm_filter(
    X: np.ndarray,
) -> np.ndarray:
    """
    L2 norm filter function.
    
    Simple filter based on vector magnitude.
    """
    norms = np.linalg.norm(X, axis=1)
    return norms.reshape(-1, 1)


def evidence_tier_filter(
    X: np.ndarray,
    evidence_tiers: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Evidence tier-based filter function.
    
    Uses evidence quality as a filter dimension.
    Combines with PCA for 2D lens.
    
    Args:
        X: Data matrix
        evidence_tiers: Evidence tier values (0-6 scale)
    """
    # PCA component
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    pca_vals = pca.fit_transform(X)
    
    if evidence_tiers is not None:
        # Normalize evidence tiers
        tier_normalized = evidence_tiers.reshape(-1, 1) / 6.0
        return np.hstack([pca_vals, tier_normalized])
    else:
        # Second PCA component
        pca2 = PCA(n_components=2)
        return pca2.fit_transform(X)


def composite_filter(
    X: np.ndarray,
    filter_names: list[str],
    weights: Optional[list[float]] = None,
    **kwargs,
) -> np.ndarray:
    """
    Composite filter combining multiple filter functions.
    
    Args:
        X: Data matrix
        filter_names: List of filter function names
        weights: Optional weights for each filter
        **kwargs: Arguments passed to individual filters
    """
    if weights is None:
        weights = [1.0] * len(filter_names)
    
    components = []
    for name in filter_names:
        filter_fn = get_filter_function(name)
        component = filter_fn(X, **kwargs.get(name, {}))
        
        # Normalize each component
        component = (component - component.mean(axis=0)) / (component.std(axis=0) + 1e-8)
        components.append(component)
    
    # Combine with weights
    combined = np.zeros_like(components[0])
    for comp, w in zip(components, weights):
        combined += w * comp
    
    return combined


def create_lens_from_peptide_data(
    embeddings: np.ndarray,
    peptide_classes: Optional[np.ndarray] = None,
    evidence_tiers: Optional[np.ndarray] = None,
    method: str = "umap",
) -> np.ndarray:
    """
    Create a lens function specialized for peptide data.
    
    Args:
        embeddings: Peptide embeddings
        peptide_classes: Categorical class labels
        evidence_tiers: Evidence tier values
        method: Base method for lens
        
    Returns:
        2D lens values
    """
    if method == "umap":
        lens = umap_filter(embeddings)
    elif method == "pca":
        lens = pca_filter(embeddings)
    else:
        lens = umap_filter(embeddings)
    
    # Optionally incorporate evidence as color but not filter
    # (Filter should be based on geometry, not labels)
    
    return lens

