"""
Topological Data Analysis module for the Peptide Atlas.

Provides Mapper algorithm and persistent homology for structure discovery.

REMINDER: This project is for research and education only.
No dosing, no protocols, no therapeutic recommendations.
"""

from peptide_atlas.tda.filters import (
    pca_filter,
    umap_filter,
    density_filter,
    eccentricity_filter,
    l2norm_filter,
    evidence_tier_filter,
    combined_filter,
    get_filter_function,
    list_available_filters,
)
from peptide_atlas.tda.mapper import (
    MapperConfig,
    MapperResult,
    MapperPipeline,
    create_mapper_from_config,
)

__all__ = [
    # Filter functions
    "pca_filter",
    "umap_filter",
    "density_filter",
    "eccentricity_filter",
    "l2norm_filter",
    "evidence_tier_filter",
    "combined_filter",
    "get_filter_function",
    "list_available_filters",
    # Mapper
    "MapperConfig",
    "MapperResult",
    "MapperPipeline",
    "create_mapper_from_config",
]
