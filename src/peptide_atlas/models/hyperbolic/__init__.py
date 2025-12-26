"""
Hyperbolic embedding module for the Peptide Atlas.

Provides Poincaré ball embeddings and operations for representing
hierarchical peptide relationships.

REMINDER: This project is for research and education only.
"""

from peptide_atlas.models.hyperbolic.poincare import (
    PoincareEmbedding,
    HyperbolicMLR,
    EuclideanToPoincareProjection,
    project_to_poincare,
    exponential_map,
    logarithmic_map,
    mobius_addition,
)
from peptide_atlas.models.hyperbolic.distance import (
    poincare_distance,
    poincare_distance_matrix,
    hyperbolic_centroid,
    hyperbolic_mlr,
)

__all__ = [
    # Classes
    "PoincareEmbedding",
    "HyperbolicMLR",
    "EuclideanToPoincareProjection",
    # Functions from poincare.py
    "project_to_poincare",
    "exponential_map",
    "logarithmic_map",
    "mobius_addition",
    # Functions from distance.py
    "poincare_distance",
    "poincare_distance_matrix",
    "hyperbolic_centroid",
    "hyperbolic_mlr",
]
