"""
Hyperbolic embedding module for the Peptide Atlas.

Provides Poincaré ball embeddings for hierarchical structure preservation.

REMINDER: This project is for research and education only.
"""

from peptide_atlas.models.hyperbolic.poincare import PoincareEmbedding
from peptide_atlas.models.hyperbolic.projection import EuclideanToPoincareProjection
from peptide_atlas.models.hyperbolic.distance import poincare_distance, poincare_distance_matrix

__all__ = [
    "PoincareEmbedding",
    "EuclideanToPoincareProjection",
    "poincare_distance",
    "poincare_distance_matrix",
]

