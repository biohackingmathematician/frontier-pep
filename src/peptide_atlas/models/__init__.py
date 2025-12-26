"""
Models module for the Peptide Atlas.

Contains GNN encoder and hyperbolic embedding components.

REMINDER: This project is for research and education only.
No dosing, no protocols, no therapeutic recommendations.
"""

from peptide_atlas.models.gnn import (
    RelationalGATLayer,
    HeterogeneousGNNEncoder,
    GNNConfig,
    TrainingConfig,
    PretrainingConfig,
)
from peptide_atlas.models.hyperbolic import (
    PoincareEmbedding,
    HyperbolicMLR,
    EuclideanToPoincareProjection,
    poincare_distance,
    project_to_poincare,
)

__all__ = [
    # GNN
    "RelationalGATLayer",
    "HeterogeneousGNNEncoder",
    "GNNConfig",
    "TrainingConfig",
    "PretrainingConfig",
    # Hyperbolic
    "PoincareEmbedding",
    "HyperbolicMLR",
    "EuclideanToPoincareProjection",
    "poincare_distance",
    "project_to_poincare",
]
