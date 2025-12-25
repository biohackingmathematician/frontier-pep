"""
Models module for the Peptide Atlas.

Contains GNN encoders and hyperbolic embedding components.

REMINDER: This project is for research and education only.
No dosing, no protocols, no therapeutic recommendations.
"""

from peptide_atlas.models.gnn import HeterogeneousGNNEncoder
from peptide_atlas.models.hyperbolic import PoincareEmbedding

__all__ = [
    "HeterogeneousGNNEncoder",
    "PoincareEmbedding",
]

