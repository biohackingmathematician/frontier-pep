"""
Graph Neural Network module for the Peptide Atlas.

REMINDER: This project is for research and education only.
"""

from peptide_atlas.models.gnn.config import GNNConfig, TrainingConfig, PretrainingConfig
from peptide_atlas.models.gnn.encoder import HeterogeneousGNNEncoder
from peptide_atlas.models.gnn.layers import RelationalGATLayer

__all__ = [
    "GNNConfig",
    "TrainingConfig",
    "PretrainingConfig",
    "HeterogeneousGNNEncoder",
    "RelationalGATLayer",
]
