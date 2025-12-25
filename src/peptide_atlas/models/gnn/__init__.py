"""
Graph Neural Network module for the Peptide Atlas.

REMINDER: This project is for research and education only.
"""

from peptide_atlas.models.gnn.encoder import HeterogeneousGNNEncoder
from peptide_atlas.models.gnn.layers import RelationalGATLayer

__all__ = [
    "HeterogeneousGNNEncoder",
    "RelationalGATLayer",
]

