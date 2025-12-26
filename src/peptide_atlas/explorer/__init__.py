"""
Explorer module for the Peptide Atlas.

Provides visual exploration tools for the knowledge graph and embeddings.

NOTE: The explorer is a convenience tool for browsing the atlas.
The core value is in the knowledge graph and embeddings, not the visualization.

REMINDER: This project is for research and education only.
"""

from peptide_atlas.explorer.scatter import create_embedding_scatter
from peptide_atlas.explorer.graph import create_kg_visualization
from peptide_atlas.explorer.dashboard import launch_explorer

__all__ = [
    "create_embedding_scatter",
    "create_kg_visualization",
    "launch_explorer",
]

