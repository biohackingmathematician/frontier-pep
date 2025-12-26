"""
Visualization module for the Peptide Atlas.

NOTE: The core value of this project is in the knowledge graph and embeddings,
not the visualizations. These are exploration tools.

REMINDER: This project is for research and education only.
"""

from peptide_atlas.viz.world_map import (
    create_world_map,
    create_mapper_visualization,
)
from peptide_atlas.viz.style import (
    get_theme,
    get_color_palette,
    format_hover_text,
    DISCLAIMER_TEXT,
)

# Visualization disclaimer
VIZ_DISCLAIMER = (
    "RESEARCH USE ONLY - Not medical advice. "
    "No dosing or protocol recommendations. "
    "Consult a healthcare professional."
)

__all__ = [
    "create_world_map",
    "create_mapper_visualization",
    "get_theme",
    "get_color_palette",
    "format_hover_text",
    "VIZ_DISCLAIMER",
    "DISCLAIMER_TEXT",
]
