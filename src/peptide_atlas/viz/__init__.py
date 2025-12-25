"""
Visualization module for the Peptide Atlas.

Provides interactive visualizations of the peptide embedding space.

REMINDER: This project is for research and education only.
No dosing, no protocols, no therapeutic recommendations.
"""

from peptide_atlas.viz.world_map import create_world_map
from peptide_atlas.viz.style import get_color_palette, DISCLAIMER_TEXT

__all__ = [
    "create_world_map",
    "get_color_palette",
    "DISCLAIMER_TEXT",
]

