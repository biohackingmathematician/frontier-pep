"""
Topological Data Analysis module for the Peptide Atlas.

Provides Mapper algorithm and persistent homology for analyzing
the topological structure of peptide embedding space.

REMINDER: This project is for research and education only.
No dosing, no protocols, no therapeutic recommendations.
"""

from peptide_atlas.tda.mapper import MapperPipeline
from peptide_atlas.tda.persistence import PersistentHomology
from peptide_atlas.tda.filters import get_filter_function
from peptide_atlas.tda.analysis import analyze_mapper_graph

__all__ = [
    "MapperPipeline",
    "PersistentHomology",
    "get_filter_function",
    "analyze_mapper_graph",
]

