"""
Knowledge Graph module for the Peptide Atlas.

Contains graph construction, querying, and export utilities.

REMINDER: This project is for research and education only.
No dosing, no protocols, no therapeutic recommendations.
"""

from peptide_atlas.kg.builder import KnowledgeGraphBuilder, build_knowledge_graph

__all__ = [
    "KnowledgeGraphBuilder",
    "build_knowledge_graph",
]

