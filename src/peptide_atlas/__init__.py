"""
The Frontier Peptide Atlas

A Foundational Knowledge Resource for Under-Characterized Peptide Mechanisms.

This is the ImageNet of regenerative peptide research — a structured,
evidence-classified knowledge graph with learned embeddings for
mechanism-based similarity search.

Author: Agna Chan
Version: 0.1.0
Repository: https://github.com/biohackingmathematician/frontier-pep

CRITICAL DISCLAIMER:
This project is for RESEARCH AND EDUCATIONAL PURPOSES ONLY.
No dosing, no protocols, no therapeutic recommendations.
"""

__version__ = "0.1.0"
__author__ = "Agna Chan"


DISCLAIMER_TEXT = """
================================================================================
                           CRITICAL DISCLAIMER
================================================================================

  The Frontier Peptide Atlas is for RESEARCH AND EDUCATIONAL PURPOSES ONLY.

  - It does NOT constitute medical advice or treatment guidance.
  - Inclusion does NOT imply safety, efficacy, or legality for any compound.
  - Many peptides are experimental, off-label, or not approved for humans.
  - NO dosing, NO protocols, NO usage recommendations are provided.
  - Consult a qualified healthcare professional for any medical decisions.

  This is a research tool, not a clinical guide.

================================================================================
"""


def print_disclaimer() -> None:
    """Print the critical disclaimer to stdout."""
    print(DISCLAIMER_TEXT)


def get_disclaimer() -> str:
    """Return the disclaimer text."""
    return DISCLAIMER_TEXT.strip()


# Lazy imports to avoid circular dependencies
def _get_atlas():
    from peptide_atlas.api.atlas import PeptideAtlas
    return PeptideAtlas


def _get_build_kg():
    from peptide_atlas.kg.builder import build_knowledge_graph
    return build_knowledge_graph


def _get_curated_peptides():
    from peptide_atlas.data.peptide_catalog import get_curated_peptides
    return get_curated_peptides


# Make PeptideAtlas available at package level
@property
def PeptideAtlas():
    return _get_atlas()


# Convenience function to build knowledge graph
def build_knowledge_graph():
    """Build the knowledge graph from curated data."""
    return _get_build_kg()()


# Convenience function to get curated peptides
def get_curated_peptides():
    """Get the curated list of peptides."""
    return _get_curated_peptides()()


__all__ = [
    # Version
    "__version__",
    "__author__",
    # Disclaimer
    "print_disclaimer",
    "get_disclaimer",
    "DISCLAIMER_TEXT",
    # Functions
    "build_knowledge_graph",
    "get_curated_peptides",
]
