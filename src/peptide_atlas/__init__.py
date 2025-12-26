"""
The Frontier Peptide Atlas.

A Foundational Knowledge Resource for Under-Characterized Peptide Mechanisms.

Graph Foundations and Topological Data Analysis for Mapping 
Under-Characterized Regenerative, Immune, and Anabolic Mechanism Space.

Author: Agna Chan
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

  This Peptide Atlas is for RESEARCH AND EDUCATIONAL PURPOSES ONLY.

  - It does NOT constitute medical advice or treatment guidance.
  - Inclusion of any peptide does NOT imply safety, efficacy, or legality.
  - Many compounds are experimental, off-label, or not approved for humans.
  - NO dosing information, NO protocols, NO usage recommendations.
  - Consult a qualified healthcare professional for any medical decisions.

================================================================================
"""


def print_disclaimer() -> None:
    """Print the critical disclaimer to stdout."""
    print(DISCLAIMER_TEXT)


def get_disclaimer() -> str:
    """Return the disclaimer text as a string."""
    return DISCLAIMER_TEXT.strip()


# Convenience imports
from peptide_atlas.kg import build_knowledge_graph
from peptide_atlas.data.peptide_catalog import get_curated_peptides
from peptide_atlas.api.atlas import PeptideAtlas

__all__ = [
    "__version__",
    "__author__",
    "print_disclaimer",
    "get_disclaimer",
    "build_knowledge_graph",
    "get_curated_peptides",
    "DISCLAIMER_TEXT",
    "PeptideAtlas",
]
