"""
The Frontier Peptide Atlas

Graph Foundations and Topological Data Analysis for Mapping
Under-Characterized Regenerative, Immune, and Anabolic Mechanism Space

CRITICAL DISCLAIMER

This Peptide Atlas is for RESEARCH AND EDUCATIONAL PURPOSES ONLY.

- It does NOT constitute medical advice, treatment guidance, or clinical recommendations.
- The inclusion of any peptide does NOT imply it is safe, effective, legal, or appropriate.
- Many peptides mapped here are experimental, off-label, inadequately studied, or not approved 
  for human use. They may carry serious, unknown, or life-threatening risks.
- We provide NO dosing information, NO protocol design, NO usage recommendations.
- Any decision to use such compounds must be made with a qualified healthcare professional 
  and must comply with all applicable laws and regulations.

If you are considering using any peptide, consult a licensed physician.
"""

__version__ = "0.1.0"
__author__ = "Agna Chan"

# Standard disclaimer to be used throughout the package
DISCLAIMER = """
CRITICAL DISCLAIMER

This Peptide Atlas is for RESEARCH AND EDUCATIONAL PURPOSES ONLY.

- It does NOT constitute medical advice, treatment guidance, or clinical recommendations.
- The inclusion of any peptide does NOT imply it is safe, effective, legal, or appropriate.
- Many peptides mapped here are experimental, off-label, inadequately studied, or not approved 
  for human use. They may carry serious, unknown, or life-threatening risks.
- We provide NO dosing information, NO protocol design, NO usage recommendations.
- Any decision to use such compounds must be made with a qualified healthcare professional 
  and must comply with all applicable laws and regulations.

If you are considering using any peptide, consult a licensed physician.
"""


def print_disclaimer() -> None:
    """Print the standard disclaimer."""
    print(DISCLAIMER)

