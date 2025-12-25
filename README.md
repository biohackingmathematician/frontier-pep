# The Frontier Peptide Atlas

> Graph Foundations and Topological Data Analysis for Mapping Under-Characterized Regenerative, Immune, and Anabolic Mechanism Space

**Author:** Agna Chan  
**Date:** December 2025  
**Repository:** [github.com/biohackingmathematician/frontier-pep](https://github.com/biohackingmathematician/frontier-pep)

---

## CRITICAL DISCLAIMER

**This Peptide Atlas is for RESEARCH AND EDUCATIONAL PURPOSES ONLY.**

- It does **NOT** constitute medical advice, treatment guidance, or clinical recommendations.
- The inclusion of any peptide does **NOT** imply it is safe, effective, legal, or appropriate for any individual.
- Many peptides mapped here are **experimental, off-label, inadequately studied, or not approved** for human use. They may carry **serious, unknown, or life-threatening risks**.
- We provide **NO dosing information, NO protocol design, NO usage recommendations**.
- Any decision to use such compounds must be made **with a qualified healthcare professional** and must comply with all applicable laws and regulations.

**If you are considering using any peptide, consult a licensed physician. Do not self-prescribe. Do not self-inject.**

---

## Overview

The Frontier Peptide Atlas is a computational research framework that organizes non-GLP-1 peptides by mechanism, effect, and risk topology using modern representation learning and topological data analysis.

### What This Project Does

- Constructs a **heterogeneous knowledge graph** linking peptides to molecular targets, pathways, effects, and risks
- Trains a **graph neural network** to learn peptide embeddings that capture mechanistic relationships
- Projects embeddings into **hyperbolic (Poincaré) space** to preserve hierarchical structure
- Applies **topological data analysis** (Mapper algorithm) to reveal cluster structure and mechanistic bridges
- Generates a **"world map"** visualization of peptide mechanism space

### What This Project Does NOT Do

- Does NOT provide dosing or protocol recommendations
- Does NOT suggest therapeutic interventions
- Does NOT claim efficacy for any peptide
- Does NOT propose new drug candidates
- Does NOT offer clinical decision support

### Peptide Classes Covered

| Class | Examples | Focus |
|-------|----------|-------|
| GH/GHRH Axis | Sermorelin-class, CJC-class | Growth hormone releasing |
| GH Secretagogues | Ipamorelin-class, MK-677-type | Ghrelin receptor agonism |
| IGF-1/Insulin Axis | IGF-1, MGF concepts | Anabolic signaling |
| Regenerative/Repair | BPC-class, TB-500-class | Tissue healing |
| Thymic/Immune | Thymosin alpha-1–class | Immune modulation |
| CNS/Neurotrophic | Semax-class, Selank-class | Neuroprotection |
| Longevity | Epithalon-class, mitochondrial peptides | Cellular resilience |

**Note:** GLP-1 agonists are deliberately de-emphasized (included as reference only).

---

## Installation

```bash
# Clone repository
git clone https://github.com/biohackingmathematician/frontier-pep.git
cd frontier-pep/peptide-atlas

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -e ".[dev]"
```

## Quick Start

```bash
# Build knowledge graph from curated data
peptide-atlas build-kg

# Train GNN encoder
peptide-atlas train --config configs/model_config.yaml

# Run TDA analysis
peptide-atlas analyze-tda --config configs/tda_config.yaml

# Generate world map visualization
peptide-atlas visualize --output outputs/world_map.html
```

## Project Structure

```
peptide-atlas/
├── src/peptide_atlas/     # Main package
│   ├── data/              # Data schemas and loaders
│   ├── kg/                # Knowledge graph construction
│   ├── models/            # GNN and hyperbolic embeddings
│   ├── tda/               # Topological data analysis
│   └── viz/               # Visualization
├── data/                  # Data files
├── configs/               # Configuration files
├── notebooks/             # Jupyter notebooks
└── tests/                 # Unit tests
```

## Evidence Taxonomy

All peptides are classified by evidence tier:

| Tier | Description | Interpretation |
|------|-------------|----------------|
| 1 | Regulatory approval | Highest confidence |
| 2 | Phase II/III RCT data | Strong clinical evidence |
| 3 | Early clinical trials | Preliminary human data |
| 4 | Preclinical only | Animal/cell studies |
| 5 | Mechanistic only | In vitro / theoretical |
| 6 | Anecdotal | Case reports, uncontrolled |
| Unknown | Insufficient data | Treat with high caution |

## Citation

If you use this work, please cite:

```bibtex
@software{frontier_peptide_atlas,
  author = {Chan, Agna},
  title = {The Frontier Peptide Atlas: Graph Foundations and Topological Data Analysis for Mapping Under-Characterized Peptide Mechanism Space},
  year = {2025},
  month = {December},
  url = {https://github.com/biohackingmathematician/frontier-pep}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Ethics & Responsible Use

See [ETHICS.md](ETHICS.md) for full discussion of ethical considerations, limitations, and responsible use guidelines.

---

**Remember: This is a research tool, not a clinical guide. Consult healthcare professionals for any medical decisions.**

