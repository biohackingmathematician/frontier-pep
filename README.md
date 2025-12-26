# The Frontier Peptide Atlas

> A Foundational Knowledge Resource for Under-Characterized Peptide Mechanisms

**Author:** Agna Chan  
**Version:** 0.1.0  
**Repository:** [github.com/biohackingmathematician/frontier-pep](https://github.com/biohackingmathematician/frontier-pep)

---

## CRITICAL DISCLAIMER

**This resource is for RESEARCH AND EDUCATIONAL PURPOSES ONLY.**

- It does **NOT** constitute medical advice, treatment guidance, or clinical recommendations.
- Inclusion of any peptide does **NOT** imply it is safe, effective, legal, or appropriate.
- Many peptides are **experimental, off-label, inadequately studied, or not approved** for human use.
- **NO dosing information, NO protocol design, NO usage recommendations** are provided.
- Any decisions regarding peptide use must be made **with a qualified healthcare professional**.

---

## What Is This?

The Frontier Peptide Atlas is a **foundational knowledge resource** that systematically maps the mechanism space of under-characterized peptides — particularly those in the regenerative, immune-modulatory, and anabolic domains that exist outside mainstream pharmaceutical development.

Think of it as:
- **ImageNet for peptide mechanisms** — A structured, labeled dataset enabling computational research
- **Gene Ontology for peptide effects** — A formal vocabulary linking compounds to biological outcomes
- **AlphaFold DB for peptide relationships** — Pretrained embeddings encoding mechanistic similarity

### Why This Matters

The peptide landscape is fragmented:
- **Approved compounds** (GLP-1 agonists, insulin) are well-documented
- **Research peptides** (BPC-157, TB-500, Thymosin alpha-1) exist in a gray zone with scattered evidence
- **No unified resource** maps these compounds by mechanism, evidence quality, and risk profile

The Peptide Atlas fills this gap with:
1. A **heterogeneous knowledge graph** linking peptides to molecular targets, pathways, effects, and risks
2. **Graph neural network embeddings** that encode mechanistic relationships
3. An **evidence taxonomy** that rigorously classifies data quality
4. **Query APIs** for programmatic access

---

## Core Assets

### 1. Knowledge Graph (`data/processed/kg.json`)

Structured relationships between:
- **40+ curated peptides** across 8 mechanism classes
- **20+ molecular targets** (receptors, tissues, cell types)
- **18 biological pathways** (GH/IGF-1 axis, PI3K-Akt-mTOR, etc.)
- **16 effect domains** (anabolic, regenerative, cognitive, etc.)
- **14 risk categories** (with severity and reversibility annotations)

### 2. Pretrained Embeddings (`data/processed/embeddings.pt`)

64-dimensional vectors for each peptide encoding:
- Mechanistic similarity (peptides with similar targets cluster together)
- Pathway relationships (shared pathway involvement)
- Effect profiles (similar outcome patterns)

### 3. Evidence Taxonomy

Every peptide and relationship is classified:

| Tier | Description | Confidence | Example |
|------|-------------|------------|---------|
| 1 | Regulatory approval | 1.00 | Semaglutide, Tesamorelin |
| 2 | Phase II/III RCT | 0.85 | Thymosin alpha-1 |
| 3 | Early clinical | 0.65 | Sermorelin, MK-677 |
| 4 | Preclinical only | 0.45 | BPC-157, TB-500 |
| 5 | Mechanistic/in vitro | 0.25 | FOXO4-DRI |
| 6 | Anecdotal | 0.15 | Many "research" peptides |
| Unknown | Insufficient data | 0.05 | Treat with extreme caution |

### 4. Query API

```python
from peptide_atlas import PeptideAtlas

atlas = PeptideAtlas.load("data/processed/")

# Find peptides by mechanism
gh_secretagogues = atlas.query_by_class("ghs_ghrelin_mimetic")

# Find similar peptides (embedding space)
similar_to_bpc157 = atlas.find_similar("BPC-157", k=5)

# Filter by evidence quality
clinical_grade = atlas.query_by_evidence(min_tier=3)

# Get pathway relationships
igf_pathway_peptides = atlas.query_by_pathway("PI3K-Akt-mTOR")
```

---

## Peptide Classes Covered

| Class | Count | Examples | Evidence Range |
|-------|-------|----------|----------------|
| GH/GHRH Axis | 4 | Sermorelin, Tesamorelin, CJC-1295 | Tier 1-4 |
| GH Secretagogues | 5 | Ipamorelin, GHRP-2, MK-677 | Tier 2-4 |
| IGF-1/Insulin Axis | 4 | Mecasermin, IGF-1 LR3, MGF | Tier 1-5 |
| Regenerative/Repair | 5 | BPC-157, TB-500, GHK-Cu | Tier 3-4 |
| Thymic/Immune | 4 | Thymosin alpha-1, LL-37 | Tier 2-4 |
| CNS/Neurotrophic | 5 | Semax, Selank, Dihexa | Tier 2-5 |
| Longevity/Cellular | 5 | Epithalon, SS-31, MOTS-c | Tier 2-5 |
| Metabolic | 3 | Pramlintide, Oxyntomodulin | Tier 1-3 |

**Note:** GLP-1 agonists (Semaglutide, Tirzepatide) are included as reference landmarks only — they are well-characterized elsewhere.

---

## Installation

```bash
git clone https://github.com/biohackingmathematician/frontier-pep.git
cd frontier-pep

# Create environment
python -m venv venv
source venv/bin/activate

# Install
pip install -e ".[dev]"
```

## Quick Start

```bash
# Build knowledge graph
peptide-atlas build-kg --output data/processed/kg.json

# Train embeddings
peptide-atlas train --epochs 100 --output data/processed/embeddings.pt

# Query the atlas
peptide-atlas query --similar-to "BPC-157" --k 5

# Launch explorer interface
peptide-atlas explore --port 8050
```

## Python API

```python
from peptide_atlas import PeptideAtlas, print_disclaimer

print_disclaimer()

# Load atlas
atlas = PeptideAtlas.load("data/processed/")

# Explore
print(f"Peptides: {atlas.num_peptides}")
print(f"Targets: {atlas.num_targets}")
print(f"Pathways: {atlas.num_pathways}")

# Query
regenerative = atlas.query_by_class("regenerative_repair")
for p in regenerative:
    print(f"  {p.name}: {p.evidence_tier}")
```

---

## Project Structure

```
frontier-pep/
├── src/peptide_atlas/
│   ├── data/           # Schemas, loaders, catalog
│   ├── kg/             # Knowledge graph construction
│   ├── models/         # GNN encoder, hyperbolic embeddings
│   ├── tda/            # Topological data analysis
│   ├── api/            # Query interface
│   └── explorer/       # Visual exploration tool
├── data/
│   ├── processed/      # KG JSON, embeddings, assets
│   └── schemas/        # Ontology definitions
├── configs/            # Model and pipeline configs
├── scripts/            # CLI entry points
├── notebooks/          # Research notebooks
└── tests/              # Unit tests
```

---

## Use Cases

### For Researchers
- Query mechanistic relationships: "What pathways does Thymosin alpha-1 modulate?"
- Find similar compounds: "What's mechanistically similar to Semax?"
- Evidence filtering: "Show me only peptides with Phase II+ data"

### For Developers
- Build on the knowledge graph for downstream applications
- Use pretrained embeddings for peptide similarity search
- Extend the ontology with new compounds

### For Writers/Educators
- Understand the evidence landscape for specific peptides
- Visualize mechanistic relationships
- Reference a structured, citable resource

---

## What This Project Does NOT Do

- Does NOT provide dosing or protocol recommendations
- Does NOT suggest therapeutic interventions
- Does NOT claim efficacy for any compound
- Does NOT offer clinical decision support
- Does NOT propose compounds for human use
- Does NOT replace consultation with healthcare professionals

---

## Citation

```bibtex
@software{frontier_peptide_atlas,
  author = {Chan, Agna},
  title = {The Frontier Peptide Atlas: A Foundational Knowledge Resource for Under-Characterized Peptide Mechanisms},
  year = {2025},
  version = {0.1.0},
  url = {https://github.com/biohackingmathematician/frontier-pep}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).

## Ethics

See [ETHICS.md](ETHICS.md) for responsible use guidelines.

---

**This is a research tool, not a clinical guide. Consult healthcare professionals for any medical decisions.**
