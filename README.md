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

### 1. Knowledge Graph

Structured relationships between:
- **55+ curated peptides** across 9 mechanism classes
- **20+ molecular targets** (receptors, tissues, cell types)
- **18 biological pathways** (GH/IGF-1 axis, PI3K-Akt-mTOR, etc.)
- **16 effect domains** (anabolic, regenerative, cognitive, etc.)
- **14 risk categories** (with severity and reversibility annotations)

### 2. Pretrained Embeddings

64-dimensional vectors for each peptide encoding:
- Mechanistic similarity (peptides with similar targets cluster together)
- Pathway relationships (shared pathway involvement)
- Effect profiles (similar outcome patterns)

### 3. Evidence Taxonomy

Every peptide and relationship is classified:

| Tier | Description | Confidence | Examples |
|------|-------------|------------|----------|
| 1 | Regulatory approval | 1.00 | Semaglutide, Tesamorelin, Mecasermin |
| 2 | Phase II/III RCT | 0.85 | Thymosin alpha-1, SS-31/Elamipretide |
| 3 | Early clinical | 0.65 | Sermorelin, MK-677, Semax |
| 4 | Preclinical only | 0.45 | BPC-157, TB-500, Epithalon |
| 5 | Mechanistic/in vitro | 0.25 | FOXO4-DRI, Humanin |
| 6 | Anecdotal | 0.15 | Many research peptides |
| Unknown | Insufficient data | 0.05 | Treat with extreme caution |

### 4. Query API

```python
from peptide_atlas import PeptideAtlas

atlas = PeptideAtlas.load("data/processed/")

# Find peptides by mechanism class
gh_secretagogues = atlas.query_by_class("ghs_ghrelin_mimetic")

# Find similar peptides (embedding space)
similar_to_bpc157 = atlas.find_similar("BPC-157", k=5)
# Returns: TB-500, Thymosin Beta-4, GHK-Cu, AOD9604, ...

# Filter by evidence quality
clinical_grade = atlas.query_by_evidence(min_tier=3)

# Get pathway relationships
igf_pathway_peptides = atlas.query_by_pathway("PI3K-Akt-mTOR")
```

---

## Peptide Classes Covered

| Class | Count | Examples | Evidence Range |
|-------|-------|----------|----------------|
| GH/GHRH Axis | 5 | Sermorelin, Tesamorelin, CJC-1295 | Tier 1-4 |
| GH Secretagogues | 6 | Ipamorelin, GHRP-2, GHRP-6, Hexarelin, MK-677 | Tier 2-4 |
| IGF-1/Insulin Axis | 5 | Mecasermin, IGF-1 LR3, MGF, PEG-MGF | Tier 1-5 |
| Regenerative/Repair | 6 | BPC-157, TB-500, Thymosin Beta-4, GHK-Cu | Tier 3-4 |
| Thymic/Immune | 5 | Thymosin alpha-1, Thymalin, LL-37 | Tier 2-4 |
| CNS/Neurotrophic | 7 | Semax, Selank, Dihexa, P21, Cerebrolysin | Tier 2-5 |
| Longevity/Cellular | 6 | Epithalon, SS-31, MOTS-c, Humanin | Tier 2-5 |
| Metabolic | 4 | Pramlintide, Oxyntomodulin | Tier 1-4 |
| Antimicrobial | 4 | LL-37, Lactoferricin, Pexiganan | Tier 2-4 |

**Note:** GLP-1 agonists (Semaglutide, Tirzepatide) are included as reference landmarks only.

---

## Installation

```bash
git clone https://github.com/biohackingmathematician/frontier-pep.git
cd frontier-pep

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -e ".[dev]"
```

## Quick Start

```bash
# Build knowledge graph
python scripts/build_kg.py --output data/processed/kg.json

# Train embeddings
python scripts/train_gnn.py --epochs 100 --output data/processed/

# Query the atlas
python scripts/query_atlas.py stats
python scripts/query_atlas.py similar "BPC-157" -k 5
python scripts/query_atlas.py list --class regenerative_repair

# Run TDA analysis
python scripts/run_tda.py --output outputs/tda/
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

# Query by class
regenerative = atlas.query_by_class("regenerative_repair")
for p in regenerative:
    print(f"  {p.canonical_name}: {p.evidence_tier.value}")

# Similarity search
similar = atlas.find_similar("Semax", k=5)
for r in similar:
    print(f"  {r.peptide.canonical_name}: {r.similarity:.3f}")
```

---

## Project Structure

```
frontier-pep/
├── src/peptide_atlas/
│   ├── data/           # Schemas, loaders, peptide catalog
│   ├── kg/             # Knowledge graph construction and export
│   ├── models/         # GNN encoder, hyperbolic embeddings
│   ├── tda/            # Topological data analysis
│   ├── api/            # Query interface (PeptideAtlas class)
│   └── viz/            # Visualization utilities
├── data/
│   ├── processed/      # KG JSON, embeddings, trained models
│   └── schemas/        # Ontology definitions
├── configs/            # Model and pipeline configurations
├── scripts/            # CLI entry points
├── notebooks/          # Research and validation notebooks
├── outputs/            # Generated outputs (TDA, visualizations)
└── tests/              # Unit tests
```

---

## Validation

Embedding quality validated via:
- **t-SNE visualization** — Peptides cluster by mechanism class
- **Silhouette score** — Quantitative cluster quality
- **Nearest neighbor accuracy** — Similar peptides share targets/pathways

See `notebooks/05_embedding_validation.ipynb` for full analysis.

---

## What This Project Does NOT Do

- Does NOT provide dosing or protocol recommendations
- Does NOT suggest therapeutic interventions
- Does NOT claim efficacy for any compound
- Does NOT offer clinical decision support
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
