# Methods

**Frontier Peptide Atlas — Technical Methods**

Author: Agna Chan  
Version: 0.1.0

---

## DISCLAIMER

This document describes research methodology for EDUCATIONAL PURPOSES ONLY.
No therapeutic recommendations are made.

---

## Overview

The Frontier Peptide Atlas is constructed through a multi-stage computational pipeline:

1. **Knowledge Curation** — Manual curation of peptides with evidence classification
2. **Graph Construction** — Heterogeneous knowledge graph linking entities
3. **Representation Learning** — Graph neural network embeddings
4. **Hyperbolic Projection** — Poincaré ball embeddings for hierarchy
5. **Topological Analysis** — Structure discovery via TDA
6. **Query Interface** — API for programmatic access

---

## 1. Knowledge Curation

### Peptide Selection Criteria

Peptides are included based on:
- Mechanistic novelty (distinct from well-characterized drug classes)
- Research interest (active investigation in literature)
- Availability of mechanistic data (target binding, pathway involvement)

### Evidence Classification

Every peptide-effect relationship receives an evidence tier:

| Tier | Criteria | Score |
|------|----------|-------|
| 1 | FDA/EMA approval for any indication | 1.00 |
| 2 | Positive Phase II/III RCT published | 0.85 |
| 3 | Phase I or early Phase II data | 0.65 |
| 4 | Preclinical (animal) data only | 0.45 |
| 5 | In vitro / mechanistic only | 0.25 |
| 6 | Anecdotal / case reports only | 0.15 |
| Unknown | Insufficient data to classify | 0.05 |

### Data Sources

- **PubMed/MEDLINE** — Primary literature
- **ClinicalTrials.gov** — Trial status and results
- **DrugBank** — Drug-target interactions
- **UniProt** — Protein sequences and annotations
- **Reactome/KEGG** — Pathway definitions

---

## 2. Knowledge Graph Schema

### Node Types

| Type | Description | Key Attributes |
|------|-------------|----------------|
| `Peptide` | Bioactive peptide compound | class, evidence_tier, regulatory_status |
| `Target` | Molecular target | target_type (receptor, tissue, etc.) |
| `Pathway` | Biological signaling pathway | category |
| `EffectDomain` | Outcome/effect category | category |
| `Risk` | Adverse effect | severity, reversibility |

### Edge Types

| Type | Source -> Target | Attributes |
|------|------------------|------------|
| `BINDS` | Peptide -> Target | binding_type, affinity |
| `MODULATES` | Peptide -> Pathway | direction, magnitude |
| `ASSOCIATED_WITH_EFFECT` | Peptide -> EffectDomain | evidence_tier, direction |
| `ASSOCIATED_WITH_RISK` | Peptide -> Risk | evidence_tier, frequency |

### Design Principles

1. **Evidence-first**: Every relationship has an evidence tier
2. **No dosing**: Dose information is explicitly prohibited
3. **Mechanism-focused**: Emphasize targets/pathways over outcomes
4. **Risk-aware**: Adverse effects are first-class entities

---

## 3. Graph Neural Network

### Architecture: Relational Graph Attention (R-GAT)

We use a heterogeneous GNN that:
- Maintains separate attention weights per edge type
- Aggregates information across multi-hop neighborhoods
- Produces L2-normalized embeddings for similarity search

**Configuration:**
- Hidden dimension: 128
- Embedding dimension: 64
- Layers: 3
- Attention heads: 4
- Dropout: 0.2

### Self-Supervised Pretraining

Three pretraining objectives:
1. **Edge prediction** — Predict edge existence and type
2. **Node attribute prediction** — Predict node type from embedding
3. **Masked node prediction** — Reconstruct masked node embeddings

---

## 4. Hyperbolic Embeddings

### Poincaré Ball Model

Peptide hierarchies (class -> subclass -> compound) are naturally tree-like. Hyperbolic space represents trees with less distortion than Euclidean space.

The Poincaré ball:

```
B^n = {x in R^n : ||x|| < 1}
```

### Projection

Euclidean GNN embeddings are projected to the Poincaré ball via the exponential map at the origin:

```
exp_0(v) = tanh(||v||) * v / ||v||
```

### Distance

Poincaré distance between points:

```
d(x, y) = arccosh(1 + 2 * ||x - y||^2 / ((1 - ||x||^2)(1 - ||y||^2)))
```

---

## 5. Topological Data Analysis

### Mapper Algorithm

The Mapper algorithm reveals cluster structure:
1. Project embeddings via filter function (UMAP, PCA, density)
2. Cover the filter space with overlapping intervals
3. Cluster points within each interval
4. Connect clusters sharing members

### Persistent Homology

Compute topological features at multiple scales:
- **H0** — Connected components (clusters)
- **H1** — Loops (mechanistic cycles)
- **H2** — Voids (unexplored regions)

Long-lived features in the persistence diagram indicate robust structure.

---

## 6. Query Interface

The Python API provides:

```python
atlas.query_by_class(peptide_class) -> List[Peptide]
atlas.query_by_evidence(min_tier) -> List[Peptide]
atlas.query_by_pathway(pathway_name) -> List[Peptide]
atlas.query_by_target(target_name) -> List[Peptide]
atlas.find_similar(peptide_name, k) -> List[Peptide]
atlas.find_bridges(class_a, class_b) -> List[Peptide]
```

---

## Limitations

1. **Curation bias** — Peptides are selected based on research interest, not exhaustiveness
2. **Evidence lag** — New clinical data may not be immediately reflected
3. **Relationship completeness** — Not all known relationships are captured
4. **Embedding quality** — Dependent on graph connectivity and training

---

## Reproducibility

All code, data, and trained models are available at:
https://github.com/biohackingmathematician/frontier-pep

```bash
# Reproduce from scratch
git clone https://github.com/biohackingmathematician/frontier-pep.git
cd frontier-pep
pip install -e ".[dev]"
python scripts/build_kg.py
python scripts/train_gnn.py --seed 42
python scripts/run_tda.py
```

---

**This is research methodology documentation, not a treatment guide.**
