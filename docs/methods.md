# Methods

**Author:** Agna Chan  
**Date:** December 2025  
**Repository:** [github.com/biohackingmathematician/frontier-pep](https://github.com/biohackingmathematician/frontier-pep)

## CRITICAL DISCLAIMER

This documentation is for RESEARCH AND EDUCATIONAL PURPOSES ONLY.

- It does NOT constitute medical advice or treatment guidance.
- No dosing information or protocol recommendations are provided.

---

## Overview

The Frontier Peptide Atlas employs a multi-stage computational pipeline:

1. **Knowledge Graph Construction** — Heterogeneous graph of peptides, targets, pathways, effects, and risks
2. **Graph Neural Network Encoding** — Relational Graph Attention Network for learning embeddings
3. **Hyperbolic Projection** — Poincaré ball embeddings for hierarchical structure
4. **Topological Data Analysis** — Mapper algorithm and persistent homology

## Knowledge Graph Schema

### Node Types

| Type | Description | Key Attributes |
|------|-------------|----------------|
| Peptide | Bioactive peptide compound | class, evidence_tier, regulatory_status |
| Target | Molecular target (receptor, tissue) | target_type |
| Pathway | Biological signaling pathway | category |
| EffectDomain | Outcome/effect category | category |
| Risk | Adverse effect category | severity, reversibility |

### Edge Types

| Type | Source | Target | Key Attributes |
|------|--------|--------|----------------|
| BINDS | Peptide | Target | binding_type, affinity |
| MODULATES | Peptide | Pathway | direction, magnitude |
| ASSOCIATED_WITH_EFFECT | Peptide | EffectDomain | evidence_tier, direction |
| ASSOCIATED_WITH_RISK | Peptide | Risk | evidence_tier, frequency |

## Graph Neural Network

### Architecture

We use a Relational Graph Attention Network (R-GAT) that:

1. Maintains separate attention mechanisms per relation type
2. Aggregates messages across heterogeneous neighborhoods
3. Applies layer normalization and residual connections
4. Produces L2-normalized embeddings

### Self-Supervised Pretraining

Three pretraining tasks:

1. **Edge Prediction** — Predict edge existence and type
2. **Node Attribute Prediction** — Predict node type from embeddings
3. **Masked Node Prediction** — Reconstruct masked node embeddings

## Hyperbolic Embeddings

### Poincaré Ball Model

The Poincaré ball model represents hyperbolic space within the unit ball:

$$\mathbb{B}^n = \{x \in \mathbb{R}^n : ||x|| < 1\}$$

Key operations:
- **Möbius Addition**: $x \oplus y$
- **Exponential Map**: Project tangent vectors to manifold
- **Logarithmic Map**: Project manifold points to tangent space

### Why Hyperbolic?

Hierarchical relationships (e.g., peptide class → subclass → individual peptide) are better represented in hyperbolic space, which has exponentially more volume at the boundary.

## Topological Data Analysis

### Mapper Algorithm

1. Apply filter function (lens) to project data to low dimensions
2. Cover the filter space with overlapping intervals
3. Cluster points within each interval
4. Connect clusters with shared members

### Persistent Homology

Computes topological features at multiple scales:

- **H₀**: Connected components
- **H₁**: Loops/cycles
- **H₂**: Voids

Persistent features (long bars in persistence diagram) indicate robust topological structure.

## Evidence Classification

All peptide-effect and peptide-risk relationships require explicit evidence tier classification:

| Tier | Score | Description |
|------|-------|-------------|
| 1 | 1.0 | Regulatory approval |
| 2 | 0.85 | Late-stage clinical trials |
| 3 | 0.65 | Early clinical trials |
| 4 | 0.45 | Preclinical only |
| 5 | 0.25 | Mechanistic/in vitro |
| 6 | 0.15 | Anecdotal |

---

**Remember: This is a research methodology description, not a treatment guide.**

