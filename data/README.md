# Data Directory

**Author:** Agna Chan  
**Date:** December 2025  
**Repository:** [github.com/biohackingmathematician/frontier-pep](https://github.com/biohackingmathematician/frontier-pep)

## CRITICAL DISCLAIMER

**This data is for RESEARCH AND EDUCATIONAL PURPOSES ONLY.**

- The data does **NOT** constitute medical advice or treatment guidance.
- Inclusion of any peptide does **NOT** imply it is safe, effective, or recommended.
- Many peptides in this dataset are experimental, off-label, or not approved for human use.
- **NO dosing information** is provided or should be inferred.

---

## Directory Structure

```
data/
├── README.md           # This file
├── raw/               # Raw source data (gitignored)
│   └── .gitkeep
├── processed/         # Processed data files
│   └── .gitkeep
└── schemas/           # Schema definitions
    ├── ontology.yaml          # Knowledge graph schema
    └── evidence_taxonomy.yaml # Evidence classification
```

## Data Sources

### Curated Peptide Data

The peptide catalog is manually curated from:

1. **PubMed/MEDLINE** — Peer-reviewed literature
2. **DrugBank** — Drug and target information
3. **ClinicalTrials.gov** — Clinical trial data
4. **Regulatory labels** — FDA, EMA approved indications
5. **UniProt** — Protein sequence data
6. **Reactome/KEGG** — Pathway information

### Evidence Classification

All data is classified by evidence tier:

| Tier | Description | Data Quality |
|------|-------------|--------------|
| 1 | Regulatory approval | Highest |
| 2 | Phase II/III RCT | High |
| 3 | Early clinical | Moderate |
| 4 | Preclinical only | Limited |
| 5 | Mechanistic only | Theoretical |
| 6 | Anecdotal | Lowest |

## Data Files

### `processed/kg.json`

The main knowledge graph file containing:
- Peptide nodes with classifications
- Target nodes (receptors, tissues, etc.)
- Pathway nodes
- Effect domain nodes
- Risk nodes
- All relationship edges

### `processed/embeddings.pt`

PyTorch tensor file containing:
- Trained GNN embeddings for all nodes
- Hyperbolic (Poincaré) projections

### `schemas/ontology.yaml`

Defines the knowledge graph schema:
- Node types and required attributes
- Edge types and constraints
- Cardinality rules

### `schemas/evidence_taxonomy.yaml`

Defines evidence classification:
- Tier definitions
- Quality criteria
- Confidence scoring

## Adding New Data

When adding new peptides or relationships:

1. **Always include evidence tier** — Required for all entries
2. **Never include dosing** — This is prohibited
3. **Document sources** — Cite PubMed IDs, DOIs, etc.
4. **Review for accuracy** — Cross-reference multiple sources
5. **Include risks** — Every peptide must have associated risks

## Data Validation

Run validation with:

```bash
peptide-atlas validate-data --path data/processed/kg.json
```

This checks:
- Required fields present
- Evidence tiers specified
- No prohibited content (dosing, protocols)
- Schema compliance

---

**Remember: Data quality affects research conclusions. Always verify with primary sources.**

