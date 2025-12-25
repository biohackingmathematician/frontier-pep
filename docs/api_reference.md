# API Reference

**Author:** Agna Chan  
**Date:** December 2025  
**Repository:** [github.com/biohackingmathematician/frontier-pep](https://github.com/biohackingmathematician/frontier-pep)

## CRITICAL DISCLAIMER

This API reference is for RESEARCH AND EDUCATIONAL PURPOSES ONLY.

---

## Data Module

### `peptide_atlas.data.schemas`

#### `PeptideNode`

```python
class PeptideNode(BaseModel):
    """A peptide entity in the knowledge graph."""
    
    id: UUID
    canonical_name: str  # Required
    synonyms: list[str]
    peptide_class: PeptideClass  # Required
    evidence_tier: EvidenceTier  # Required
    regulatory_status: RegulatoryStatus
    description: Optional[str]
    # ... other optional fields
```

#### `KnowledgeGraph`

```python
class KnowledgeGraph(BaseModel):
    """Container for the full knowledge graph."""
    
    peptides: list[PeptideNode]
    targets: list[TargetNode]
    pathways: list[PathwayNode]
    effect_domains: list[EffectDomainNode]
    risks: list[RiskNode]
    
    # Edges
    binds_edges: list[BindsEdge]
    modulates_edges: list[ModulatesEdge]
    effect_edges: list[AssociatedWithEffectEdge]
    risk_edges: list[AssociatedWithRiskEdge]
    
    def get_peptide_by_name(self, name: str) -> Optional[PeptideNode]: ...
    def get_node_by_id(self, node_id: UUID) -> Optional[BaseModel]: ...
    
    @property
    def node_count(self) -> int: ...
    
    @property
    def edge_count(self) -> int: ...
```

### `peptide_atlas.data.peptide_catalog`

```python
def get_curated_peptides() -> list[PeptideNode]:
    """Returns the curated list of peptides."""
    
def get_peptide_count_by_class() -> dict[PeptideClass, int]:
    """Returns count of peptides per class."""
    
def get_peptide_count_by_evidence_tier() -> dict[EvidenceTier, int]:
    """Returns count of peptides per evidence tier."""
```

## Knowledge Graph Module

### `peptide_atlas.kg.builder`

```python
class KnowledgeGraphBuilder:
    """Builds the peptide knowledge graph from curated data."""
    
    def build(self) -> KnowledgeGraph:
        """Build the complete knowledge graph."""
        
    def to_networkx(self) -> nx.MultiDiGraph:
        """Convert to NetworkX graph."""
        
    def save(self, path: Path) -> None:
        """Save knowledge graph to JSON."""
        
    @classmethod
    def load(cls, path: Path) -> KnowledgeGraph:
        """Load knowledge graph from JSON."""

def build_knowledge_graph() -> KnowledgeGraph:
    """Main entry point for building the knowledge graph."""
```

### `peptide_atlas.kg.queries`

```python
def get_peptides_by_class(kg, peptide_class) -> list[PeptideNode]: ...
def get_peptides_by_evidence_tier(kg, min_tier) -> list[PeptideNode]: ...
def get_peptides_sharing_target(kg, target_name) -> list[PeptideNode]: ...
def summarize_knowledge_graph(kg) -> dict: ...
```

## Models Module

### `peptide_atlas.models.gnn.encoder`

```python
class HeterogeneousGNNEncoder(nn.Module):
    """Heterogeneous Graph Neural Network encoder."""
    
    def __init__(
        self,
        config: GNNConfig,
        num_node_types: int,
        num_edge_types: int,
        node_feature_dims: Optional[dict[int, int]] = None,
    ): ...
    
    def forward(
        self,
        node_types: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        node_features: Optional[dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Encode the graph, returns node embeddings."""
```

### `peptide_atlas.models.hyperbolic`

```python
def poincare_distance(x, y, curvature=1.0) -> torch.Tensor:
    """Compute Poincaré distance between points."""

class EuclideanToPoincareProjection(nn.Module):
    """Projects Euclidean embeddings to the Poincaré ball."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

## TDA Module

### `peptide_atlas.tda.mapper`

```python
class MapperPipeline:
    """Complete Mapper pipeline."""
    
    def __init__(self, config: Optional[MapperConfig] = None): ...
    
    def fit(self, X, lens=None, lens_fn=None) -> MapperResult:
        """Apply Mapper to data."""
        
    def visualize_html(self, result, output_path, **kwargs) -> str:
        """Generate interactive HTML visualization."""
```

### `peptide_atlas.tda.persistence`

```python
class PersistentHomology:
    """Persistent homology computation."""
    
    def __init__(self, config: Optional[PersistenceConfig] = None): ...
    
    def fit(self, X: np.ndarray) -> PersistenceDiagram:
        """Compute persistent homology."""
        
    def plot_diagram(self, diagram, output_path=None, **kwargs):
        """Plot persistence diagram."""
```

## Visualization Module

### `peptide_atlas.viz.world_map`

```python
def create_world_map(
    embeddings_2d: np.ndarray,
    names: list[str],
    peptide_classes: list[str],
    evidence_tiers: list[str],
    descriptions: Optional[list[str]] = None,
    edges: Optional[list[tuple[int, int]]] = None,
    title: str = "Frontier Peptide Atlas",
    color_by: str = "peptide_class",
    theme: str = "dark",
    output_path: Optional[str] = None,
    show_disclaimer: bool = True,
) -> plotly.graph_objects.Figure:
    """Create an interactive world map visualization."""
```

## CLI

```bash
# Build knowledge graph
peptide-atlas build-kg --output data/processed/kg.json

# Train GNN
peptide-atlas train --config configs/model_config.yaml

# Run TDA
peptide-atlas analyze-tda --config configs/tda_config.yaml

# Generate visualization
peptide-atlas visualize --output outputs/world_map.html

# List peptides
peptide-atlas list-peptides --class ghrh_analog

# Validate KG
peptide-atlas validate data/processed/kg.json
```

---

**API for research use only.**

