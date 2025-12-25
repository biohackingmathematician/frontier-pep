"""
Pydantic schemas for Peptide Atlas entities.

REMINDER: This project is for research and education only.
No dosing, no protocols, no therapeutic recommendations.
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

from peptide_atlas.constants import (
    AdministrationRoute,
    BindingType,
    Confidence,
    EffectDirection,
    EffectDomain,
    EvidenceSourceType,
    EvidenceTier,
    PathwayCategory,
    PeptideClass,
    RegulatoryStatus,
    Reversibility,
    RiskCategory,
    RiskSeverity,
    TargetType,
)


# =============================================================================
# NODE SCHEMAS
# =============================================================================


class PeptideNode(BaseModel):
    """
    A peptide entity in the knowledge graph.
    
    NOTE: No dosing or protocol information is stored.
    """
    
    id: UUID = Field(default_factory=uuid4)
    canonical_name: str = Field(..., description="Primary identifier name")
    synonyms: list[str] = Field(default_factory=list, description="Alternative names")
    
    # Classification
    peptide_class: PeptideClass
    subclass: Optional[str] = None
    
    # Molecular properties (optional)
    sequence: Optional[str] = Field(None, description="Amino acid sequence if known")
    molecular_weight_da: Optional[float] = Field(None, ge=0)
    
    # Regulatory status
    regulatory_status: RegulatoryStatus = RegulatoryStatus.UNKNOWN
    
    # Evidence classification (REQUIRED)
    evidence_tier: EvidenceTier = Field(
        ...,
        description="Highest quality evidence available for this peptide"
    )
    
    # Administration (general, not specific dosing)
    administration_routes: list[AdministrationRoute] = Field(default_factory=list)
    
    # Metadata
    description: Optional[str] = None
    notes: Optional[str] = None
    pubchem_cid: Optional[str] = None
    drugbank_id: Optional[str] = None
    uniprot_id: Optional[str] = None
    
    @field_validator("canonical_name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("canonical_name cannot be empty")
        return v.strip()
    
    class Config:
        use_enum_values = False


class TargetNode(BaseModel):
    """A molecular target (receptor, enzyme, tissue, etc.)."""
    
    id: UUID = Field(default_factory=uuid4)
    name: str
    target_type: TargetType
    
    # Molecular identifiers
    gene_symbol: Optional[str] = None
    uniprot_id: Optional[str] = None
    
    # For tissue/cell type targets
    cell_types: list[str] = Field(default_factory=list)
    tissues: list[str] = Field(default_factory=list)
    
    description: Optional[str] = None


class PathwayNode(BaseModel):
    """A biological signaling pathway."""
    
    id: UUID = Field(default_factory=uuid4)
    name: str
    category: PathwayCategory
    
    # Pathway identifiers
    reactome_id: Optional[str] = None
    kegg_id: Optional[str] = None
    
    # Components
    key_components: list[str] = Field(default_factory=list)
    
    description: Optional[str] = None


class EffectDomainNode(BaseModel):
    """An effect/outcome domain."""
    
    id: UUID = Field(default_factory=uuid4)
    name: str
    category: EffectDomain
    
    description: Optional[str] = None
    measurement_proxies: list[str] = Field(
        default_factory=list,
        description="How this effect is typically measured"
    )


class RiskNode(BaseModel):
    """An adverse effect/risk."""
    
    id: UUID = Field(default_factory=uuid4)
    name: str
    category: RiskCategory
    
    severity_typical: RiskSeverity = RiskSeverity.UNKNOWN
    reversibility: Reversibility = Reversibility.UNKNOWN
    
    description: Optional[str] = None
    mechanism_if_known: Optional[str] = None


class EvidenceSourceNode(BaseModel):
    """An evidence source (study, database, etc.)."""
    
    id: UUID = Field(default_factory=uuid4)
    source_type: EvidenceSourceType
    
    # Identifiers
    pubmed_id: Optional[str] = None
    doi: Optional[str] = None
    nct_id: Optional[str] = None  # ClinicalTrials.gov
    
    # Study characteristics
    title: Optional[str] = None
    year: Optional[int] = None
    population: Optional[str] = None
    sample_size: Optional[int] = Field(None, ge=0)
    
    quality_notes: Optional[str] = None


# =============================================================================
# EDGE SCHEMAS
# =============================================================================


class BindsEdge(BaseModel):
    """PEPTIDE -[BINDS]-> TARGET relationship."""
    
    source_id: UUID  # Peptide
    target_id: UUID  # Target
    
    binding_type: BindingType = BindingType.UNKNOWN
    affinity_value: Optional[float] = None
    affinity_unit: Optional[str] = None  # e.g., "nM", "uM"
    
    evidence_source_ids: list[UUID] = Field(default_factory=list)
    confidence: Confidence = Confidence.LOW


class ModulatesEdge(BaseModel):
    """PEPTIDE -[MODULATES]-> PATHWAY relationship."""
    
    source_id: UUID  # Peptide
    target_id: UUID  # Pathway
    
    direction: str = Field(..., pattern="^(activates|inhibits|modulates)$")
    magnitude: Optional[str] = Field(None, pattern="^(strong|moderate|weak|unknown)$")
    
    evidence_source_ids: list[UUID] = Field(default_factory=list)
    confidence: Confidence = Confidence.LOW


class AssociatedWithEffectEdge(BaseModel):
    """PEPTIDE -[ASSOCIATED_WITH_EFFECT]-> EFFECT_DOMAIN relationship."""
    
    source_id: UUID  # Peptide
    target_id: UUID  # EffectDomain
    
    direction: EffectDirection = EffectDirection.AMBIGUOUS
    effect_size: Optional[str] = Field(None, pattern="^(large|moderate|small|unknown)$")
    timescale: Optional[str] = Field(None, pattern="^(acute|short_term|long_term|unknown)$")
    
    # Evidence (REQUIRED for effects)
    evidence_tier: EvidenceTier
    evidence_source_ids: list[UUID] = Field(default_factory=list)
    confidence: Confidence = Confidence.LOW
    
    notes: Optional[str] = None


class AssociatedWithRiskEdge(BaseModel):
    """PEPTIDE -[ASSOCIATED_WITH_RISK]-> RISK relationship."""
    
    source_id: UUID  # Peptide
    target_id: UUID  # Risk
    
    frequency: Optional[str] = Field(None, pattern="^(common|uncommon|rare|unknown)$")
    dose_dependent: Optional[str] = Field(None, pattern="^(yes|no|unknown)$")
    
    # Evidence (REQUIRED for risks)
    evidence_tier: EvidenceTier
    evidence_source_ids: list[UUID] = Field(default_factory=list)
    confidence: Confidence = Confidence.LOW
    
    notes: Optional[str] = None


class PathwayInvolvedInEffectEdge(BaseModel):
    """PATHWAY -[INVOLVED_IN]-> EFFECT_DOMAIN relationship."""
    
    source_id: UUID  # Pathway
    target_id: UUID  # EffectDomain
    
    mechanism_notes: Optional[str] = None


class RiskLinkedToEdge(BaseModel):
    """RISK -[LINKED_TO]-> TARGET/PATHWAY relationship."""
    
    source_id: UUID  # Risk
    target_id: UUID  # Target or Pathway
    target_type: str = Field(..., pattern="^(target|pathway)$")
    
    mechanism_hypothesis: Optional[str] = None


# =============================================================================
# KNOWLEDGE GRAPH CONTAINER
# =============================================================================


class KnowledgeGraph(BaseModel):
    """Container for the full knowledge graph."""
    
    # Nodes
    peptides: list[PeptideNode] = Field(default_factory=list)
    targets: list[TargetNode] = Field(default_factory=list)
    pathways: list[PathwayNode] = Field(default_factory=list)
    effect_domains: list[EffectDomainNode] = Field(default_factory=list)
    risks: list[RiskNode] = Field(default_factory=list)
    evidence_sources: list[EvidenceSourceNode] = Field(default_factory=list)
    
    # Edges
    binds_edges: list[BindsEdge] = Field(default_factory=list)
    modulates_edges: list[ModulatesEdge] = Field(default_factory=list)
    effect_edges: list[AssociatedWithEffectEdge] = Field(default_factory=list)
    risk_edges: list[AssociatedWithRiskEdge] = Field(default_factory=list)
    pathway_effect_edges: list[PathwayInvolvedInEffectEdge] = Field(default_factory=list)
    risk_mechanism_edges: list[RiskLinkedToEdge] = Field(default_factory=list)
    
    def get_peptide_by_name(self, name: str) -> Optional[PeptideNode]:
        """Find peptide by canonical name or synonym."""
        name_lower = name.lower()
        for p in self.peptides:
            if p.canonical_name.lower() == name_lower:
                return p
            if any(s.lower() == name_lower for s in p.synonyms):
                return p
        return None
    
    def get_node_by_id(self, node_id: UUID) -> Optional[BaseModel]:
        """Find any node by ID."""
        for collection in [
            self.peptides, self.targets, self.pathways,
            self.effect_domains, self.risks, self.evidence_sources
        ]:
            for node in collection:
                if node.id == node_id:
                    return node
        return None
    
    @property
    def node_count(self) -> int:
        return (
            len(self.peptides) + len(self.targets) + len(self.pathways) +
            len(self.effect_domains) + len(self.risks) + len(self.evidence_sources)
        )
    
    @property
    def edge_count(self) -> int:
        return (
            len(self.binds_edges) + len(self.modulates_edges) +
            len(self.effect_edges) + len(self.risk_edges) +
            len(self.pathway_effect_edges) + len(self.risk_mechanism_edges)
        )

