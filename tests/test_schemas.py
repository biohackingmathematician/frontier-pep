"""
Tests for data schemas.

REMINDER: This project is for research and education only.
"""

import pytest
from uuid import UUID

from peptide_atlas.data.schemas import (
    PeptideNode,
    TargetNode,
    PathwayNode,
    EffectDomainNode,
    RiskNode,
    KnowledgeGraph,
    BindsEdge,
    AssociatedWithEffectEdge,
)
from peptide_atlas.constants import (
    PeptideClass,
    EvidenceTier,
    TargetType,
    PathwayCategory,
    EffectDomain,
    RiskCategory,
    BindingType,
    Confidence,
    EffectDirection,
)


class TestPeptideNode:
    """Tests for PeptideNode schema."""
    
    def test_create_peptide(self):
        """Test creating a peptide node."""
        peptide = PeptideNode(
            canonical_name="Test Peptide",
            peptide_class=PeptideClass.GHRH_ANALOG,
            evidence_tier=EvidenceTier.TIER_3_EARLY_CLINICAL,
        )
        
        assert peptide.canonical_name == "Test Peptide"
        assert peptide.peptide_class == PeptideClass.GHRH_ANALOG
        assert peptide.evidence_tier == EvidenceTier.TIER_3_EARLY_CLINICAL
        assert isinstance(peptide.id, UUID)
    
    def test_peptide_with_synonyms(self):
        """Test peptide with synonyms."""
        peptide = PeptideNode(
            canonical_name="Sermorelin",
            synonyms=["GRF 1-29", "Geref"],
            peptide_class=PeptideClass.GHRH_ANALOG,
            evidence_tier=EvidenceTier.TIER_2_LATE_CLINICAL,
        )
        
        assert len(peptide.synonyms) == 2
        assert "GRF 1-29" in peptide.synonyms
    
    def test_peptide_empty_name_fails(self):
        """Test that empty canonical name fails validation."""
        with pytest.raises(ValueError):
            PeptideNode(
                canonical_name="   ",
                peptide_class=PeptideClass.OTHER,
                evidence_tier=EvidenceTier.TIER_UNKNOWN,
            )
    
    def test_evidence_tier_confidence(self):
        """Test evidence tier confidence scores."""
        assert EvidenceTier.TIER_1_APPROVED.confidence_score == 1.0
        assert EvidenceTier.TIER_UNKNOWN.confidence_score == 0.05
        assert EvidenceTier.TIER_4_PRECLINICAL.confidence_score == 0.45


class TestTargetNode:
    """Tests for TargetNode schema."""
    
    def test_create_target(self):
        """Test creating a target node."""
        target = TargetNode(
            name="GHSR",
            target_type=TargetType.RECEPTOR,
            description="Ghrelin receptor",
        )
        
        assert target.name == "GHSR"
        assert target.target_type == TargetType.RECEPTOR


class TestKnowledgeGraph:
    """Tests for KnowledgeGraph container."""
    
    def test_create_empty_kg(self):
        """Test creating an empty knowledge graph."""
        kg = KnowledgeGraph()
        
        assert kg.node_count == 0
        assert kg.edge_count == 0
    
    def test_kg_node_count(self, sample_knowledge_graph):
        """Test node count calculation."""
        assert sample_knowledge_graph.node_count == 5
    
    def test_get_peptide_by_name(self, sample_knowledge_graph):
        """Test finding peptide by name."""
        peptide = sample_knowledge_graph.get_peptide_by_name("Test Peptide")
        assert peptide is not None
        assert peptide.canonical_name == "Test Peptide"
    
    def test_get_peptide_by_synonym(self, sample_peptide):
        """Test finding peptide by synonym."""
        kg = KnowledgeGraph(peptides=[sample_peptide])
        
        peptide = kg.get_peptide_by_name("TP-1")
        assert peptide is not None
        assert peptide.canonical_name == "Test Peptide"
    
    def test_get_node_by_id(self, sample_knowledge_graph, sample_peptide):
        """Test finding node by ID."""
        node = sample_knowledge_graph.get_node_by_id(sample_peptide.id)
        assert node is not None
        assert node.id == sample_peptide.id


class TestEdges:
    """Tests for edge schemas."""
    
    def test_binds_edge(self, sample_peptide, sample_target):
        """Test creating a binds edge."""
        edge = BindsEdge(
            source_id=sample_peptide.id,
            target_id=sample_target.id,
            binding_type=BindingType.AGONIST,
            confidence=Confidence.HIGH,
        )
        
        assert edge.source_id == sample_peptide.id
        assert edge.binding_type == BindingType.AGONIST
    
    def test_effect_edge_requires_evidence_tier(self, sample_peptide, sample_effect):
        """Test that effect edge requires evidence tier."""
        edge = AssociatedWithEffectEdge(
            source_id=sample_peptide.id,
            target_id=sample_effect.id,
            evidence_tier=EvidenceTier.TIER_4_PRECLINICAL,
            direction=EffectDirection.BENEFICIAL,
        )
        
        assert edge.evidence_tier == EvidenceTier.TIER_4_PRECLINICAL

