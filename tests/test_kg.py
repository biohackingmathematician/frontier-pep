"""
Tests for knowledge graph building and queries.

REMINDER: This project is for research and education only.
"""

import pytest
import tempfile
from pathlib import Path

from peptide_atlas.kg.builder import KnowledgeGraphBuilder, build_knowledge_graph
from peptide_atlas.kg.queries import (
    get_peptides_by_class,
    get_peptides_by_evidence_tier,
    get_peptides_sharing_target,
    summarize_knowledge_graph,
)
from peptide_atlas.constants import PeptideClass, EvidenceTier


class TestKnowledgeGraphBuilder:
    """Tests for the KnowledgeGraphBuilder."""
    
    def test_build_creates_peptides(self):
        """Test that build creates peptides."""
        builder = KnowledgeGraphBuilder()
        kg = builder.build()
        
        assert len(kg.peptides) > 0
    
    def test_build_creates_targets(self):
        """Test that build creates targets."""
        builder = KnowledgeGraphBuilder()
        kg = builder.build()
        
        assert len(kg.targets) > 0
    
    def test_build_creates_pathways(self):
        """Test that build creates pathways."""
        builder = KnowledgeGraphBuilder()
        kg = builder.build()
        
        assert len(kg.pathways) > 0
    
    def test_build_creates_relationships(self):
        """Test that build creates relationships."""
        builder = KnowledgeGraphBuilder()
        kg = builder.build()
        
        assert len(kg.binds_edges) > 0
        assert len(kg.modulates_edges) > 0
        assert len(kg.effect_edges) > 0
        assert len(kg.risk_edges) > 0
    
    def test_all_peptides_have_evidence_tier(self):
        """Test that all peptides have an evidence tier."""
        builder = KnowledgeGraphBuilder()
        kg = builder.build()
        
        for peptide in kg.peptides:
            assert peptide.evidence_tier is not None
    
    def test_all_effect_edges_have_evidence_tier(self):
        """Test that all effect edges have evidence tier."""
        builder = KnowledgeGraphBuilder()
        kg = builder.build()
        
        for edge in kg.effect_edges:
            assert edge.evidence_tier is not None
    
    def test_save_and_load(self):
        """Test saving and loading knowledge graph."""
        builder = KnowledgeGraphBuilder()
        kg = builder.build()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "kg.json"
            builder.save(path)
            
            loaded = KnowledgeGraphBuilder.load(path)
            
            assert loaded.node_count == kg.node_count
            assert loaded.edge_count == kg.edge_count
    
    def test_to_networkx(self):
        """Test conversion to NetworkX graph."""
        builder = KnowledgeGraphBuilder()
        kg = builder.build()
        
        G = builder.to_networkx()
        
        assert G.number_of_nodes() == kg.node_count
        assert G.number_of_edges() > 0


class TestKnowledgeGraphQueries:
    """Tests for knowledge graph queries."""
    
    @pytest.fixture
    def kg(self):
        """Build a knowledge graph for testing."""
        return build_knowledge_graph()
    
    def test_get_peptides_by_class(self, kg):
        """Test filtering peptides by class."""
        ghrh = get_peptides_by_class(kg, PeptideClass.GHRH_ANALOG)
        
        assert len(ghrh) > 0
        for p in ghrh:
            assert p.peptide_class == PeptideClass.GHRH_ANALOG
    
    def test_get_peptides_by_evidence_tier(self, kg):
        """Test filtering peptides by evidence tier."""
        approved = get_peptides_by_evidence_tier(kg, EvidenceTier.TIER_1_APPROVED)
        
        assert len(approved) > 0
        for p in approved:
            assert p.evidence_tier.confidence_score >= EvidenceTier.TIER_1_APPROVED.confidence_score
    
    def test_get_peptides_sharing_target(self, kg):
        """Test finding peptides sharing a target."""
        ghsr_peptides = get_peptides_sharing_target(kg, "GHSR")
        
        assert len(ghsr_peptides) > 0
    
    def test_summarize(self, kg):
        """Test knowledge graph summary."""
        summary = summarize_knowledge_graph(kg)
        
        assert "nodes" in summary
        assert "edges" in summary
        assert summary["nodes"]["peptides"] > 0
        assert summary["edges"]["total"] > 0

