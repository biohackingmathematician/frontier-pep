"""
Tests for data validators.

REMINDER: This project is for research and education only.
"""

import pytest

from peptide_atlas.data.validators import (
    check_no_dosing_info,
    validate_peptide,
    validate_knowledge_graph,
    validate_no_prohibited_content,
)
from peptide_atlas.data.schemas import PeptideNode
from peptide_atlas.constants import PeptideClass, EvidenceTier


class TestDosingDetection:
    """Tests for dosing information detection."""
    
    def test_detects_mg_dosing(self):
        """Test detection of mg dosing."""
        text = "Administer 5 mg subcutaneously"
        violations = check_no_dosing_info(text)
        
        assert len(violations) > 0
    
    def test_detects_iu_dosing(self):
        """Test detection of IU dosing."""
        text = "Use 100 IU daily"
        violations = check_no_dosing_info(text)
        
        assert len(violations) > 0
    
    def test_detects_frequency(self):
        """Test detection of dosing frequency."""
        text = "Take 3x per day for best results"
        violations = check_no_dosing_info(text)
        
        assert len(violations) > 0
    
    def test_detects_cycle(self):
        """Test detection of cycle information."""
        text = "Run a 12-week cycle with PCT"
        violations = check_no_dosing_info(text)
        
        assert len(violations) > 0
    
    def test_allows_clean_text(self):
        """Test that clean text passes."""
        text = "This peptide activates the GH receptor in preclinical studies."
        violations = check_no_dosing_info(text)
        
        assert len(violations) == 0
    
    def test_allows_none(self):
        """Test that None text passes."""
        violations = check_no_dosing_info(None)
        
        assert len(violations) == 0


class TestPeptideValidation:
    """Tests for peptide validation."""
    
    def test_valid_peptide_passes(self, sample_peptide):
        """Test that valid peptide passes validation."""
        result = validate_peptide(sample_peptide)
        
        assert result.is_valid
    
    def test_unknown_tier_warns(self):
        """Test that unknown tier generates warning."""
        peptide = PeptideNode(
            canonical_name="Unknown Peptide",
            peptide_class=PeptideClass.OTHER,
            evidence_tier=EvidenceTier.TIER_UNKNOWN,
        )
        
        result = validate_peptide(peptide)
        
        assert len(result.warnings) > 0
    
    def test_dosing_in_description_fails(self):
        """Test that dosing in description fails."""
        peptide = PeptideNode(
            canonical_name="Bad Peptide",
            peptide_class=PeptideClass.OTHER,
            evidence_tier=EvidenceTier.TIER_4_PRECLINICAL,
            description="Inject 5 mg daily for optimal results.",
        )
        
        result = validate_peptide(peptide)
        
        assert not result.is_valid
        assert len(result.errors) > 0


class TestKnowledgeGraphValidation:
    """Tests for knowledge graph validation."""
    
    def test_valid_kg_passes(self, sample_knowledge_graph):
        """Test that valid KG passes validation."""
        result = validate_knowledge_graph(sample_knowledge_graph)
        
        assert result.is_valid
    
    def test_empty_kg_passes(self):
        """Test that empty KG passes validation."""
        from peptide_atlas.data.schemas import KnowledgeGraph
        
        kg = KnowledgeGraph()
        result = validate_knowledge_graph(kg)
        
        assert result.is_valid

