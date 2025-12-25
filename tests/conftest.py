"""
Pytest fixtures for the Peptide Atlas tests.

REMINDER: This project is for research and education only.
"""

import pytest
import numpy as np

from peptide_atlas.data.schemas import (
    PeptideNode,
    TargetNode,
    PathwayNode,
    EffectDomainNode,
    RiskNode,
    KnowledgeGraph,
)
from peptide_atlas.constants import (
    PeptideClass,
    EvidenceTier,
    TargetType,
    PathwayCategory,
    EffectDomain,
    RiskCategory,
    RiskSeverity,
)


@pytest.fixture
def sample_peptide():
    """Create a sample peptide for testing."""
    return PeptideNode(
        canonical_name="Test Peptide",
        synonyms=["TP-1", "TestP"],
        peptide_class=PeptideClass.REGENERATIVE_REPAIR,
        evidence_tier=EvidenceTier.TIER_4_PRECLINICAL,
        description="A test peptide for unit testing.",
    )


@pytest.fixture
def sample_target():
    """Create a sample target for testing."""
    return TargetNode(
        name="Test Receptor",
        target_type=TargetType.RECEPTOR,
        description="A test receptor.",
    )


@pytest.fixture
def sample_pathway():
    """Create a sample pathway for testing."""
    return PathwayNode(
        name="Test Pathway",
        category=PathwayCategory.REPAIR_REGENERATION,
        description="A test pathway.",
    )


@pytest.fixture
def sample_effect():
    """Create a sample effect domain for testing."""
    return EffectDomainNode(
        name="Test Effect",
        category=EffectDomain.REGENERATION_REPAIR,
    )


@pytest.fixture
def sample_risk():
    """Create a sample risk for testing."""
    return RiskNode(
        name="Test Risk",
        category=RiskCategory.UNKNOWN_LONG_TERM,
        severity_typical=RiskSeverity.UNKNOWN,
    )


@pytest.fixture
def sample_knowledge_graph(sample_peptide, sample_target, sample_pathway, sample_effect, sample_risk):
    """Create a sample knowledge graph for testing."""
    return KnowledgeGraph(
        peptides=[sample_peptide],
        targets=[sample_target],
        pathways=[sample_pathway],
        effect_domains=[sample_effect],
        risks=[sample_risk],
    )


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    np.random.seed(42)
    return np.random.randn(10, 64).astype(np.float32)


@pytest.fixture
def sample_embeddings_2d():
    """Create sample 2D embeddings for testing."""
    np.random.seed(42)
    return np.random.randn(10, 2).astype(np.float32)

