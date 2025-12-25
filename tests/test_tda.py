"""
Tests for TDA functionality.

REMINDER: This project is for research and education only.
"""

import pytest
import numpy as np


class TestFilters:
    """Tests for filter functions."""
    
    def test_pca_filter(self, sample_embeddings):
        """Test PCA filter function."""
        from peptide_atlas.tda.filters import pca_filter
        
        result = pca_filter(sample_embeddings, n_components=2)
        
        assert result.shape == (10, 2)
    
    def test_density_filter(self, sample_embeddings):
        """Test density filter function."""
        from peptide_atlas.tda.filters import density_filter
        
        result = density_filter(sample_embeddings, n_neighbors=3)
        
        assert result.shape == (10, 1)
        assert np.all(result > 0)
    
    def test_eccentricity_filter(self, sample_embeddings):
        """Test eccentricity filter."""
        from peptide_atlas.tda.filters import eccentricity_filter
        
        result = eccentricity_filter(sample_embeddings)
        
        assert result.shape == (10, 1)
        assert np.all(result >= 0)
    
    def test_get_filter_function(self):
        """Test filter function factory."""
        from peptide_atlas.tda.filters import get_filter_function
        
        pca_fn = get_filter_function("pca")
        
        assert callable(pca_fn)
    
    def test_unknown_filter_raises(self):
        """Test that unknown filter raises error."""
        from peptide_atlas.tda.filters import get_filter_function
        
        with pytest.raises(ValueError):
            get_filter_function("unknown_filter")


class TestPersistence:
    """Tests for persistent homology."""
    
    def test_persistence_config(self):
        """Test persistence configuration."""
        from peptide_atlas.tda.persistence import PersistenceConfig
        
        config = PersistenceConfig(
            max_dimension=2,
            max_edge_length=2.0,
        )
        
        assert config.max_dimension == 2
        assert config.max_edge_length == 2.0
    
    @pytest.mark.skipif(
        True,  # Skip by default - requires ripser or gudhi
        reason="Persistence libraries not guaranteed",
    )
    def test_persistence_computation(self, sample_embeddings):
        """Test persistence computation."""
        from peptide_atlas.tda.persistence import PersistentHomology
        
        ph = PersistentHomology()
        diagram = ph.fit(sample_embeddings)
        
        assert diagram.h0 is not None
        assert len(diagram.dgms) > 0


class TestMapperAnalysis:
    """Tests for Mapper analysis."""
    
    def test_cluster_info_dataclass(self):
        """Test ClusterInfo dataclass."""
        from peptide_atlas.tda.analysis import ClusterInfo
        
        cluster = ClusterInfo(
            node_id="cluster_0",
            member_indices=[0, 1, 2],
            size=3,
        )
        
        assert cluster.size == 3
        assert len(cluster.member_indices) == 3
    
    def test_bridge_info_dataclass(self):
        """Test BridgeInfo dataclass."""
        from peptide_atlas.tda.analysis import BridgeInfo
        
        bridge = BridgeInfo(
            node_a="cluster_0",
            node_b="cluster_1",
            shared_indices=[1, 2],
            shared_count=2,
        )
        
        assert bridge.shared_count == 2

