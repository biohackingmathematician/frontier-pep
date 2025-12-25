"""
Tests for GNN and hyperbolic models.

REMINDER: This project is for research and education only.
"""

import pytest
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


class TestGNNLayers:
    """Tests for GNN layer implementations."""
    
    def test_relational_gat_layer(self):
        """Test R-GAT layer forward pass."""
        from peptide_atlas.models.gnn.layers import RelationalGATLayer
        
        layer = RelationalGATLayer(
            in_dim=32,
            out_dim=16,
            num_relations=4,
            num_heads=2,
        )
        
        # Create test data
        x = torch.randn(10, 32)
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 5],
        ])
        edge_type = torch.tensor([0, 1, 2, 3, 0])
        
        out = layer(x, edge_index, edge_type)
        
        assert out.shape == (10, 16 * 2)  # num_heads * out_dim
    
    def test_heterogeneous_node_embedding(self):
        """Test node embedding layer."""
        from peptide_atlas.models.gnn.layers import HeterogeneousNodeEmbedding
        
        layer = HeterogeneousNodeEmbedding(
            num_node_types=5,
            embedding_dim=32,
        )
        
        node_types = torch.tensor([0, 1, 2, 3, 4, 0, 1])
        
        out = layer(node_types)
        
        assert out.shape == (7, 32)


class TestGNNEncoder:
    """Tests for the full GNN encoder."""
    
    def test_encoder_forward(self):
        """Test encoder forward pass."""
        from peptide_atlas.models.gnn.encoder import HeterogeneousGNNEncoder
        from peptide_atlas.models.gnn.config import GNNConfig
        
        config = GNNConfig(
            hidden_dim=64,
            embedding_dim=32,
            num_layers=2,
            num_heads=2,
        )
        
        encoder = HeterogeneousGNNEncoder(
            config=config,
            num_node_types=5,
            num_edge_types=4,
        )
        
        node_types = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 6],
        ])
        edge_type = torch.tensor([0, 1, 2, 3, 0, 1])
        
        out = encoder(node_types, edge_index, edge_type)
        
        assert out.shape == (8, 32)
        
        # Check L2 normalized
        norms = torch.norm(out, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestHyperbolicOperations:
    """Tests for hyperbolic geometry operations."""
    
    def test_poincare_distance(self):
        """Test Poincaré distance computation."""
        from peptide_atlas.models.hyperbolic.distance import poincare_distance
        
        x = torch.tensor([[0.1, 0.2], [0.0, 0.0]])
        y = torch.tensor([[0.3, 0.4], [0.5, 0.5]])
        
        dist = poincare_distance(x, y)
        
        assert dist.shape == (2,)
        assert torch.all(dist >= 0)
    
    def test_poincare_distance_matrix(self):
        """Test pairwise Poincaré distance."""
        from peptide_atlas.models.hyperbolic.distance import poincare_distance_matrix
        
        x = torch.randn(5, 2) * 0.1
        y = torch.randn(3, 2) * 0.1
        
        dist_matrix = poincare_distance_matrix(x, y)
        
        assert dist_matrix.shape == (5, 3)
        assert torch.all(dist_matrix >= 0)
    
    def test_exponential_map(self):
        """Test exponential map at origin."""
        from peptide_atlas.models.hyperbolic.poincare import ExponentialMap
        
        exp_map = ExponentialMap(curvature=1.0)
        
        v = torch.randn(10, 4) * 0.1
        
        result = exp_map(v)
        
        # Result should be in the ball (norm < 1)
        norms = torch.norm(result, dim=1)
        assert torch.all(norms < 1.0)
    
    def test_projection_to_poincare(self):
        """Test Euclidean to Poincaré projection."""
        from peptide_atlas.models.hyperbolic.projection import EuclideanToPoincareProjection
        
        proj = EuclideanToPoincareProjection(
            euclidean_dim=32,
            hyperbolic_dim=16,
        )
        
        x = torch.randn(10, 32)
        h = proj(x)
        
        assert h.shape == (10, 16)
        
        # Check in ball
        norms = torch.norm(h, dim=1)
        assert torch.all(norms < 1.0)

