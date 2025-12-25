"""
Full GNN encoder for the Peptide Atlas.

Combines R-GAT layers into a complete heterogeneous graph encoder.

REMINDER: This project is for research and education only.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from peptide_atlas.models.gnn.config import GNNConfig
from peptide_atlas.models.gnn.layers import HeterogeneousNodeEmbedding, RelationalGATLayer


class HeterogeneousGNNEncoder(nn.Module):
    """
    Heterogeneous Graph Neural Network encoder.
    
    Processes a knowledge graph with multiple node and edge types
    to produce node embeddings suitable for downstream tasks.
    """
    
    def __init__(
        self,
        config: GNNConfig,
        num_node_types: int,
        num_edge_types: int,
        node_feature_dims: Optional[dict[int, int]] = None,
    ):
        """
        Initialize the encoder.
        
        Args:
            config: GNN configuration
            num_node_types: Number of distinct node types
            num_edge_types: Number of distinct edge/relation types
            node_feature_dims: Dict mapping node type to feature dimension
        """
        super().__init__()
        
        self.config = config
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        
        # Initial node embeddings
        self.node_embedding = HeterogeneousNodeEmbedding(
            num_node_types=num_node_types,
            embedding_dim=config.hidden_dim,
            feature_dims=node_feature_dims,
        )
        
        # R-GAT layers
        self.layers = nn.ModuleList()
        
        for i in range(config.num_layers):
            in_dim = config.hidden_dim
            out_dim = config.hidden_dim // config.num_heads
            
            # Last layer outputs embedding_dim
            if i == config.num_layers - 1:
                out_dim = config.embedding_dim // config.num_heads
            
            layer = RelationalGATLayer(
                in_dim=in_dim,
                out_dim=out_dim,
                num_relations=num_edge_types,
                num_heads=config.num_heads,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                residual=True,
                layer_norm=config.layer_norm,
            )
            self.layers.append(layer)
        
        # Activation function
        self.activation = self._get_activation(config.activation)
        
        # Final projection
        final_dim = config.embedding_dim if config.num_layers > 0 else config.hidden_dim
        self.output_projection = nn.Linear(final_dim, config.embedding_dim)
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
            "leaky_relu": nn.LeakyReLU(0.2),
        }
        return activations.get(name, nn.ELU())
    
    def forward(
        self,
        node_types: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        node_features: Optional[dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Encode the graph.
        
        Args:
            node_types: Node type indices [num_nodes]
            edge_index: Edge indices [2, num_edges]
            edge_type: Edge type indices [num_edges]
            node_features: Optional dict mapping type to features
            
        Returns:
            Node embeddings [num_nodes, embedding_dim]
        """
        # Get initial embeddings
        x = self.node_embedding(node_types, node_features)
        
        # Apply R-GAT layers
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_type)
            
            # Apply activation (except last layer)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        
        # Final projection
        x = self.output_projection(x)
        
        # L2 normalize embeddings
        x = F.normalize(x, p=2, dim=-1)
        
        return x
    
    def get_peptide_embeddings(
        self,
        node_types: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        peptide_mask: torch.Tensor,
        node_features: Optional[dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Get embeddings for peptide nodes only.
        
        Args:
            node_types: Node type indices [num_nodes]
            edge_index: Edge indices [2, num_edges]
            edge_type: Edge type indices [num_edges]
            peptide_mask: Boolean mask for peptide nodes [num_nodes]
            node_features: Optional dict mapping type to features
            
        Returns:
            Peptide embeddings [num_peptides, embedding_dim]
        """
        all_embeddings = self.forward(node_types, edge_index, edge_type, node_features)
        return all_embeddings[peptide_mask]


class EmbeddingMLPDecoder(nn.Module):
    """
    MLP decoder for link prediction and other tasks.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(
        self,
        src_embeddings: torch.Tensor,
        dst_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict link probability.
        
        Args:
            src_embeddings: Source node embeddings [batch, dim]
            dst_embeddings: Destination node embeddings [batch, dim]
            
        Returns:
            Predictions [batch, num_classes]
        """
        combined = torch.cat([src_embeddings, dst_embeddings], dim=-1)
        return self.mlp(combined)

