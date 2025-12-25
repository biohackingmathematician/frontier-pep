"""
GNN layer implementations for the Peptide Atlas.

Includes Relational Graph Attention (R-GAT) layers for heterogeneous graphs.

REMINDER: This project is for research and education only.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationalGATLayer(nn.Module):
    """
    Relational Graph Attention Layer.
    
    Extends GAT to handle multiple edge types (relations) in a heterogeneous graph.
    Each relation type has its own attention mechanism.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_relations: int,
        num_heads: int = 4,
        dropout: float = 0.2,
        attention_dropout: float = 0.1,
        negative_slope: float = 0.2,
        residual: bool = True,
        layer_norm: bool = True,
    ):
        """
        Initialize R-GAT layer.
        
        Args:
            in_dim: Input feature dimension
            out_dim: Output feature dimension (per head)
            num_relations: Number of relation (edge) types
            num_heads: Number of attention heads
            dropout: Dropout rate for features
            attention_dropout: Dropout rate for attention weights
            negative_slope: LeakyReLU negative slope
            residual: Whether to use residual connection
            layer_norm: Whether to apply layer normalization
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.num_heads = num_heads
        self.negative_slope = negative_slope
        self.residual = residual
        
        # Linear transformations per relation
        self.W = nn.ModuleList([
            nn.Linear(in_dim, out_dim * num_heads, bias=False)
            for _ in range(num_relations)
        ])
        
        # Attention parameters per relation
        self.attn_src = nn.ParameterList([
            nn.Parameter(torch.Tensor(1, num_heads, out_dim))
            for _ in range(num_relations)
        ])
        self.attn_dst = nn.ParameterList([
            nn.Parameter(torch.Tensor(1, num_heads, out_dim))
            for _ in range(num_relations)
        ])
        
        # Bias
        self.bias = nn.Parameter(torch.Tensor(num_heads * out_dim))
        
        # Dropout layers
        self.feat_dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attention_dropout)
        
        # Optional layer norm
        self.layer_norm = nn.LayerNorm(num_heads * out_dim) if layer_norm else None
        
        # Residual projection if dimensions don't match
        if residual:
            if in_dim != num_heads * out_dim:
                self.res_proj = nn.Linear(in_dim, num_heads * out_dim, bias=False)
            else:
                self.res_proj = None
        else:
            self.res_proj = None
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters."""
        gain = nn.init.calculate_gain('relu')
        for w in self.W:
            nn.init.xavier_uniform_(w.weight, gain=gain)
        for attn in self.attn_src:
            nn.init.xavier_uniform_(attn, gain=gain)
        for attn in self.attn_dst:
            nn.init.xavier_uniform_(attn, gain=gain)
        nn.init.zeros_(self.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        size: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, in_dim] or tuple for bipartite
            edge_index: Edge indices [2, num_edges]
            edge_type: Edge types [num_edges]
            size: Size tuple for bipartite graphs
            
        Returns:
            Updated node features [num_nodes, num_heads * out_dim]
        """
        # Handle bipartite case
        if isinstance(x, tuple):
            x_src, x_dst = x
        else:
            x_src = x_dst = x
        
        num_nodes = x_dst.size(0)
        
        # Apply dropout to input features
        x_src = self.feat_dropout(x_src)
        x_dst = self.feat_dropout(x_dst)
        
        # Aggregate messages from each relation type
        out = torch.zeros(
            num_nodes, self.num_heads * self.out_dim,
            device=x_dst.device, dtype=x_dst.dtype
        )
        
        for rel_idx in range(self.num_relations):
            # Find edges of this relation type
            mask = edge_type == rel_idx
            if not mask.any():
                continue
            
            rel_edge_index = edge_index[:, mask]
            
            # Transform features
            h_src = self.W[rel_idx](x_src).view(-1, self.num_heads, self.out_dim)
            h_dst = self.W[rel_idx](x_dst).view(-1, self.num_heads, self.out_dim)
            
            # Compute attention scores
            src_idx = rel_edge_index[0]
            dst_idx = rel_edge_index[1]
            
            # Source and destination attention contributions
            e_src = (h_src[src_idx] * self.attn_src[rel_idx]).sum(dim=-1)
            e_dst = (h_dst[dst_idx] * self.attn_dst[rel_idx]).sum(dim=-1)
            e = F.leaky_relu(e_src + e_dst, negative_slope=self.negative_slope)
            
            # Softmax over incoming edges
            alpha = self._edge_softmax(e, dst_idx, num_nodes)
            alpha = self.attn_dropout(alpha)
            
            # Aggregate
            msg = alpha.unsqueeze(-1) * h_src[src_idx]  # [num_edges, num_heads, out_dim]
            out_rel = torch.zeros(
                num_nodes, self.num_heads, self.out_dim,
                device=x_dst.device, dtype=x_dst.dtype
            )
            out_rel.scatter_add_(0, dst_idx.unsqueeze(-1).unsqueeze(-1).expand_as(msg), msg)
            
            out += out_rel.view(num_nodes, -1)
        
        # Add bias
        out = out + self.bias
        
        # Residual connection
        if self.residual:
            if self.res_proj is not None:
                out = out + self.res_proj(x_dst)
            else:
                out = out + x_dst
        
        # Layer normalization
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        
        return out
    
    def _edge_softmax(
        self,
        e: torch.Tensor,
        dst_idx: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Compute softmax over edges grouped by destination node."""
        # Subtract max for numerical stability
        e_max = torch.zeros(num_nodes, e.size(1), device=e.device, dtype=e.dtype)
        e_max.scatter_reduce_(0, dst_idx.unsqueeze(-1).expand_as(e), e, reduce='amax')
        e = e - e_max[dst_idx]
        
        # Compute exp
        e_exp = torch.exp(e)
        
        # Sum of exp per node
        e_sum = torch.zeros(num_nodes, e.size(1), device=e.device, dtype=e.dtype)
        e_sum.scatter_add_(0, dst_idx.unsqueeze(-1).expand_as(e), e_exp)
        
        # Normalize
        alpha = e_exp / (e_sum[dst_idx] + 1e-8)
        
        return alpha


class HeterogeneousNodeEmbedding(nn.Module):
    """
    Learnable node type embeddings.
    
    Provides initial embeddings based on node type and features.
    """
    
    def __init__(
        self,
        num_node_types: int,
        embedding_dim: int,
        feature_dims: Optional[dict[int, int]] = None,
    ):
        """
        Initialize node embeddings.
        
        Args:
            num_node_types: Number of distinct node types
            embedding_dim: Output embedding dimension
            feature_dims: Dict mapping node type index to input feature dim
        """
        super().__init__()
        
        self.num_node_types = num_node_types
        self.embedding_dim = embedding_dim
        
        # Type embeddings
        self.type_embeddings = nn.Embedding(num_node_types, embedding_dim)
        
        # Feature projections per type (if features provided)
        if feature_dims is not None:
            self.feature_projections = nn.ModuleDict({
                str(type_idx): nn.Linear(feat_dim, embedding_dim)
                for type_idx, feat_dim in feature_dims.items()
            })
        else:
            self.feature_projections = None
    
    def forward(
        self,
        node_types: torch.Tensor,
        node_features: Optional[dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Compute initial node embeddings.
        
        Args:
            node_types: Node type indices [num_nodes]
            node_features: Optional dict mapping type to features
            
        Returns:
            Node embeddings [num_nodes, embedding_dim]
        """
        # Get type embeddings
        embeddings = self.type_embeddings(node_types)
        
        # Add feature projections if available
        if self.feature_projections is not None and node_features is not None:
            for type_idx, features in node_features.items():
                type_mask = node_types == type_idx
                if type_mask.any() and str(type_idx) in self.feature_projections:
                    proj = self.feature_projections[str(type_idx)](features)
                    embeddings[type_mask] = embeddings[type_mask] + proj
        
        return embeddings

