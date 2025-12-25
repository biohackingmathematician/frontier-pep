"""
Self-supervised pretraining tasks for the GNN.

REMINDER: This project is for research and education only.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from peptide_atlas.models.gnn.config import PretrainingConfig
from peptide_atlas.models.gnn.encoder import HeterogeneousGNNEncoder


class EdgePredictionTask(nn.Module):
    """
    Edge prediction pretraining task.
    
    Predicts whether an edge exists between two nodes.
    """
    
    def __init__(self, embedding_dim: int, num_edge_types: int):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_edge_types + 1),  # +1 for "no edge"
        )
    
    def forward(
        self,
        src_emb: torch.Tensor,
        dst_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Predict edge type (including no edge)."""
        combined = torch.cat([src_emb, dst_emb], dim=-1)
        return self.decoder(combined)


class NodeAttributePredictionTask(nn.Module):
    """
    Node attribute prediction task.
    
    Predicts node type or other categorical attributes.
    """
    
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, num_classes),
        )
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Predict node attributes."""
        return self.classifier(embeddings)


class MaskedNodePredictionTask(nn.Module):
    """
    Masked node prediction task.
    
    Reconstructs masked node features from context.
    """
    
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        
        self.reconstructor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Reconstruct original embeddings."""
        return self.reconstructor(embeddings)


class PretrainingModule(nn.Module):
    """
    Combined pretraining module for the GNN encoder.
    """
    
    def __init__(
        self,
        encoder: HeterogeneousGNNEncoder,
        config: PretrainingConfig,
    ):
        super().__init__()
        
        self.encoder = encoder
        self.config = config
        
        embedding_dim = encoder.config.embedding_dim
        
        # Initialize tasks
        if config.edge_prediction:
            self.edge_pred_task = EdgePredictionTask(
                embedding_dim=embedding_dim,
                num_edge_types=encoder.num_edge_types,
            )
        else:
            self.edge_pred_task = None
        
        if config.node_attribute_prediction:
            self.node_attr_task = NodeAttributePredictionTask(
                embedding_dim=embedding_dim,
                num_classes=encoder.num_node_types,
            )
        else:
            self.node_attr_task = None
        
        if config.masked_node_prediction:
            self.masked_node_task = MaskedNodePredictionTask(
                embedding_dim=embedding_dim,
                hidden_dim=encoder.config.hidden_dim,
            )
        else:
            self.masked_node_task = None
    
    def compute_loss(
        self,
        node_types: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        node_features: Optional[dict[int, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined pretraining loss.
        
        Returns dict with individual loss components and total.
        """
        losses = {}
        
        # Get embeddings
        embeddings = self.encoder(node_types, edge_index, edge_type, node_features)
        
        # Edge prediction loss
        if self.edge_pred_task is not None:
            edge_loss = self._compute_edge_prediction_loss(
                embeddings, edge_index, edge_type
            )
            losses["edge_prediction"] = edge_loss * self.config.edge_pred_weight
        
        # Node attribute prediction loss
        if self.node_attr_task is not None:
            attr_loss = self._compute_node_attribute_loss(embeddings, node_types)
            losses["node_attribute"] = attr_loss * self.config.node_attr_weight
        
        # Masked node prediction loss
        if self.masked_node_task is not None:
            masked_loss = self._compute_masked_node_loss(
                node_types, edge_index, edge_type, node_features
            )
            losses["masked_node"] = masked_loss * self.config.masked_node_weight
        
        # Total loss
        losses["total"] = sum(losses.values())
        
        return losses
    
    def _compute_edge_prediction_loss(
        self,
        embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """Compute edge prediction loss with negative sampling."""
        num_nodes = embeddings.size(0)
        num_edges = edge_index.size(1)
        
        # Positive samples
        src_emb = embeddings[edge_index[0]]
        dst_emb = embeddings[edge_index[1]]
        pos_pred = self.edge_pred_task(src_emb, dst_emb)
        
        # Negative samples (random node pairs)
        num_neg = num_edges * self.config.negative_sampling_ratio
        neg_src = torch.randint(0, num_nodes, (num_neg,), device=embeddings.device)
        neg_dst = torch.randint(0, num_nodes, (num_neg,), device=embeddings.device)
        neg_src_emb = embeddings[neg_src]
        neg_dst_emb = embeddings[neg_dst]
        neg_pred = self.edge_pred_task(neg_src_emb, neg_dst_emb)
        
        # Labels: edge_type for positive, num_edge_types (no edge) for negative
        pos_labels = edge_type
        neg_labels = torch.full(
            (num_neg,), 
            self.encoder.num_edge_types,  # "no edge" class
            device=embeddings.device,
            dtype=torch.long,
        )
        
        # Combined loss
        all_pred = torch.cat([pos_pred, neg_pred], dim=0)
        all_labels = torch.cat([pos_labels, neg_labels], dim=0)
        
        return F.cross_entropy(all_pred, all_labels)
    
    def _compute_node_attribute_loss(
        self,
        embeddings: torch.Tensor,
        node_types: torch.Tensor,
    ) -> torch.Tensor:
        """Compute node type prediction loss."""
        pred = self.node_attr_task(embeddings)
        return F.cross_entropy(pred, node_types)
    
    def _compute_masked_node_loss(
        self,
        node_types: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        node_features: Optional[dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute masked node reconstruction loss."""
        num_nodes = node_types.size(0)
        
        # Create mask
        mask = torch.rand(num_nodes, device=node_types.device) < self.config.mask_ratio
        
        # Get original embeddings (without masking)
        with torch.no_grad():
            original_emb = self.encoder(node_types, edge_index, edge_type, node_features)
        
        # Mask node types (replace with special masked type)
        masked_types = node_types.clone()
        masked_types[mask] = 0  # Could use special token
        
        # Get embeddings with masking
        masked_emb = self.encoder(masked_types, edge_index, edge_type, node_features)
        
        # Reconstruct
        reconstructed = self.masked_node_task(masked_emb[mask])
        
        # MSE loss
        return F.mse_loss(reconstructed, original_emb[mask])

