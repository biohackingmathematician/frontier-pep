"""
Projection from Euclidean to hyperbolic space.

REMINDER: This project is for research and education only.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from peptide_atlas.models.hyperbolic.poincare import ExponentialMap


class EuclideanToPoincareProjection(nn.Module):
    """
    Projects Euclidean embeddings to the Poincaré ball.
    
    Uses a learned linear transformation followed by exponential map.
    """
    
    def __init__(
        self,
        euclidean_dim: int,
        hyperbolic_dim: int,
        curvature: float = 1.0,
        max_norm: float = 0.99,
    ):
        """
        Initialize projection layer.
        
        Args:
            euclidean_dim: Input Euclidean dimension
            hyperbolic_dim: Output hyperbolic dimension
            curvature: Curvature of the Poincaré ball
            max_norm: Maximum norm for stability
        """
        super().__init__()
        
        self.euclidean_dim = euclidean_dim
        self.hyperbolic_dim = hyperbolic_dim
        self.curvature = curvature
        self.max_norm = max_norm
        
        # Linear projection to tangent space
        self.linear = nn.Linear(euclidean_dim, hyperbolic_dim)
        
        # Exponential map
        self.exp_map = ExponentialMap(curvature)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project Euclidean vectors to Poincaré ball.
        
        Args:
            x: Euclidean embeddings [*, euclidean_dim]
            
        Returns:
            Poincaré embeddings [*, hyperbolic_dim]
        """
        # Project to tangent space at origin
        v = self.linear(x)
        
        # Scale for stability
        v = v * 0.1
        
        # Apply exponential map at origin
        h = self.exp_map(v)
        
        # Ensure within ball
        h = self._project_to_ball(h)
        
        return h
    
    def _project_to_ball(self, x: torch.Tensor) -> torch.Tensor:
        """Project to inside of Poincaré ball."""
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        cond = norm > self.max_norm
        return torch.where(cond, x * self.max_norm / norm, x)


class HybridEmbedding(nn.Module):
    """
    Combines Euclidean and hyperbolic embeddings.
    
    Useful for capturing both flat and hierarchical structure.
    """
    
    def __init__(
        self,
        euclidean_dim: int,
        hyperbolic_dim: int,
        output_dim: int,
        curvature: float = 1.0,
    ):
        """
        Initialize hybrid embedding.
        
        Args:
            euclidean_dim: Dimension of Euclidean component
            hyperbolic_dim: Dimension of hyperbolic component
            output_dim: Final output dimension
            curvature: Curvature for hyperbolic space
        """
        super().__init__()
        
        self.poincare_proj = EuclideanToPoincareProjection(
            euclidean_dim=euclidean_dim,
            hyperbolic_dim=hyperbolic_dim,
            curvature=curvature,
        )
        
        # Combine both embeddings
        self.combiner = nn.Linear(euclidean_dim + hyperbolic_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute hybrid embeddings.
        
        Args:
            x: Input Euclidean embeddings [*, dim]
            
        Returns:
            Tuple of (combined, euclidean, hyperbolic) embeddings
        """
        # Get hyperbolic projection
        h = self.poincare_proj(x)
        
        # Combine
        combined = torch.cat([x, h], dim=-1)
        output = self.combiner(combined)
        
        return output, x, h

