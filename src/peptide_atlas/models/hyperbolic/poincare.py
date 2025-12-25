"""
Poincaré ball operations for hyperbolic embeddings.

REMINDER: This project is for research and education only.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PoincareEmbedding(nn.Module):
    """
    Poincaré ball embedding layer.
    
    Maps points to the Poincaré ball (hyperbolic space) which is better
    suited for representing hierarchical relationships.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        curvature: float = 1.0,
        max_norm: float = 0.99,
    ):
        """
        Initialize Poincaré embeddings.
        
        Args:
            num_embeddings: Number of embedding vectors
            embedding_dim: Dimension of each embedding
            curvature: Curvature of the hyperbolic space (default 1.0)
            max_norm: Maximum norm for numerical stability
        """
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.curvature = curvature
        self.max_norm = max_norm
        
        # Initialize embeddings uniformly in a small ball
        embeddings = torch.randn(num_embeddings, embedding_dim) * 0.01
        self.embeddings = nn.Parameter(embeddings)
    
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for given indices.
        
        Args:
            indices: Indices of embeddings to retrieve [batch_size]
            
        Returns:
            Poincaré ball embeddings [batch_size, embedding_dim]
        """
        emb = self.embeddings[indices]
        return self._project_to_ball(emb)
    
    def _project_to_ball(self, x: torch.Tensor) -> torch.Tensor:
        """Project points to the Poincaré ball (ensure ||x|| < 1)."""
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        max_norm = self.max_norm
        
        # Clip norms that are too large
        cond = norm > max_norm
        x = torch.where(cond, x * max_norm / norm, x)
        
        return x
    
    def get_all_embeddings(self) -> torch.Tensor:
        """Get all embeddings projected to the ball."""
        return self._project_to_ball(self.embeddings)


class MobiusAddition(nn.Module):
    """Möbius addition in the Poincaré ball."""
    
    def __init__(self, curvature: float = 1.0):
        super().__init__()
        self.curvature = curvature
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute Möbius addition x ⊕ y.
        
        Args:
            x: First point [*, dim]
            y: Second point [*, dim]
            
        Returns:
            Möbius sum [*, dim]
        """
        c = self.curvature
        
        x_sq = torch.sum(x * x, dim=-1, keepdim=True)
        y_sq = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        
        num = (1 + 2 * c * xy + c * y_sq) * x + (1 - c * x_sq) * y
        denom = 1 + 2 * c * xy + c * c * x_sq * y_sq
        
        return num / (denom + 1e-8)


class ExponentialMap(nn.Module):
    """Exponential map from tangent space to Poincaré ball."""
    
    def __init__(self, curvature: float = 1.0):
        super().__init__()
        self.curvature = curvature
    
    def forward(
        self,
        v: torch.Tensor,
        x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Map tangent vector v at point x to the manifold.
        
        Args:
            v: Tangent vector [*, dim]
            x: Base point (default: origin) [*, dim]
            
        Returns:
            Point on manifold [*, dim]
        """
        if x is None:
            # Exponential map at origin
            return self._exp_map_zero(v)
        else:
            # Exponential map at x
            return self._exp_map(v, x)
    
    def _exp_map_zero(self, v: torch.Tensor) -> torch.Tensor:
        """Exponential map at origin."""
        c = self.curvature
        v_norm = torch.norm(v, p=2, dim=-1, keepdim=True)
        
        # tanh(sqrt(c) * ||v||) / (sqrt(c) * ||v||) * v
        sqrt_c = c ** 0.5
        coef = torch.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm + 1e-8)
        
        return coef * v
    
    def _exp_map(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Exponential map at point x."""
        c = self.curvature
        
        # Conformal factor
        x_sq = torch.sum(x * x, dim=-1, keepdim=True)
        lambda_x = 2 / (1 - c * x_sq + 1e-8)
        
        v_norm = torch.norm(v, p=2, dim=-1, keepdim=True)
        sqrt_c = c ** 0.5
        
        # Compute direction and magnitude
        direction = v / (v_norm + 1e-8)
        magnitude = torch.tanh(sqrt_c * lambda_x * v_norm / 2) / sqrt_c
        
        # Apply Möbius addition
        mobius = MobiusAddition(c)
        return mobius(x, magnitude * direction)


class LogarithmicMap(nn.Module):
    """Logarithmic map from Poincaré ball to tangent space."""
    
    def __init__(self, curvature: float = 1.0):
        super().__init__()
        self.curvature = curvature
    
    def forward(
        self,
        y: torch.Tensor,
        x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Map point y to tangent space at x.
        
        Args:
            y: Point on manifold [*, dim]
            x: Base point (default: origin) [*, dim]
            
        Returns:
            Tangent vector [*, dim]
        """
        if x is None:
            return self._log_map_zero(y)
        else:
            return self._log_map(y, x)
    
    def _log_map_zero(self, y: torch.Tensor) -> torch.Tensor:
        """Logarithmic map at origin."""
        c = self.curvature
        y_norm = torch.norm(y, p=2, dim=-1, keepdim=True)
        sqrt_c = c ** 0.5
        
        # arctanh(sqrt(c) * ||y||) / (sqrt(c) * ||y||) * y
        coef = torch.arctanh(sqrt_c * y_norm.clamp(max=1 - 1e-5))
        coef = coef / (sqrt_c * y_norm + 1e-8)
        
        return coef * y
    
    def _log_map(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Logarithmic map at point x."""
        c = self.curvature
        
        # Compute -x ⊕ y
        mobius = MobiusAddition(c)
        neg_x = -x
        diff = mobius(neg_x, y)
        
        # Conformal factor
        x_sq = torch.sum(x * x, dim=-1, keepdim=True)
        lambda_x = 2 / (1 - c * x_sq + 1e-8)
        
        diff_norm = torch.norm(diff, p=2, dim=-1, keepdim=True)
        sqrt_c = c ** 0.5
        
        coef = 2 / (sqrt_c * lambda_x) * torch.arctanh(
            sqrt_c * diff_norm.clamp(max=1 - 1e-5)
        )
        
        return coef * diff / (diff_norm + 1e-8)

