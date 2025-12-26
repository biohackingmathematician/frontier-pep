"""
Poincaré ball embedding for the Peptide Atlas.

Projects Euclidean embeddings from the GNN to the Poincaré ball,
which better represents hierarchical relationships.

REMINDER: This project is for research and education only.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def project_to_poincare(
    x: torch.Tensor,
    max_norm: float = 0.99,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Project points to the Poincaré ball.
    
    Args:
        x: Points to project [*, dim]
        max_norm: Maximum norm (must be < 1 for stability)
        eps: Small value for numerical stability
        
    Returns:
        Projected points inside the unit ball
    """
    norm = torch.norm(x, p=2, dim=-1, keepdim=True).clamp(min=eps)
    cond = norm > max_norm
    return torch.where(cond, x * max_norm / norm, x)


def exponential_map(
    v: torch.Tensor,
    base_point: Optional[torch.Tensor] = None,
    curvature: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Exponential map from tangent space to Poincaré ball.
    
    Maps tangent vectors at base_point to points on the manifold.
    
    Args:
        v: Tangent vectors [*, dim]
        base_point: Base point (default: origin)
        curvature: Curvature of the hyperbolic space
        eps: Small value for numerical stability
        
    Returns:
        Points on the Poincaré ball
    """
    c = curvature
    sqrt_c = c ** 0.5
    
    if base_point is None:
        # Exponential map at origin
        v_norm = torch.norm(v, p=2, dim=-1, keepdim=True).clamp(min=eps)
        return torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)
    
    # General exponential map
    v_norm = torch.norm(v, p=2, dim=-1, keepdim=True).clamp(min=eps)
    
    # Conformal factor at base point
    base_sq = torch.sum(base_point * base_point, dim=-1, keepdim=True)
    lambda_x = 2 / (1 - c * base_sq + eps)
    
    # Compute exponential map
    second_term = torch.tanh(sqrt_c * lambda_x * v_norm / 2) * v / (sqrt_c * v_norm)
    
    # Möbius addition with base point
    result = mobius_addition(base_point, second_term, curvature)
    
    return project_to_poincare(result)


def logarithmic_map(
    y: torch.Tensor,
    base_point: Optional[torch.Tensor] = None,
    curvature: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Logarithmic map from Poincaré ball to tangent space.
    
    Maps points on the manifold to tangent vectors at base_point.
    
    Args:
        y: Points on the Poincaré ball [*, dim]
        base_point: Base point (default: origin)
        curvature: Curvature of the hyperbolic space
        eps: Small value for numerical stability
        
    Returns:
        Tangent vectors at base_point
    """
    c = curvature
    sqrt_c = c ** 0.5
    
    if base_point is None:
        # Logarithmic map at origin
        y_norm = torch.norm(y, p=2, dim=-1, keepdim=True).clamp(min=eps)
        return torch.atanh(sqrt_c * y_norm.clamp(max=1-eps)) * y / (sqrt_c * y_norm)
    
    # Compute (-base_point) ⊕ y
    diff = mobius_addition(-base_point, y, curvature)
    diff_norm = torch.norm(diff, p=2, dim=-1, keepdim=True).clamp(min=eps)
    
    # Conformal factor at base point
    base_sq = torch.sum(base_point * base_point, dim=-1, keepdim=True)
    lambda_x = 2 / (1 - c * base_sq + eps)
    
    # Compute logarithmic map
    result = (2 / (sqrt_c * lambda_x)) * torch.atanh(sqrt_c * diff_norm.clamp(max=1-eps)) * diff / diff_norm
    
    return result


def mobius_addition(
    x: torch.Tensor,
    y: torch.Tensor,
    curvature: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Möbius addition in the Poincaré ball.
    
    This is the generalization of vector addition to hyperbolic space.
    
    Args:
        x: First operand [*, dim]
        y: Second operand [*, dim]
        curvature: Curvature of the hyperbolic space
        eps: Small value for numerical stability
        
    Returns:
        x ⊕ y
    """
    c = curvature
    
    x_sq = torch.sum(x * x, dim=-1, keepdim=True)
    y_sq = torch.sum(y * y, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    
    num = (1 + 2 * c * xy + c * y_sq) * x + (1 - c * x_sq) * y
    denom = 1 + 2 * c * xy + c * c * x_sq * y_sq
    
    return num / (denom + eps)


class PoincareEmbedding(nn.Module):
    """
    Projects Euclidean embeddings to the Poincaré ball.
    
    The Poincaré ball is a model of hyperbolic space that is particularly
    suited for representing hierarchical data, as it has more "room" near
    the boundary for leaf nodes while keeping parent nodes near the center.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: Optional[int] = None,
        curvature: float = 1.0,
        max_norm: float = 0.99,
        learnable_curvature: bool = False,
    ):
        """
        Initialize Poincaré embedding layer.
        
        Args:
            input_dim: Dimension of input Euclidean embeddings
            output_dim: Dimension of output hyperbolic embeddings (default: same as input)
            curvature: Curvature of the hyperbolic space (default: 1.0)
            max_norm: Maximum norm for numerical stability (must be < 1)
            learnable_curvature: Whether to learn the curvature parameter
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        self.max_norm = max_norm
        
        # Curvature parameter
        if learnable_curvature:
            self.curvature = nn.Parameter(torch.tensor(curvature))
        else:
            self.register_buffer("curvature", torch.tensor(curvature))
        
        # Optional projection layer if dimensions differ
        if self.input_dim != self.output_dim:
            self.projection = nn.Linear(input_dim, self.output_dim, bias=False)
        else:
            self.projection = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project Euclidean embeddings to Poincaré ball.
        
        Args:
            x: Euclidean embeddings [batch_size, input_dim]
            
        Returns:
            Hyperbolic embeddings [batch_size, output_dim]
        """
        # Optional linear projection
        if self.projection is not None:
            x = self.projection(x)
        
        # Project to Poincaré ball using exponential map at origin
        # First normalize to tangent space
        x_tangent = x / (1 + torch.norm(x, dim=-1, keepdim=True))
        
        # Apply exponential map
        x_hyp = exponential_map(
            x_tangent,
            base_point=None,  # Map from origin
            curvature=self.curvature.item() if isinstance(self.curvature, torch.Tensor) else self.curvature,
        )
        
        # Ensure we stay inside the ball
        x_hyp = project_to_poincare(x_hyp, max_norm=self.max_norm)
        
        return x_hyp
    
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise Poincaré distances.
        
        Args:
            x: First set of points [batch_size, dim]
            y: Second set of points [batch_size, dim]
            
        Returns:
            Distances [batch_size]
        """
        from peptide_atlas.models.hyperbolic.distance import poincare_distance
        curv = self.curvature.item() if isinstance(self.curvature, torch.Tensor) else self.curvature
        return poincare_distance(x, y, curvature=curv)
    
    def centroid(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the Einstein midpoint (hyperbolic centroid).
        
        Args:
            x: Points in the Poincaré ball [n_points, dim]
            weights: Optional weights for each point [n_points]
            
        Returns:
            Centroid [dim]
        """
        if weights is None:
            weights = torch.ones(x.shape[0], device=x.device)
        
        weights = weights / weights.sum()
        
        curv = self.curvature.item() if isinstance(self.curvature, torch.Tensor) else self.curvature
        
        # Einstein midpoint formula
        gamma = 1 / torch.sqrt(1 - curv * torch.sum(x * x, dim=-1) + 1e-8)  # Lorentz factors
        
        weighted_sum = (weights * gamma).unsqueeze(-1) * x
        weighted_sum = weighted_sum.sum(dim=0)
        
        gamma_sum = (weights * gamma).sum()
        
        centroid = weighted_sum / (gamma_sum + 1e-8)
        
        # Project back to ball
        return project_to_poincare(centroid.unsqueeze(0), self.max_norm).squeeze(0)


class HyperbolicMLR(nn.Module):
    """
    Multinomial Logistic Regression in hyperbolic space.
    
    Useful for classification tasks on hyperbolic embeddings.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        curvature: float = 1.0,
    ):
        """
        Initialize hyperbolic MLR.
        
        Args:
            input_dim: Dimension of hyperbolic embeddings
            num_classes: Number of output classes
            curvature: Curvature of hyperbolic space
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.curvature = curvature
        
        # Class representatives (points on the ball)
        self.p = nn.Parameter(torch.zeros(num_classes, input_dim))
        
        # Hyperplane normals in tangent space
        self.a = nn.Parameter(torch.randn(num_classes, input_dim) * 0.01)
        
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters with small values inside the ball."""
        nn.init.uniform_(self.p, -0.001, 0.001)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute class logits for hyperbolic embeddings.
        
        Args:
            x: Hyperbolic embeddings [batch_size, input_dim]
            
        Returns:
            Logits [batch_size, num_classes]
        """
        from peptide_atlas.models.hyperbolic.distance import hyperbolic_mlr
        
        # Ensure p is inside the ball
        p = project_to_poincare(self.p, max_norm=0.95)
        
        return hyperbolic_mlr(x, p, self.a, curvature=self.curvature)


class EuclideanToPoincareProjection(nn.Module):
    """
    Simple projection from Euclidean to Poincaré space.
    
    Wraps the projection operation for use in pipelines.
    """
    
    def __init__(self, max_norm: float = 0.99):
        super().__init__()
        self.max_norm = max_norm
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project to Poincaré ball."""
        return project_to_poincare(x, self.max_norm)
