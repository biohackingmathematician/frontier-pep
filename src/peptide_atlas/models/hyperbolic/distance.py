"""
Hyperbolic distance functions.

REMINDER: This project is for research and education only.
"""

from __future__ import annotations

import torch


def poincare_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    curvature: float = 1.0,
) -> torch.Tensor:
    """
    Compute Poincaré distance between points.
    
    The Poincaré distance is:
    d(x, y) = (1/sqrt(c)) * arcosh(1 + 2c * ||x-y||^2 / ((1-c||x||^2)(1-c||y||^2)))
    
    Args:
        x: First point(s) [*, dim]
        y: Second point(s) [*, dim]
        curvature: Curvature parameter
        
    Returns:
        Distances [*]
    """
    c = curvature
    
    # Compute norms
    x_sq = torch.sum(x * x, dim=-1)
    y_sq = torch.sum(y * y, dim=-1)
    
    # Compute squared distance
    diff_sq = torch.sum((x - y) ** 2, dim=-1)
    
    # Compute denominator terms
    denom = (1 - c * x_sq) * (1 - c * y_sq)
    denom = denom.clamp(min=1e-8)
    
    # Compute argument to arcosh
    arg = 1 + 2 * c * diff_sq / denom
    arg = arg.clamp(min=1 + 1e-7)  # arcosh(x) requires x >= 1
    
    # Compute distance
    sqrt_c = c ** 0.5
    dist = (1 / sqrt_c) * torch.acosh(arg)
    
    return dist


def poincare_distance_matrix(
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    curvature: float = 1.0,
) -> torch.Tensor:
    """
    Compute pairwise Poincaré distances.
    
    Args:
        x: First set of points [n, dim]
        y: Second set of points [m, dim] (default: same as x)
        curvature: Curvature parameter
        
    Returns:
        Distance matrix [n, m]
    """
    if y is None:
        y = x
    
    n = x.size(0)
    m = y.size(0)
    
    # Expand for broadcasting
    x_expanded = x.unsqueeze(1).expand(n, m, -1)  # [n, m, dim]
    y_expanded = y.unsqueeze(0).expand(n, m, -1)  # [n, m, dim]
    
    return poincare_distance(x_expanded, y_expanded, curvature)


def hyperbolic_centroid(
    points: torch.Tensor,
    weights: torch.Tensor | None = None,
    curvature: float = 1.0,
    num_iters: int = 10,
) -> torch.Tensor:
    """
    Compute weighted centroid in hyperbolic space.
    
    Uses iterative algorithm since there's no closed-form solution.
    
    Args:
        points: Points in Poincaré ball [n, dim]
        weights: Optional weights [n] (default: uniform)
        curvature: Curvature parameter
        num_iters: Number of optimization iterations
        
    Returns:
        Centroid point [dim]
    """
    if weights is None:
        weights = torch.ones(points.size(0), device=points.device)
    
    weights = weights / weights.sum()
    
    # Initialize at Euclidean mean (projected)
    centroid = (weights.unsqueeze(-1) * points).sum(dim=0)
    centroid = _project_to_ball(centroid, curvature)
    
    # Iterative refinement
    for _ in range(num_iters):
        # Compute distances
        dists = poincare_distance(
            centroid.unsqueeze(0).expand_as(points),
            points,
            curvature,
        )
        
        # Weighted mean in tangent space at centroid
        # (Simplified version - could use proper Riemannian gradient)
        diff = points - centroid
        gradient = (weights * dists).unsqueeze(-1) * diff / (dists.unsqueeze(-1) + 1e-8)
        gradient = gradient.sum(dim=0)
        
        # Update
        centroid = centroid + 0.1 * gradient
        centroid = _project_to_ball(centroid, curvature)
    
    return centroid


def _project_to_ball(
    x: torch.Tensor,
    curvature: float = 1.0,
    max_norm: float = 0.99,
) -> torch.Tensor:
    """Project point to Poincaré ball."""
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    cond = norm > max_norm
    return torch.where(cond, x * max_norm / norm, x)


def hyperbolic_mlr(
    x: torch.Tensor,
    p: torch.Tensor,
    a: torch.Tensor,
    curvature: float = 1.0,
) -> torch.Tensor:
    """
    Hyperbolic multinomial logistic regression.
    
    Computes decision values for classification in hyperbolic space.
    
    Args:
        x: Input points [batch, dim]
        p: Class hyperplane points [num_classes, dim]
        a: Class hyperplane normals [num_classes, dim]
        curvature: Curvature parameter
        
    Returns:
        Logits [batch, num_classes]
    """
    c = curvature
    sqrt_c = c ** 0.5
    
    # Compute conformal factor
    x_sq = torch.sum(x * x, dim=-1, keepdim=True)  # [batch, 1]
    lambda_x = 2 / (1 - c * x_sq + 1e-8)  # [batch, 1]
    
    # Compute (-p) ⊕ x using Möbius addition
    p_neg = -p  # [num_classes, dim]
    
    # Expand for broadcasting
    batch = x.size(0)
    num_classes = p.size(0)
    
    x_exp = x.unsqueeze(1).expand(batch, num_classes, -1)
    p_neg_exp = p_neg.unsqueeze(0).expand(batch, num_classes, -1)
    
    # Möbius addition (-p) ⊕ x
    x_sq_exp = torch.sum(x_exp * x_exp, dim=-1, keepdim=True)
    p_sq_exp = torch.sum(p_neg_exp * p_neg_exp, dim=-1, keepdim=True)
    xp = torch.sum(x_exp * p_neg_exp, dim=-1, keepdim=True)
    
    num = (1 + 2 * c * xp + c * p_sq_exp) * x_exp + (1 - c * x_sq_exp) * p_neg_exp
    denom = 1 + 2 * c * xp + c * c * x_sq_exp * p_sq_exp
    diff = num / (denom + 1e-8)  # [batch, num_classes, dim]
    
    # Compute inner product with normal vectors
    a_exp = a.unsqueeze(0).expand(batch, num_classes, -1)
    inner = torch.sum(diff * a_exp, dim=-1)  # [batch, num_classes]
    
    # Scale by conformal factor and norm of normal
    a_norm = torch.norm(a, p=2, dim=-1).unsqueeze(0)  # [1, num_classes]
    lambda_x_exp = lambda_x.expand(batch, num_classes)
    
    logits = (2 / sqrt_c) * a_norm * torch.asinh(
        sqrt_c * lambda_x_exp * inner / (a_norm + 1e-8)
    )
    
    return logits

