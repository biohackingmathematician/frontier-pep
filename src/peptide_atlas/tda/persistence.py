"""
Persistent homology for the Peptide Atlas.

Computes topological features (Betti numbers, persistence diagrams)
that reveal the intrinsic structure of the embedding space.

REMINDER: This project is for research and education only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger

try:
    import gudhi
    HAS_GUDHI = True
except ImportError:
    HAS_GUDHI = False
    logger.warning("GUDHI not installed. Some persistence features unavailable.")

try:
    from ripser import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False


@dataclass
class PersistenceConfig:
    """Configuration for persistent homology computation."""
    
    max_dimension: int = 2
    max_edge_length: float = 2.0
    min_persistence: float = 0.1
    backend: str = "ripser"  # or "gudhi"


@dataclass
class PersistenceDiagram:
    """Result from persistent homology computation."""
    
    # Persistence pairs by dimension
    # Each is array of [birth, death] pairs
    dgms: dict[int, np.ndarray]
    
    # Betti numbers at different scales
    betti_numbers: Optional[dict[float, list[int]]] = None
    
    # Summary statistics
    total_persistence: Optional[dict[int, float]] = None
    num_features: Optional[dict[int, int]] = None
    
    @property
    def h0(self) -> np.ndarray:
        """Get H0 (connected components) diagram."""
        return self.dgms.get(0, np.array([]))
    
    @property
    def h1(self) -> np.ndarray:
        """Get H1 (loops) diagram."""
        return self.dgms.get(1, np.array([]))
    
    @property
    def h2(self) -> np.ndarray:
        """Get H2 (voids) diagram."""
        return self.dgms.get(2, np.array([]))


class PersistentHomology:
    """
    Persistent homology computation for peptide embeddings.
    
    Reveals topological features at multiple scales.
    """
    
    def __init__(self, config: Optional[PersistenceConfig] = None):
        """
        Initialize persistence computation.
        
        Args:
            config: Configuration for computation
        """
        self.config = config or PersistenceConfig()
        
        # Check backends
        if self.config.backend == "ripser" and not HAS_RIPSER:
            if HAS_GUDHI:
                logger.warning("Ripser not available, falling back to GUDHI")
                self.config.backend = "gudhi"
            else:
                raise ImportError(
                    "Neither ripser nor gudhi is installed. "
                    "Install with: pip install ripser gudhi"
                )
    
    def fit(self, X: np.ndarray) -> PersistenceDiagram:
        """
        Compute persistent homology.
        
        Args:
            X: Data matrix [n_samples, n_features] or distance matrix
            
        Returns:
            PersistenceDiagram with results
        """
        logger.info(f"Computing persistence with {self.config.backend}")
        
        if self.config.backend == "ripser":
            return self._compute_ripser(X)
        else:
            return self._compute_gudhi(X)
    
    def _compute_ripser(self, X: np.ndarray) -> PersistenceDiagram:
        """Compute using ripser."""
        result = ripser(
            X,
            maxdim=self.config.max_dimension,
            thresh=self.config.max_edge_length,
        )
        
        dgms = {i: dgm for i, dgm in enumerate(result["dgms"])}
        
        return self._create_diagram(dgms)
    
    def _compute_gudhi(self, X: np.ndarray) -> PersistenceDiagram:
        """Compute using GUDHI."""
        # Build Rips complex
        rips = gudhi.RipsComplex(
            points=X,
            max_edge_length=self.config.max_edge_length,
        )
        
        simplex_tree = rips.create_simplex_tree(
            max_dimension=self.config.max_dimension + 1
        )
        
        # Compute persistence
        simplex_tree.compute_persistence()
        
        # Extract diagrams
        dgms = {}
        for dim in range(self.config.max_dimension + 1):
            pairs = simplex_tree.persistence_intervals_in_dimension(dim)
            dgms[dim] = np.array(pairs) if len(pairs) > 0 else np.array([]).reshape(0, 2)
        
        return self._create_diagram(dgms)
    
    def _create_diagram(self, dgms: dict[int, np.ndarray]) -> PersistenceDiagram:
        """Create PersistenceDiagram with statistics."""
        # Filter by minimum persistence
        filtered_dgms = {}
        for dim, dgm in dgms.items():
            if len(dgm) > 0:
                persistence = dgm[:, 1] - dgm[:, 0]
                # Handle infinite values
                persistence = np.where(np.isinf(persistence), 0, persistence)
                mask = persistence >= self.config.min_persistence
                filtered_dgms[dim] = dgm[mask]
            else:
                filtered_dgms[dim] = dgm
        
        # Compute statistics
        total_persistence = {}
        num_features = {}
        
        for dim, dgm in filtered_dgms.items():
            if len(dgm) > 0:
                pers = dgm[:, 1] - dgm[:, 0]
                # Handle infinite values
                pers = np.where(np.isinf(pers), self.config.max_edge_length, pers)
                total_persistence[dim] = float(np.sum(pers))
                num_features[dim] = len(dgm)
            else:
                total_persistence[dim] = 0.0
                num_features[dim] = 0
        
        logger.info(
            f"Persistence computed: "
            f"H0={num_features.get(0, 0)}, "
            f"H1={num_features.get(1, 0)}, "
            f"H2={num_features.get(2, 0)} features"
        )
        
        return PersistenceDiagram(
            dgms=filtered_dgms,
            total_persistence=total_persistence,
            num_features=num_features,
        )
    
    def plot_diagram(
        self,
        diagram: PersistenceDiagram,
        output_path: Optional[str] = None,
        title: str = "Persistence Diagram",
    ):
        """
        Plot persistence diagram.
        
        Args:
            diagram: Computed persistence diagram
            output_path: Optional path to save figure
            title: Plot title
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
        labels = {0: "H₀ (components)", 1: "H₁ (loops)", 2: "H₂ (voids)"}
        
        max_val = 0
        
        for dim in range(self.config.max_dimension + 1):
            dgm = diagram.dgms.get(dim, np.array([]))
            if len(dgm) > 0:
                # Handle infinite deaths
                deaths = np.where(
                    np.isinf(dgm[:, 1]),
                    self.config.max_edge_length * 1.1,
                    dgm[:, 1]
                )
                ax.scatter(
                    dgm[:, 0],
                    deaths,
                    c=colors.get(dim, "#7f7f7f"),
                    label=labels.get(dim, f"H{dim}"),
                    alpha=0.7,
                    s=50,
                )
                max_val = max(max_val, np.max(dgm[:, 0]), np.max(deaths))
        
        # Diagonal line
        ax.plot([0, max_val * 1.1], [0, max_val * 1.1], "k--", alpha=0.3)
        
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        ax.set_title(f"{title}\n(Research Use Only)")
        ax.legend()
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved persistence diagram to {output_path}")
        
        return fig


def compute_persistence_from_embeddings(
    embeddings: np.ndarray,
    config: Optional[PersistenceConfig] = None,
) -> PersistenceDiagram:
    """
    Convenience function to compute persistence from embeddings.
    
    Args:
        embeddings: Peptide embeddings [n_peptides, dim]
        config: Persistence configuration
        
    Returns:
        PersistenceDiagram
    """
    ph = PersistentHomology(config)
    return ph.fit(embeddings)

