"""
Main PeptideAtlas class — unified interface to the knowledge resource.

REMINDER: This project is for research and education only.
No dosing, no protocols, no therapeutic recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
from loguru import logger

from peptide_atlas import print_disclaimer
from peptide_atlas.constants import EvidenceTier, PeptideClass
from peptide_atlas.data.schemas import KnowledgeGraph, PeptideNode
from peptide_atlas.data.loaders import load_knowledge_graph, load_embeddings


@dataclass
class SimilarityResult:
    """Result from similarity search."""
    peptide: PeptideNode
    distance: float
    similarity: float  # 1 - normalized_distance


@dataclass
class AtlasStats:
    """Statistics about the loaded atlas."""
    num_peptides: int
    num_targets: int
    num_pathways: int
    num_effect_domains: int
    num_risks: int
    num_edges: int
    embedding_dim: Optional[int]
    version: str


class PeptideAtlas:
    """
    Unified interface to the Frontier Peptide Atlas.
    
    Provides query access to the knowledge graph and embedding-based
    similarity search.
    
    Example:
        ```python
        from peptide_atlas.api import PeptideAtlas
        
        atlas = PeptideAtlas.load("data/processed/")
        
        # Find regenerative peptides
        regen = atlas.query_by_class("regenerative_repair")
        
        # Find similar to BPC-157
        similar = atlas.find_similar("BPC-157", k=5)
        ```
    """
    
    VERSION = "0.1.0"
    
    def __init__(
        self,
        kg: KnowledgeGraph,
        embeddings: Optional[np.ndarray] = None,
        peptide_names: Optional[List[str]] = None,
    ):
        """
        Initialize the atlas.
        
        Args:
            kg: Loaded KnowledgeGraph
            embeddings: Optional peptide embeddings [n_peptides, dim]
            peptide_names: Names corresponding to embedding rows
        """
        self.kg = kg
        self._embeddings = embeddings
        self._peptide_names = peptide_names
        
        # Build lookup indices
        self._name_to_peptide: dict[str, PeptideNode] = {}
        self._name_to_idx: dict[str, int] = {}
        
        for i, peptide in enumerate(kg.peptides):
            name_lower = peptide.canonical_name.lower()
            self._name_to_peptide[name_lower] = peptide
            self._name_to_idx[name_lower] = i
            
            # Also index synonyms
            for syn in peptide.synonyms:
                self._name_to_peptide[syn.lower()] = peptide
        
        logger.info(f"PeptideAtlas initialized: {self.num_peptides} peptides")
    
    @classmethod
    def load(
        cls,
        data_dir: Union[str, Path],
        kg_filename: str = "kg.json",
        embeddings_filename: str = "embeddings.npy",
        show_disclaimer: bool = True,
    ) -> "PeptideAtlas":
        """
        Load atlas from directory.
        
        Args:
            data_dir: Directory containing kg.json and embeddings
            kg_filename: Name of knowledge graph file
            embeddings_filename: Name of embeddings file
            show_disclaimer: Whether to print disclaimer
            
        Returns:
            Loaded PeptideAtlas instance
        """
        if show_disclaimer:
            print_disclaimer()
        
        data_dir = Path(data_dir)
        
        # Load knowledge graph
        kg_path = data_dir / kg_filename
        if not kg_path.exists():
            raise FileNotFoundError(f"Knowledge graph not found: {kg_path}")
        
        kg = load_knowledge_graph(kg_path)
        
        # Load embeddings if available
        embeddings = None
        peptide_names = None
        emb_path = data_dir / embeddings_filename
        
        if emb_path.exists():
            try:
                embeddings = load_embeddings(emb_path)
                peptide_names = [p.canonical_name for p in kg.peptides]
                logger.info(f"Loaded embeddings: {embeddings.shape}")
            except Exception as e:
                logger.warning(f"Could not load embeddings: {e}")
        
        return cls(kg, embeddings, peptide_names)
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def num_peptides(self) -> int:
        return len(self.kg.peptides)
    
    @property
    def num_targets(self) -> int:
        return len(self.kg.targets)
    
    @property
    def num_pathways(self) -> int:
        return len(self.kg.pathways)
    
    @property
    def num_effect_domains(self) -> int:
        return len(self.kg.effect_domains)
    
    @property
    def num_risks(self) -> int:
        return len(self.kg.risks)
    
    @property
    def has_embeddings(self) -> bool:
        return self._embeddings is not None
    
    @property
    def embedding_dim(self) -> Optional[int]:
        if self._embeddings is not None:
            return self._embeddings.shape[1]
        return None
    
    def stats(self) -> AtlasStats:
        """Get atlas statistics."""
        return AtlasStats(
            num_peptides=self.num_peptides,
            num_targets=self.num_targets,
            num_pathways=self.num_pathways,
            num_effect_domains=self.num_effect_domains,
            num_risks=self.num_risks,
            num_edges=self.kg.edge_count,
            embedding_dim=self.embedding_dim,
            version=self.VERSION,
        )
    
    # =========================================================================
    # Query Methods
    # =========================================================================
    
    def get_peptide(self, name: str) -> Optional[PeptideNode]:
        """
        Get peptide by name (case-insensitive).
        
        Args:
            name: Peptide name or synonym
            
        Returns:
            PeptideNode or None if not found
        """
        return self._name_to_peptide.get(name.lower())
    
    def list_peptides(self) -> List[str]:
        """List all peptide names."""
        return [p.canonical_name for p in self.kg.peptides]
    
    def list_classes(self) -> List[str]:
        """List all peptide classes."""
        return [c.value for c in PeptideClass]
    
    def list_evidence_tiers(self) -> List[str]:
        """List all evidence tiers."""
        return [t.value for t in EvidenceTier]
    
    def query_by_class(
        self,
        peptide_class: Union[str, PeptideClass],
    ) -> List[PeptideNode]:
        """
        Get all peptides of a given class.
        
        Args:
            peptide_class: Class name or enum
            
        Returns:
            List of matching peptides
        """
        if isinstance(peptide_class, str):
            peptide_class = PeptideClass(peptide_class)
        
        return [
            p for p in self.kg.peptides
            if p.peptide_class == peptide_class
        ]
    
    def query_by_evidence(
        self,
        min_tier: int = 4,
        max_tier: int = 1,
    ) -> List[PeptideNode]:
        """
        Get peptides within evidence tier range.
        
        Args:
            min_tier: Minimum tier (4 = preclinical)
            max_tier: Maximum tier (1 = approved)
            
        Returns:
            List of matching peptides
            
        Note:
            Lower tier number = better evidence.
            min_tier=4, max_tier=1 returns tiers 1, 2, 3, 4.
        """
        tier_order = [
            EvidenceTier.TIER_1_APPROVED,
            EvidenceTier.TIER_2_LATE_CLINICAL,
            EvidenceTier.TIER_3_EARLY_CLINICAL,
            EvidenceTier.TIER_4_PRECLINICAL,
            EvidenceTier.TIER_5_MECHANISTIC,
            EvidenceTier.TIER_6_ANECDOTAL,
            EvidenceTier.TIER_UNKNOWN,
        ]
        
        valid_tiers = tier_order[max_tier - 1 : min_tier]
        
        return [
            p for p in self.kg.peptides
            if p.evidence_tier in valid_tiers
        ]
    
    def query_by_pathway(self, pathway_name: str) -> List[PeptideNode]:
        """
        Get peptides that modulate a given pathway.
        
        Args:
            pathway_name: Name of pathway (partial match)
            
        Returns:
            List of peptides modulating that pathway
        """
        # Find pathway node
        pathway_name_lower = pathway_name.lower()
        matching_pathways = [
            p for p in self.kg.pathways
            if pathway_name_lower in p.name.lower()
        ]
        
        if not matching_pathways:
            return []
        
        pathway_ids = {str(p.id) for p in matching_pathways}
        
        # Find peptides with modulates edges to these pathways
        peptide_ids = set()
        for edge in self.kg.modulates_edges:
            if str(edge.target_id) in pathway_ids:
                peptide_ids.add(str(edge.source_id))
        
        return [
            p for p in self.kg.peptides
            if str(p.id) in peptide_ids
        ]
    
    def query_by_target(self, target_name: str) -> List[PeptideNode]:
        """
        Get peptides that bind a given target.
        
        Args:
            target_name: Name of target (partial match)
            
        Returns:
            List of peptides binding that target
        """
        target_name_lower = target_name.lower()
        matching_targets = [
            t for t in self.kg.targets
            if target_name_lower in t.name.lower()
        ]
        
        if not matching_targets:
            return []
        
        target_ids = {str(t.id) for t in matching_targets}
        
        peptide_ids = set()
        for edge in self.kg.binds_edges:
            if str(edge.target_id) in target_ids:
                peptide_ids.add(str(edge.source_id))
        
        return [
            p for p in self.kg.peptides
            if str(p.id) in peptide_ids
        ]
    
    # =========================================================================
    # Similarity Search
    # =========================================================================
    
    def find_similar(
        self,
        peptide_name: str,
        k: int = 5,
        exclude_self: bool = True,
    ) -> List[SimilarityResult]:
        """
        Find k most similar peptides in embedding space.
        
        Args:
            peptide_name: Name of query peptide
            k: Number of results
            exclude_self: Whether to exclude the query peptide
            
        Returns:
            List of SimilarityResult ordered by similarity
            
        Raises:
            ValueError: If embeddings not loaded or peptide not found
        """
        if not self.has_embeddings:
            raise ValueError("Embeddings not loaded. Train model first.")
        
        peptide = self.get_peptide(peptide_name)
        if peptide is None:
            raise ValueError(f"Peptide not found: {peptide_name}")
        
        query_idx = self._name_to_idx.get(peptide.canonical_name.lower())
        if query_idx is None:
            raise ValueError(f"Peptide not in embedding index: {peptide_name}")
        
        query_emb = self._embeddings[query_idx]
        
        # Compute distances (L2 for normalized embeddings = related to cosine)
        distances = np.linalg.norm(self._embeddings - query_emb, axis=1)
        
        # Sort
        sorted_indices = np.argsort(distances)
        
        results = []
        for idx in sorted_indices:
            if exclude_self and idx == query_idx:
                continue
            
            if len(results) >= k:
                break
            
            dist = distances[idx]
            # Similarity: 1 for identical, 0 for max distance
            max_dist = 2.0  # Max L2 distance for unit vectors
            sim = 1.0 - (dist / max_dist)
            
            results.append(SimilarityResult(
                peptide=self.kg.peptides[idx],
                distance=float(dist),
                similarity=float(sim),
            ))
        
        return results
    
    def find_bridges(
        self,
        class_a: str,
        class_b: str,
        k: int = 5,
    ) -> List[PeptideNode]:
        """
        Find peptides that bridge two mechanism classes.
        
        Uses embedding space to find peptides that are close to
        members of both classes.
        
        Args:
            class_a: First peptide class
            class_b: Second peptide class
            k: Number of results
            
        Returns:
            Peptides ordered by bridge potential
        """
        if not self.has_embeddings:
            raise ValueError("Embeddings not loaded")
        
        peptides_a = self.query_by_class(class_a)
        peptides_b = self.query_by_class(class_b)
        
        if not peptides_a or not peptides_b:
            return []
        
        # Get embeddings for each class
        indices_a = [self._name_to_idx[p.canonical_name.lower()] for p in peptides_a]
        indices_b = [self._name_to_idx[p.canonical_name.lower()] for p in peptides_b]
        
        emb_a = self._embeddings[indices_a].mean(axis=0)
        emb_b = self._embeddings[indices_b].mean(axis=0)
        
        # Find peptides close to both centroids
        dist_a = np.linalg.norm(self._embeddings - emb_a, axis=1)
        dist_b = np.linalg.norm(self._embeddings - emb_b, axis=1)
        
        # Bridge score: low distance to both
        bridge_score = dist_a + dist_b
        
        sorted_indices = np.argsort(bridge_score)[:k]
        
        return [self.kg.peptides[i] for i in sorted_indices]
    
    # =========================================================================
    # Export
    # =========================================================================
    
    def export_summary(self) -> dict[str, Any]:
        """Export atlas summary as dictionary."""
        return {
            "version": self.VERSION,
            "stats": {
                "peptides": self.num_peptides,
                "targets": self.num_targets,
                "pathways": self.num_pathways,
                "effect_domains": self.num_effect_domains,
                "risks": self.num_risks,
                "edges": self.kg.edge_count,
            },
            "peptides": [
                {
                    "name": p.canonical_name,
                    "class": p.peptide_class.value,
                    "evidence_tier": p.evidence_tier.value,
                    "regulatory_status": p.regulatory_status.value,
                }
                for p in self.kg.peptides
            ],
            "classes": {
                c.value: len(self.query_by_class(c))
                for c in PeptideClass
            },
        }

