#!/usr/bin/env python3
"""
Generate the complete Peptide Atlas visualization.

CRITICAL DISCLAIMER:
This script is for RESEARCH AND EDUCATIONAL PURPOSES ONLY.
No dosing, no protocols, no therapeutic recommendations.
"""

import argparse
from pathlib import Path

import numpy as np
from loguru import logger

from peptide_atlas import print_disclaimer
from peptide_atlas.data.loaders import load_knowledge_graph, load_yaml_config
from peptide_atlas.viz.world_map import create_world_map


def main():
    print_disclaimer()
    
    parser = argparse.ArgumentParser(
        description="Generate the Peptide Atlas visualization"
    )
    parser.add_argument(
        "--kg", "-k",
        type=Path,
        default=Path("data/processed/kg.json"),
        help="Knowledge graph JSON file",
    )
    parser.add_argument(
        "--embeddings", "-e",
        type=Path,
        default=None,
        help="2D embeddings file (.npy)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("outputs/world_map.html"),
        help="Output HTML file",
    )
    parser.add_argument(
        "--theme",
        choices=["dark", "light"],
        default="dark",
        help="Color theme",
    )
    args = parser.parse_args()
    
    print("\n=== Generating Peptide Atlas ===\n")
    
    # Load knowledge graph
    logger.info(f"Loading knowledge graph from {args.kg}")
    kg = load_knowledge_graph(args.kg)
    
    # Load or generate embeddings
    if args.embeddings and args.embeddings.exists():
        embeddings_2d = np.load(args.embeddings)
        logger.info(f"Loaded embeddings: {embeddings_2d.shape}")
    else:
        logger.info("Generating UMAP projection from peptide features")
        
        # Create simple feature vectors from peptide properties
        from peptide_atlas.constants import PeptideClass, EvidenceTier
        
        n_peptides = len(kg.peptides)
        
        # Use class and evidence as features
        class_ids = [list(PeptideClass).index(p.peptide_class) for p in kg.peptides]
        tier_scores = [p.evidence_tier.confidence_score for p in kg.peptides]
        
        features = np.column_stack([class_ids, tier_scores])
        
        # Add some noise and project to 2D
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=5)
            embeddings_2d = reducer.fit_transform(features + np.random.randn(*features.shape) * 0.1)
        except ImportError:
            from sklearn.decomposition import PCA
            embeddings_2d = PCA(n_components=2).fit_transform(features)
            embeddings_2d += np.random.randn(*embeddings_2d.shape) * 0.3
    
    # Extract peptide data
    names = [p.canonical_name for p in kg.peptides]
    classes = [p.peptide_class.value for p in kg.peptides]
    tiers = [p.evidence_tier.value for p in kg.peptides]
    descriptions = [p.description or "" for p in kg.peptides]
    
    # Create visualization
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    fig = create_world_map(
        embeddings_2d=embeddings_2d,
        names=names,
        peptide_classes=classes,
        evidence_tiers=tiers,
        descriptions=descriptions,
        title="Frontier Peptide Atlas",
        color_by="peptide_class",
        size_by="evidence",
        theme=args.theme,
        output_path=str(args.output),
        show_legend=True,
        show_disclaimer=True,
    )
    
    logger.info(f"Saved visualization to {args.output}")
    
    # Also save embeddings
    embeddings_out = args.output.parent / "embeddings_2d.npy"
    np.save(embeddings_out, embeddings_2d)
    logger.info(f"Saved 2D embeddings to {embeddings_out}")
    
    print("\n=== Atlas Generation Complete ===\n")
    print(f"Open {args.output} in a browser to view the atlas.")
    print()
    
    return 0


if __name__ == "__main__":
    exit(main())

