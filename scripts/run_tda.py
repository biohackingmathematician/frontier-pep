#!/usr/bin/env python3
"""
Run TDA analysis on peptide embeddings.

CRITICAL DISCLAIMER:
This script is for RESEARCH AND EDUCATIONAL PURPOSES ONLY.
No dosing, no protocols, no therapeutic recommendations.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from loguru import logger

from peptide_atlas import print_disclaimer
from peptide_atlas.data.loaders import load_knowledge_graph, load_yaml_config


def main():
    print_disclaimer()
    
    parser = argparse.ArgumentParser(
        description="Run TDA analysis on peptide embeddings"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path("configs/tda_config.yaml"),
        help="TDA configuration file",
    )
    parser.add_argument(
        "--embeddings", "-e",
        type=Path,
        default=Path("outputs/models/peptide_embeddings.npy"),
        help="Peptide embeddings file",
    )
    parser.add_argument(
        "--kg", "-k",
        type=Path,
        default=Path("data/processed/kg.json"),
        help="Knowledge graph for labels",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("outputs/tda"),
        help="Output directory",
    )
    args = parser.parse_args()
    
    print("\n=== Running TDA Analysis ===\n")
    
    # Load config
    config_dict = load_yaml_config(args.config)
    logger.info(f"Loaded TDA config from {args.config}")
    
    # Load embeddings
    if args.embeddings.exists():
        embeddings = np.load(args.embeddings)
        logger.info(f"Loaded embeddings: {embeddings.shape}")
    else:
        logger.warning(f"Embeddings not found at {args.embeddings}")
        logger.info("Generating random embeddings for demonstration")
        
        # Load KG to get peptide count
        kg = load_knowledge_graph(args.kg)
        n_peptides = len(kg.peptides)
        embeddings = np.random.randn(n_peptides, 64)
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Run Mapper
    logger.info("Running Mapper algorithm...")
    try:
        from peptide_atlas.tda.mapper import MapperPipeline, MapperConfig
        
        mapper_config = MapperConfig(
            n_cubes=config_dict.get("mapper", {}).get("cover", {}).get("n_cubes", 15),
            overlap_perc=config_dict.get("mapper", {}).get("cover", {}).get("overlap_perc", 0.5),
        )
        
        pipeline = MapperPipeline(mapper_config)
        result = pipeline.fit(embeddings)
        
        logger.info(f"Mapper complete: {result.num_nodes} nodes, {result.num_edges} edges")
        
        # Save Mapper HTML
        kg = load_knowledge_graph(args.kg)
        labels = [p.peptide_class.value for p in kg.peptides]
        
        html_path = args.output / "mapper_graph.html"
        pipeline.visualize_html(
            result,
            output_path=str(html_path),
            title="Peptide Atlas Mapper",
        )
        
        # Analyze
        from peptide_atlas.tda.analysis import analyze_mapper_graph
        
        label_array = np.array([list(set(labels)).index(l) for l in labels])
        analysis = analyze_mapper_graph(
            result,
            labels=label_array,
            class_names=list(set(labels)),
        )
        
        # Save analysis
        analysis_path = args.output / "mapper_analysis.json"
        with open(analysis_path, "w") as f:
            json.dump({
                "num_clusters": analysis.num_clusters,
                "num_bridges": analysis.num_bridges,
                "num_connected_components": analysis.num_connected_components,
                "avg_cluster_size": analysis.avg_cluster_size,
            }, f, indent=2)
        
        logger.info(f"Saved analysis to {analysis_path}")
        
    except ImportError as e:
        logger.warning(f"Mapper not available: {e}")
    
    # Run persistent homology
    logger.info("Computing persistent homology...")
    try:
        from peptide_atlas.tda.persistence import PersistentHomology, PersistenceConfig
        
        ph_config = PersistenceConfig(
            max_dimension=config_dict.get("persistence", {}).get("rips", {}).get("max_dimension", 2),
            max_edge_length=config_dict.get("persistence", {}).get("rips", {}).get("max_edge_length", 2.0),
        )
        
        ph = PersistentHomology(ph_config)
        diagram = ph.fit(embeddings)
        
        logger.info(f"H0: {diagram.num_features.get(0, 0)}, H1: {diagram.num_features.get(1, 0)}")
        
        # Plot diagram
        diagram_path = args.output / "persistence_diagram.png"
        ph.plot_diagram(diagram, output_path=str(diagram_path))
        
        # Save diagram data
        dgm_path = args.output / "persistence_diagrams.npz"
        np.savez(dgm_path, **{f"h{k}": v for k, v in diagram.dgms.items()})
        logger.info(f"Saved diagrams to {dgm_path}")
        
    except ImportError as e:
        logger.warning(f"Persistence computation not available: {e}")
    
    print("\n=== TDA Analysis Complete ===\n")
    return 0


if __name__ == "__main__":
    exit(main())

