#!/usr/bin/env python3
"""
Train the GNN encoder for the Peptide Atlas.

CRITICAL DISCLAIMER:
This script is for RESEARCH AND EDUCATIONAL PURPOSES ONLY.
No dosing, no protocols, no therapeutic recommendations.
"""

import argparse
from pathlib import Path

import torch
import numpy as np
from loguru import logger

from peptide_atlas import print_disclaimer
from peptide_atlas.data.loaders import load_knowledge_graph, load_yaml_config
from peptide_atlas.kg.export import export_for_pytorch_geometric
from peptide_atlas.models.gnn.config import GNNConfig, TrainingConfig
from peptide_atlas.models.gnn.encoder import HeterogeneousGNNEncoder


def main():
    print_disclaimer()
    
    parser = argparse.ArgumentParser(
        description="Train the GNN encoder"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path("configs/model_config.yaml"),
        help="Model configuration file",
    )
    parser.add_argument(
        "--kg", "-k",
        type=Path,
        default=Path("data/processed/kg.json"),
        help="Knowledge graph JSON file",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("outputs/models"),
        help="Output directory",
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    args = parser.parse_args()
    
    print("\n=== Training GNN Encoder ===\n")
    
    # Load config
    config_dict = load_yaml_config(args.config)
    gnn_config = GNNConfig(**config_dict.get("model", {}))
    train_config = TrainingConfig(**config_dict.get("training", {}))
    
    # Load knowledge graph
    logger.info(f"Loading knowledge graph from {args.kg}")
    kg = load_knowledge_graph(args.kg)
    
    # Export for PyTorch Geometric
    logger.info("Preparing graph data for PyTorch")
    graph_data = export_for_pytorch_geometric(kg)
    
    # Create model
    num_node_types = len(set(graph_data["node_types"]))
    num_edge_types = len(set(graph_data["edge_types"]))
    
    logger.info(f"Creating model: {num_node_types} node types, {num_edge_types} edge types")
    
    model = HeterogeneousGNNEncoder(
        config=gnn_config,
        num_node_types=num_node_types,
        num_edge_types=num_edge_types,
    )
    
    # Prepare tensors
    node_types = torch.tensor([
        ["peptide", "target", "pathway", "effect_domain", "risk"].index(nt)
        if nt in ["peptide", "target", "pathway", "effect_domain", "risk"]
        else 0
        for nt in graph_data["node_types"]
    ], dtype=torch.long)
    
    edge_index = torch.tensor(graph_data["edge_index"], dtype=torch.long)
    
    edge_type_map = {et: i for i, et in enumerate(sorted(set(graph_data["edge_types"])))}
    edge_type = torch.tensor([
        edge_type_map[et] for et in graph_data["edge_types"]
    ], dtype=torch.long)
    
    logger.info(f"Graph: {len(node_types)} nodes, {edge_index.shape[1]} edges")
    
    # Simple training loop (self-supervised)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    
    logger.info(f"Starting training for {args.epochs} epochs")
    
    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        
        # Forward pass
        embeddings = model(node_types, edge_index, edge_type)
        
        # Simple reconstruction loss (positive pairs should be close)
        # Use edge endpoints as positive pairs
        if edge_index.shape[1] > 0:
            src_emb = embeddings[edge_index[0]]
            dst_emb = embeddings[edge_index[1]]
            
            # Cosine similarity loss
            similarity = torch.nn.functional.cosine_similarity(src_emb, dst_emb)
            loss = -similarity.mean()  # Maximize similarity
        else:
            loss = torch.tensor(0.0)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}/{args.epochs}, Loss: {loss.item():.4f}")
    
    # Save model
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "gnn_encoder.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Save embeddings
    model.eval()
    with torch.no_grad():
        final_embeddings = model(node_types, edge_index, edge_type)
    
    embeddings_path = args.output / "embeddings.npy"
    np.save(embeddings_path, final_embeddings.numpy())
    logger.info(f"Saved embeddings to {embeddings_path}")
    
    # Get peptide embeddings specifically
    peptide_mask = torch.tensor([nt == "peptide" for nt in graph_data["node_types"]])
    peptide_embeddings = final_embeddings[peptide_mask].numpy()
    
    peptide_embeddings_path = args.output / "peptide_embeddings.npy"
    np.save(peptide_embeddings_path, peptide_embeddings)
    logger.info(f"Saved peptide embeddings to {peptide_embeddings_path}")
    
    print("\n=== Training Complete ===\n")
    return 0


if __name__ == "__main__":
    exit(main())

