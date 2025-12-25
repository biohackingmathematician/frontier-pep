"""
Data loading utilities for the Peptide Atlas.

REMINDER: This project is for research and education only.
No dosing, no protocols, no therapeutic recommendations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger

from peptide_atlas.data.schemas import KnowledgeGraph


def load_knowledge_graph(path: Path) -> KnowledgeGraph:
    """
    Load a knowledge graph from a JSON file.
    
    Args:
        path: Path to the JSON file
        
    Returns:
        Loaded KnowledgeGraph instance
    """
    logger.info(f"Loading knowledge graph from {path}")
    
    with open(path, "r") as f:
        data = json.load(f)
    
    kg = KnowledgeGraph.model_validate(data)
    logger.info(f"Loaded KG with {kg.node_count} nodes and {kg.edge_count} edges")
    
    return kg


def save_knowledge_graph(kg: KnowledgeGraph, path: Path) -> None:
    """
    Save a knowledge graph to a JSON file.
    
    Args:
        kg: KnowledgeGraph instance to save
        path: Path for the output JSON file
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(kg.model_dump(mode="json"), f, indent=2, default=str)
    
    logger.info(f"Saved knowledge graph to {path}")


def load_yaml_config(path: Path) -> dict:
    """
    Load a YAML configuration file.
    
    Args:
        path: Path to the YAML file
        
    Returns:
        Configuration dictionary
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def load_model_config(config_dir: Optional[Path] = None) -> dict:
    """Load model configuration."""
    if config_dir is None:
        from peptide_atlas.config import settings
        config_dir = settings.config_path
    
    return load_yaml_config(config_dir / "model_config.yaml")


def load_tda_config(config_dir: Optional[Path] = None) -> dict:
    """Load TDA configuration."""
    if config_dir is None:
        from peptide_atlas.config import settings
        config_dir = settings.config_path
    
    return load_yaml_config(config_dir / "tda_config.yaml")


def load_viz_config(config_dir: Optional[Path] = None) -> dict:
    """Load visualization configuration."""
    if config_dir is None:
        from peptide_atlas.config import settings
        config_dir = settings.config_path
    
    return load_yaml_config(config_dir / "viz_config.yaml")

