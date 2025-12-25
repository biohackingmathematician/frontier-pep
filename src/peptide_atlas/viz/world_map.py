"""
World map visualization for the Peptide Atlas.

Creates an interactive 2D projection of the peptide embedding space.

REMINDER: This project is for research and education only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
from loguru import logger

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    logger.warning("Plotly not installed. Visualization will be limited.")

from peptide_atlas.viz.style import (
    DISCLAIMER_TEXT,
    format_hover_text,
    get_class_color,
    get_color_palette,
    get_evidence_color,
    get_node_size,
    get_theme,
)
from peptide_atlas.constants import EvidenceTier, PeptideClass


def create_world_map(
    embeddings_2d: np.ndarray,
    names: list[str],
    peptide_classes: list[str],
    evidence_tiers: list[str],
    descriptions: Optional[list[str]] = None,
    edges: Optional[list[tuple[int, int]]] = None,
    edge_weights: Optional[list[float]] = None,
    title: str = "Frontier Peptide Atlas",
    color_by: str = "peptide_class",
    size_by: str = "evidence",
    theme: str = "dark",
    output_path: Optional[str] = None,
    show_legend: bool = True,
    show_disclaimer: bool = True,
) -> Any:
    """
    Create an interactive world map visualization.
    
    Args:
        embeddings_2d: 2D coordinates [n_peptides, 2]
        names: Peptide names
        peptide_classes: Peptide class labels
        evidence_tiers: Evidence tier labels
        descriptions: Optional descriptions
        edges: Optional edge list as (source_idx, target_idx)
        edge_weights: Optional edge weights
        title: Plot title
        color_by: What to color nodes by
        size_by: What to size nodes by
        theme: Color theme
        output_path: Optional path to save HTML
        show_legend: Whether to show legend
        show_disclaimer: Whether to show disclaimer overlay
        
    Returns:
        Plotly figure
    """
    if not HAS_PLOTLY:
        raise ImportError("Plotly is required for visualization")
    
    theme_colors = get_theme(theme)
    color_palette = get_color_palette(color_by)
    
    # Prepare data
    n_points = len(names)
    x = embeddings_2d[:, 0]
    y = embeddings_2d[:, 1]
    
    # Compute sizes
    if size_by == "evidence":
        # Map evidence tiers to confidence scores
        tier_scores = []
        for tier in evidence_tiers:
            try:
                et = EvidenceTier(tier)
                tier_scores.append(et.confidence_score)
            except ValueError:
                tier_scores.append(0.5)
        sizes = [get_node_size(s, "medium", 0, 1) for s in tier_scores]
    else:
        sizes = [20] * n_points
    
    # Compute colors
    if color_by == "peptide_class":
        colors = [color_palette.get(pc, "#7f7f7f") for pc in peptide_classes]
    elif color_by == "evidence_tier":
        colors = [color_palette.get(et, "#7f7f7f") for et in evidence_tiers]
    else:
        colors = ["#4a9eff"] * n_points
    
    # Prepare hover text
    if descriptions is None:
        descriptions = [""] * n_points
    
    hover_texts = [
        format_hover_text(name, pc, et, desc)
        for name, pc, et, desc in zip(names, peptide_classes, evidence_tiers, descriptions)
    ]
    
    # Create figure
    fig = go.Figure()
    
    # Add edges if provided
    if edges is not None:
        edge_x = []
        edge_y = []
        
        for src, dst in edges:
            edge_x.extend([x[src], x[dst], None])
            edge_y.extend([y[src], y[dst], None])
        
        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(
                width=0.5,
                color=theme_colors["grid"],
            ),
            hoverinfo="skip",
            showlegend=False,
        ))
    
    # Group by class for legend
    if show_legend and color_by == "peptide_class":
        unique_classes = sorted(set(peptide_classes))
        
        for pc in unique_classes:
            mask = [i for i, c in enumerate(peptide_classes) if c == pc]
            
            fig.add_trace(go.Scatter(
                x=[x[i] for i in mask],
                y=[y[i] for i in mask],
                mode="markers+text",
                marker=dict(
                    size=[sizes[i] for i in mask],
                    color=color_palette.get(pc, "#7f7f7f"),
                    line=dict(width=1, color="white"),
                    opacity=0.85,
                ),
                text=[names[i] for i in mask],
                textposition="top center",
                textfont=dict(size=8, color=theme_colors["text_secondary"]),
                hovertext=[hover_texts[i] for i in mask],
                hoverinfo="text",
                name=pc.replace("_", " ").title(),
            ))
    else:
        # Single trace
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="markers+text",
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(width=1, color="white"),
                opacity=0.85,
            ),
            text=names,
            textposition="top center",
            textfont=dict(size=8, color=theme_colors["text_secondary"]),
            hovertext=hover_texts,
            hoverinfo="text",
            showlegend=False,
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup style='color:#888888;font-style:italic'>Research Use Only - Not Medical Advice</sup>",
            x=0.5,
            font=dict(size=20, color=theme_colors["text"]),
        ),
        paper_bgcolor=theme_colors["paper_bg"],
        plot_bgcolor=theme_colors["background"],
        font=dict(color=theme_colors["text"]),
        xaxis=dict(
            showgrid=True,
            gridcolor=theme_colors["grid"],
            zeroline=False,
            showticklabels=False,
            title="",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=theme_colors["grid"],
            zeroline=False,
            showticklabels=False,
            title="",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor=theme_colors["grid"],
            borderwidth=1,
        ),
        hovermode="closest",
        margin=dict(l=50, r=50, t=80, b=50),
    )
    
    # Add disclaimer annotation if requested
    if show_disclaimer:
        fig.add_annotation(
            text="RESEARCH USE ONLY - No medical advice, no dosing, no recommendations",
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.05,
            showarrow=False,
            font=dict(size=10, color="#888888", family="serif"),
            align="center",
        )
    
    # Save if path provided
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add more prominent disclaimer in HTML
        fig.write_html(
            str(path),
            include_plotlyjs="cdn",
            full_html=True,
        )
        logger.info(f"Saved world map to {output_path}")
    
    return fig


def create_mapper_visualization(
    mapper_result: Any,
    node_colors: Optional[dict[str, str]] = None,
    title: str = "Mapper Graph",
    theme: str = "dark",
    output_path: Optional[str] = None,
) -> Any:
    """
    Create visualization of a Mapper graph.
    
    Args:
        mapper_result: Result from MapperPipeline
        node_colors: Optional color for each node
        title: Plot title
        theme: Color theme
        output_path: Optional path to save HTML
        
    Returns:
        Plotly figure
    """
    if not HAS_PLOTLY:
        raise ImportError("Plotly is required for visualization")
    
    import networkx as nx
    
    theme_colors = get_theme(theme)
    
    # Build NetworkX graph for layout
    G = nx.Graph()
    for node_id in mapper_result.node_ids:
        G.add_node(node_id, size=len(mapper_result.node_members[node_id]))
    
    for src, dst in mapper_result.edges:
        G.add_edge(src, dst)
    
    # Compute layout
    pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
    
    # Create figure
    fig = go.Figure()
    
    # Add edges
    edge_x = []
    edge_y = []
    for src, dst in mapper_result.edges:
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    fig.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1, color=theme_colors["grid"]),
        hoverinfo="skip",
        showlegend=False,
    ))
    
    # Add nodes
    node_x = [pos[n][0] for n in mapper_result.node_ids]
    node_y = [pos[n][1] for n in mapper_result.node_ids]
    node_sizes = [len(mapper_result.node_members[n]) * 3 + 10 for n in mapper_result.node_ids]
    
    if node_colors:
        colors = [node_colors.get(n, theme_colors["accent"]) for n in mapper_result.node_ids]
    else:
        colors = [theme_colors["accent"]] * len(mapper_result.node_ids)
    
    hover_texts = [
        f"Cluster: {n}<br>Members: {len(mapper_result.node_members[n])}"
        for n in mapper_result.node_ids
    ]
    
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker=dict(
            size=node_sizes,
            color=colors,
            line=dict(width=1, color="white"),
        ),
        hovertext=hover_texts,
        hoverinfo="text",
        showlegend=False,
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup style='color:#888888;font-style:italic'>Research Use Only</sup>",
            x=0.5,
            font=dict(size=18, color=theme_colors["text"]),
        ),
        paper_bgcolor=theme_colors["paper_bg"],
        plot_bgcolor=theme_colors["background"],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=60, b=40),
    )
    
    if output_path:
        fig.write_html(str(output_path), include_plotlyjs="cdn")
        logger.info(f"Saved Mapper visualization to {output_path}")
    
    return fig

