"""
Interactive visualization components.

REMINDER: This project is for research and education only.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from loguru import logger

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from peptide_atlas.viz.style import get_theme, DISCLAIMER_TEXT


def create_embedding_explorer(
    embeddings_2d: np.ndarray,
    names: list[str],
    metadata: dict[str, list[Any]],
    title: str = "Embedding Explorer",
    theme: str = "dark",
) -> Any:
    """
    Create an interactive embedding explorer with filtering.
    
    Args:
        embeddings_2d: 2D embeddings
        names: Point names
        metadata: Dictionary of metadata columns
        title: Figure title
        theme: Color theme
        
    Returns:
        Plotly figure with interactive controls
    """
    if not HAS_PLOTLY:
        raise ImportError("Plotly required")
    
    theme_colors = get_theme(theme)
    
    fig = go.Figure()
    
    # Add main scatter
    fig.add_trace(go.Scatter(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        mode="markers+text",
        marker=dict(size=12, color=theme_colors["accent"]),
        text=names,
        textposition="top center",
        textfont=dict(size=8),
        hovertext=[
            "<br>".join([f"{k}: {v[i]}" for k, v in metadata.items()])
            for i in range(len(names))
        ],
        hoverinfo="text+name",
        name="Peptides",
    ))
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup style='font-style:italic'>Research Use Only</sup>",
            x=0.5,
        ),
        paper_bgcolor=theme_colors["paper_bg"],
        plot_bgcolor=theme_colors["background"],
        font=dict(color=theme_colors["text"]),
        xaxis=dict(showgrid=True, gridcolor=theme_colors["grid"]),
        yaxis=dict(showgrid=True, gridcolor=theme_colors["grid"]),
    )
    
    return fig


def create_comparison_view(
    peptide_a: dict[str, Any],
    peptide_b: dict[str, Any],
    shared_targets: list[str],
    shared_pathways: list[str],
    theme: str = "dark",
) -> Any:
    """
    Create a comparison view between two peptides.
    
    Shows shared mechanisms while avoiding any dosing comparisons.
    """
    if not HAS_PLOTLY:
        raise ImportError("Plotly required")
    
    theme_colors = get_theme(theme)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[peptide_a.get("name", "Peptide A"), peptide_b.get("name", "Peptide B")],
        horizontal_spacing=0.1,
    )
    
    # Peptide A properties
    props_a = [
        f"Class: {peptide_a.get('class', 'Unknown')}",
        f"Evidence: {peptide_a.get('evidence_tier', 'Unknown')}",
        f"Targets: {len(peptide_a.get('targets', []))}",
        f"Pathways: {len(peptide_a.get('pathways', []))}",
    ]
    
    fig.add_trace(go.Table(
        header=dict(values=["Property"], fill_color=theme_colors["paper_bg"]),
        cells=dict(values=[props_a], fill_color=theme_colors["background"]),
    ), row=1, col=1)
    
    # Peptide B properties
    props_b = [
        f"Class: {peptide_b.get('class', 'Unknown')}",
        f"Evidence: {peptide_b.get('evidence_tier', 'Unknown')}",
        f"Targets: {len(peptide_b.get('targets', []))}",
        f"Pathways: {len(peptide_b.get('pathways', []))}",
    ]
    
    fig.add_trace(go.Table(
        header=dict(values=["Property"], fill_color=theme_colors["paper_bg"]),
        cells=dict(values=[props_b], fill_color=theme_colors["background"]),
    ), row=1, col=2)
    
    fig.update_layout(
        title=dict(
            text="Peptide Comparison (Research Use Only)",
            x=0.5,
        ),
        paper_bgcolor=theme_colors["paper_bg"],
    )
    
    return fig


def add_disclaimer_overlay(fig: Any) -> Any:
    """Add a disclaimer overlay to a Plotly figure."""
    fig.add_annotation(
        text=DISCLAIMER_TEXT.replace("\n", "<br>"),
        xref="paper",
        yref="paper",
        x=0.5,
        y=1.05,
        showarrow=False,
        font=dict(size=9, color="#ff6b6b"),
        align="center",
        bgcolor="rgba(0,0,0,0.7)",
        bordercolor="#ff6b6b",
        borderwidth=1,
        borderpad=5,
    )
    return fig

