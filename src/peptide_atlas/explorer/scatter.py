"""
Scatter plot visualization for peptide embeddings.

NOTE: This is an exploration tool, not the core deliverable.

REMINDER: This project is for research and education only.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from peptide_atlas.constants import PEPTIDE_CLASS_COLORS, PeptideClass


EXPLORER_DISCLAIMER = (
    "RESEARCH USE ONLY - Not medical advice. "
    "No dosing or protocol recommendations."
)


def create_embedding_scatter(
    embeddings_2d: np.ndarray,
    names: List[str],
    peptide_classes: List[str],
    evidence_tiers: List[str],
    descriptions: Optional[List[str]] = None,
    title: str = "Peptide Atlas Explorer",
    color_by: str = "peptide_class",
    output_path: Optional[Union[str, Path]] = None,
    width: int = 1000,
    height: int = 700,
) -> "go.Figure":
    """
    Create interactive scatter plot of peptide embeddings.
    
    This is an EXPLORATION TOOL for browsing the atlas.
    The core value is in the knowledge graph and embeddings.
    
    Args:
        embeddings_2d: 2D coordinates [n_peptides, 2]
        names: Peptide names
        peptide_classes: Class for each peptide
        evidence_tiers: Evidence tier for each peptide
        descriptions: Optional descriptions
        title: Plot title
        color_by: "peptide_class" or "evidence_tier"
        output_path: Optional path to save HTML
        width: Figure width
        height: Figure height
        
    Returns:
        Plotly Figure
    """
    if not HAS_PLOTLY:
        raise ImportError("Plotly required: pip install plotly")
    
    n_points = len(names)
    
    # Build hover text
    hover_texts = []
    for i in range(n_points):
        text = f"<b>{names[i]}</b><br>"
        text += f"Class: {peptide_classes[i].replace('_', ' ').title()}<br>"
        text += f"Evidence: {evidence_tiers[i].replace('_', ' ').title()}"
        if descriptions and descriptions[i]:
            desc = descriptions[i][:100] + "..." if len(descriptions[i]) > 100 else descriptions[i]
            text += f"<br><br>{desc}"
        hover_texts.append(text)
    
    fig = go.Figure()
    
    # Add points by class for legend
    unique_classes = sorted(set(peptide_classes))
    
    for pclass in unique_classes:
        mask = [c == pclass for c in peptide_classes]
        indices = [i for i, m in enumerate(mask) if m]
        
        color = PEPTIDE_CLASS_COLORS.get(PeptideClass(pclass), "#888888")
        
        fig.add_trace(go.Scatter(
            x=embeddings_2d[indices, 0],
            y=embeddings_2d[indices, 1],
            mode="markers+text",
            marker=dict(size=10, color=color, line=dict(width=1, color="white")),
            text=[names[i] for i in indices],
            textposition="top center",
            textfont=dict(size=8),
            hovertext=[hover_texts[i] for i in indices],
            hoverinfo="text",
            name=pclass.replace("_", " ").title(),
        ))
    
    fig.update_layout(
        title=f"{title}<br><sub>{EXPLORER_DISCLAIMER}</sub>",
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        width=width,
        height=height,
        hovermode="closest",
    )
    
    if output_path:
        fig.write_html(str(output_path))
    
    return fig

