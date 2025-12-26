"""
Knowledge graph visualization for the Peptide Atlas.

NOTE: This is an exploration tool, not the core deliverable.

REMINDER: This project is for research and education only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from peptide_atlas.data.schemas import KnowledgeGraph
from peptide_atlas.constants import PEPTIDE_CLASS_COLORS, PeptideClass


EXPLORER_DISCLAIMER = (
    "RESEARCH USE ONLY - Not medical advice. "
    "No dosing or protocol recommendations."
)


def create_kg_visualization(
    kg: KnowledgeGraph,
    layout: str = "spring",
    title: str = "Knowledge Graph Explorer",
    output_path: Optional[Union[str, Path]] = None,
    width: int = 1200,
    height: int = 800,
) -> "go.Figure":
    """
    Create interactive visualization of the knowledge graph.
    
    This is an EXPLORATION TOOL for browsing the atlas.
    The core value is in the knowledge graph structure, not this visualization.
    
    Args:
        kg: KnowledgeGraph to visualize
        layout: NetworkX layout algorithm ("spring", "kamada_kawai", "circular")
        title: Plot title
        output_path: Optional path to save HTML
        width: Figure width
        height: Figure height
        
    Returns:
        Plotly Figure
    """
    if not HAS_PLOTLY:
        raise ImportError("Plotly required: pip install plotly")
    if not HAS_NETWORKX:
        raise ImportError("NetworkX required: pip install networkx")
    
    # Build NetworkX graph
    G = nx.Graph()
    
    # Add peptide nodes
    for p in kg.peptides:
        G.add_node(
            str(p.id),
            label=p.canonical_name,
            node_type="peptide",
            peptide_class=p.peptide_class.value,
        )
    
    # Add target nodes
    for t in kg.targets:
        G.add_node(
            str(t.id),
            label=t.name,
            node_type="target",
        )
    
    # Add pathway nodes
    for p in kg.pathways:
        G.add_node(
            str(p.id),
            label=p.name,
            node_type="pathway",
        )
    
    # Add edges
    for e in kg.binds_edges:
        G.add_edge(str(e.source_id), str(e.target_id), edge_type="binds")
    
    for e in kg.modulates_edges:
        G.add_edge(str(e.source_id), str(e.target_id), edge_type="modulates")
    
    # Compute layout
    if layout == "spring":
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Create figure
    fig = go.Figure()
    
    # Add edges
    edge_x = []
    edge_y = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    fig.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.5, color="#888"),
        hoverinfo="skip",
        showlegend=False,
    ))
    
    # Add nodes by type
    node_types = {
        "peptide": {"color": "#4a9eff", "size": 15},
        "target": {"color": "#ff6b6b", "size": 10},
        "pathway": {"color": "#51cf66", "size": 10},
    }
    
    for node_type, style in node_types.items():
        nodes = [n for n in G.nodes() if G.nodes[n].get("node_type") == node_type]
        
        if not nodes:
            continue
        
        x = [pos[n][0] for n in nodes]
        y = [pos[n][1] for n in nodes]
        labels = [G.nodes[n].get("label", n) for n in nodes]
        
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="markers+text",
            marker=dict(
                size=style["size"],
                color=style["color"],
                line=dict(width=1, color="white"),
            ),
            text=labels,
            textposition="top center",
            textfont=dict(size=8),
            hovertext=labels,
            hoverinfo="text",
            name=node_type.title(),
        ))
    
    fig.update_layout(
        title=f"{title}<br><sub>{EXPLORER_DISCLAIMER}</sub>",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=width,
        height=height,
        hovermode="closest",
    )
    
    if output_path:
        fig.write_html(str(output_path))
    
    return fig

