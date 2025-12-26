"""
Interactive dashboard for exploring the Peptide Atlas.

NOTE: This is an exploration tool, not the core deliverable.

REMINDER: This project is for research and education only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger


def launch_explorer(
    data_dir: str = "data/processed",
    port: int = 8050,
    debug: bool = False,
) -> None:
    """
    Launch interactive explorer dashboard.
    
    Requires: pip install dash plotly
    
    Args:
        data_dir: Directory with kg.json and embeddings
        port: Port to serve on
        debug: Enable debug mode
    """
    try:
        import dash
        from dash import dcc, html
        from dash.dependencies import Input, Output
    except ImportError:
        raise ImportError(
            "Dash required for explorer: pip install dash plotly"
        )
    
    from peptide_atlas.api import PeptideAtlas
    
    # Load atlas
    atlas = PeptideAtlas.load(data_dir, show_disclaimer=True)
    
    # Create Dash app
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        html.H1("Peptide Atlas Explorer"),
        html.P(
            "RESEARCH USE ONLY - Not medical advice. "
            "No dosing or protocol recommendations.",
            style={"color": "#666", "fontStyle": "italic", "fontWeight": "bold"}
        ),
        html.Hr(),
        
        # Stats
        html.Div([
            html.H3("Atlas Statistics"),
            html.P(f"Peptides: {atlas.num_peptides}"),
            html.P(f"Targets: {atlas.num_targets}"),
            html.P(f"Pathways: {atlas.num_pathways}"),
        ]),
        
        html.Hr(),
        
        # Query interface
        html.Div([
            html.H3("Query"),
            dcc.Dropdown(
                id="class-dropdown",
                options=[
                    {"label": c.replace("_", " ").title(), "value": c}
                    for c in atlas.list_classes()
                ],
                placeholder="Select peptide class...",
            ),
            html.Div(id="query-results"),
        ]),
        
        html.Hr(),
        
        # Similarity search
        html.Div([
            html.H3("Similarity Search"),
            dcc.Dropdown(
                id="peptide-dropdown",
                options=[
                    {"label": name, "value": name}
                    for name in atlas.list_peptides()
                ],
                placeholder="Select peptide...",
            ),
            html.Div(id="similar-results"),
        ]) if atlas.has_embeddings else html.P("Embeddings not loaded"),
    ])
    
    @app.callback(
        Output("query-results", "children"),
        Input("class-dropdown", "value"),
    )
    def update_query(peptide_class):
        if not peptide_class:
            return ""
        
        peptides = atlas.query_by_class(peptide_class)
        return html.Ul([
            html.Li(f"{p.canonical_name} ({p.evidence_tier.value})")
            for p in peptides
        ])
    
    @app.callback(
        Output("similar-results", "children"),
        Input("peptide-dropdown", "value"),
    )
    def update_similar(peptide_name):
        if not peptide_name or not atlas.has_embeddings:
            return ""
        
        try:
            similar = atlas.find_similar(peptide_name, k=5)
            return html.Ul([
                html.Li(f"{r.peptide.canonical_name} (similarity: {r.similarity:.2f})")
                for r in similar
            ])
        except Exception as e:
            return html.P(f"Error: {e}")
    
    logger.info(f"Launching explorer on http://localhost:{port}")
    app.run_server(port=port, debug=debug)

