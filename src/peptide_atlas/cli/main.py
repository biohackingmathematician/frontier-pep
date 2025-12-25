"""
Command-line interface for the Peptide Atlas.

CRITICAL DISCLAIMER:
This tool is for RESEARCH AND EDUCATIONAL PURPOSES ONLY.
No dosing, no protocols, no therapeutic recommendations.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from peptide_atlas import DISCLAIMER, __version__

app = typer.Typer(
    name="peptide-atlas",
    help="Frontier Peptide Atlas - Graph and TDA analysis of peptide mechanism space",
    add_completion=False,
)
console = Console()


def print_disclaimer():
    """Print the standard disclaimer."""
    console.print(Panel(
        DISCLAIMER,
        title="CRITICAL DISCLAIMER",
        border_style="red",
    ))


@app.callback()
def main_callback():
    """
    Frontier Peptide Atlas CLI.
    
    Research and educational tool for analyzing peptide mechanism space.
    """
    pass


@app.command()
def build_kg(
    output: Path = typer.Option(
        Path("data/processed/kg.json"),
        "--output", "-o",
        help="Output path for knowledge graph JSON",
    ),
    validate: bool = typer.Option(
        True,
        "--validate/--no-validate",
        help="Validate the knowledge graph after building",
    ),
):
    """Build the peptide knowledge graph from curated data."""
    print_disclaimer()
    
    from peptide_atlas.kg import build_knowledge_graph
    from peptide_atlas.kg.builder import KnowledgeGraphBuilder
    from peptide_atlas.data.validators import validate_knowledge_graph
    
    console.print("\n[bold blue]Building Knowledge Graph...[/bold blue]\n")
    
    # Build KG
    builder = KnowledgeGraphBuilder()
    kg = builder.build()
    
    # Validate if requested
    if validate:
        console.print("\n[bold]Validating...[/bold]")
        result = validate_knowledge_graph(kg)
        
        if not result.is_valid:
            console.print(f"[red]Validation failed with {len(result.errors)} errors[/red]")
            for error in result.errors[:10]:
                console.print(f"  [red]• {error}[/red]")
            raise typer.Exit(1)
        
        if result.warnings:
            console.print(f"[yellow]Validation passed with {len(result.warnings)} warnings[/yellow]")
    
    # Save
    builder.save(output)
    
    # Summary
    table = Table(title="Knowledge Graph Summary")
    table.add_column("Entity Type", style="cyan")
    table.add_column("Count", justify="right")
    
    table.add_row("Peptides", str(len(kg.peptides)))
    table.add_row("Targets", str(len(kg.targets)))
    table.add_row("Pathways", str(len(kg.pathways)))
    table.add_row("Effect Domains", str(len(kg.effect_domains)))
    table.add_row("Risks", str(len(kg.risks)))
    table.add_row("---", "---")
    table.add_row("Total Nodes", str(kg.node_count))
    table.add_row("Total Edges", str(kg.edge_count))
    
    console.print(table)
    console.print(f"\n[dim]Saved to {output}[/dim]\n")


@app.command()
def train(
    config: Path = typer.Option(
        Path("configs/model_config.yaml"),
        "--config", "-c",
        help="Path to model configuration YAML",
    ),
    kg_path: Path = typer.Option(
        Path("data/processed/kg.json"),
        "--kg", "-k",
        help="Path to knowledge graph JSON",
    ),
    output_dir: Path = typer.Option(
        Path("outputs/models"),
        "--output", "-o",
        help="Output directory for trained model",
    ),
):
    """Train the GNN encoder on the knowledge graph."""
    print_disclaimer()
    
    console.print("\n[bold blue]Training GNN Encoder...[/bold blue]\n")
    console.print("[yellow]Note: Full training requires PyTorch and PyTorch Geometric[/yellow]\n")
    
    # Load config
    from peptide_atlas.data.loaders import load_yaml_config, load_knowledge_graph
    
    model_config = load_yaml_config(config)
    console.print(f"Loaded config from {config}")
    
    # Load KG
    kg = load_knowledge_graph(kg_path)
    console.print(f"Loaded KG with {kg.node_count} nodes")
    
    # TODO: Implement full training loop
    console.print("\n[yellow]Training implementation pending - see scripts/train_gnn.py[/yellow]")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"\n[green]Output directory prepared: {output_dir}[/green]\n")


@app.command()
def analyze_tda(
    config: Path = typer.Option(
        Path("configs/tda_config.yaml"),
        "--config", "-c",
        help="Path to TDA configuration YAML",
    ),
    embeddings_path: Optional[Path] = typer.Option(
        None,
        "--embeddings", "-e",
        help="Path to embeddings file (.npy or .pt)",
    ),
    output_dir: Path = typer.Option(
        Path("outputs/tda"),
        "--output", "-o",
        help="Output directory for TDA results",
    ),
):
    """Run topological data analysis on peptide embeddings."""
    print_disclaimer()
    
    console.print("\n[bold blue]Running TDA Analysis...[/bold blue]\n")
    
    from peptide_atlas.data.loaders import load_yaml_config
    
    tda_config = load_yaml_config(config)
    console.print(f"Loaded TDA config from {config}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # TODO: Implement TDA pipeline
    console.print("\n[yellow]TDA implementation pending - see scripts/run_tda.py[/yellow]")
    console.print(f"\n[green]Output directory prepared: {output_dir}[/green]\n")


@app.command()
def visualize(
    kg_path: Path = typer.Option(
        Path("data/processed/kg.json"),
        "--kg", "-k",
        help="Path to knowledge graph JSON",
    ),
    embeddings_path: Optional[Path] = typer.Option(
        None,
        "--embeddings", "-e",
        help="Path to 2D embeddings file",
    ),
    output: Path = typer.Option(
        Path("outputs/world_map.html"),
        "--output", "-o",
        help="Output path for visualization HTML",
    ),
    theme: str = typer.Option(
        "dark",
        "--theme", "-t",
        help="Color theme (dark/light)",
    ),
):
    """Generate the world map visualization."""
    print_disclaimer()
    
    console.print("\n[bold blue]Generating World Map Visualization...[/bold blue]\n")
    
    import numpy as np
    from peptide_atlas.data.loaders import load_knowledge_graph
    from peptide_atlas.viz.world_map import create_world_map
    
    # Load KG
    kg = load_knowledge_graph(kg_path)
    
    # Generate embeddings if not provided
    if embeddings_path and embeddings_path.exists():
        embeddings_2d = np.load(embeddings_path)
    else:
        console.print("[yellow]No embeddings provided, using random layout[/yellow]")
        n_peptides = len(kg.peptides)
        embeddings_2d = np.random.randn(n_peptides, 2)
    
    # Extract peptide data
    names = [p.canonical_name for p in kg.peptides]
    classes = [p.peptide_class.value for p in kg.peptides]
    tiers = [p.evidence_tier.value for p in kg.peptides]
    descriptions = [p.description or "" for p in kg.peptides]
    
    # Create visualization
    output.parent.mkdir(parents=True, exist_ok=True)
    
    fig = create_world_map(
        embeddings_2d=embeddings_2d,
        names=names,
        peptide_classes=classes,
        evidence_tiers=tiers,
        descriptions=descriptions,
        title="Frontier Peptide Atlas",
        theme=theme,
        output_path=str(output),
    )
    
    console.print(f"\n[dim]Visualization saved to {output}[/dim]\n")


@app.command()
def list_peptides(
    filter_class: Optional[str] = typer.Option(
        None,
        "--class", "-c",
        help="Filter by peptide class",
    ),
    filter_tier: Optional[str] = typer.Option(
        None,
        "--tier", "-t",
        help="Filter by minimum evidence tier (1-6)",
    ),
):
    """List peptides in the catalog."""
    print_disclaimer()
    
    from peptide_atlas.data.peptide_catalog import get_curated_peptides
    from peptide_atlas.constants import EvidenceTier, PeptideClass
    
    peptides = get_curated_peptides()
    
    # Apply filters
    if filter_class:
        try:
            pc = PeptideClass(filter_class)
            peptides = [p for p in peptides if p.peptide_class == pc]
        except ValueError:
            console.print(f"[red]Unknown class: {filter_class}[/red]")
            console.print(f"Available: {[e.value for e in PeptideClass]}")
            raise typer.Exit(1)
    
    if filter_tier:
        tier_map = {
            "1": EvidenceTier.TIER_1_APPROVED,
            "2": EvidenceTier.TIER_2_LATE_CLINICAL,
            "3": EvidenceTier.TIER_3_EARLY_CLINICAL,
            "4": EvidenceTier.TIER_4_PRECLINICAL,
            "5": EvidenceTier.TIER_5_MECHANISTIC,
            "6": EvidenceTier.TIER_6_ANECDOTAL,
        }
        min_tier = tier_map.get(filter_tier)
        if min_tier:
            min_score = min_tier.confidence_score
            peptides = [p for p in peptides if p.evidence_tier.confidence_score >= min_score]
    
    # Display table
    table = Table(title=f"Peptide Catalog ({len(peptides)} peptides)")
    table.add_column("Name", style="cyan")
    table.add_column("Class", style="green")
    table.add_column("Evidence", style="yellow")
    table.add_column("Status")
    
    for p in peptides:
        table.add_row(
            p.canonical_name,
            p.peptide_class.value.replace("_", " ").title(),
            p.evidence_tier.value.replace("tier_", "Tier ").replace("_", " ").title(),
            p.regulatory_status.value.replace("_", " ").title(),
        )
    
    console.print(table)


@app.command()
def version():
    """Show version information."""
    console.print(f"Peptide Atlas v{__version__}")
    console.print("\n[dim]Research and educational use only.[/dim]")


@app.command()
def validate(
    kg_path: Path = typer.Argument(
        ...,
        help="Path to knowledge graph JSON to validate",
    ),
):
    """Validate a knowledge graph file."""
    print_disclaimer()
    
    from peptide_atlas.data.loaders import load_knowledge_graph
    from peptide_atlas.data.validators import (
        validate_knowledge_graph,
        validate_no_prohibited_content,
    )
    
    console.print(f"\n[bold]Validating {kg_path}...[/bold]\n")
    
    kg = load_knowledge_graph(kg_path)
    
    # Run validation
    result = validate_knowledge_graph(kg)
    
    # Check for prohibited content
    prohibited = validate_no_prohibited_content(kg)
    
    # Report results
    if result.is_valid and prohibited.is_valid:
        console.print("[dim]Validation passed[/dim]")
    else:
        console.print("[red]✗ Validation failed[/red]")
        
        for error in result.errors + prohibited.errors:
            console.print(f"  [red]• {error}[/red]")
    
    if result.warnings:
        console.print(f"\n[yellow]Warnings ({len(result.warnings)}):[/yellow]")
        for warning in result.warnings[:10]:
            console.print(f"  [yellow]• {warning}[/yellow]")


if __name__ == "__main__":
    app()

