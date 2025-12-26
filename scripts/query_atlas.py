#!/usr/bin/env python3
"""
Query the Peptide Atlas from command line.

CRITICAL DISCLAIMER:
This tool is for RESEARCH AND EDUCATIONAL PURPOSES ONLY.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from peptide_atlas import print_disclaimer
from peptide_atlas.api.atlas import PeptideAtlas


def main():
    print_disclaimer()
    
    parser = argparse.ArgumentParser(
        description="Query the Peptide Atlas"
    )
    parser.add_argument(
        "--data-dir", "-d",
        type=Path,
        default=Path("data/processed"),
        help="Data directory",
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Stats command
    subparsers.add_parser("stats", help="Show atlas statistics")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List peptides")
    list_parser.add_argument("--class", dest="peptide_class", help="Filter by class")
    list_parser.add_argument("--min-evidence", type=int, default=6, help="Min evidence tier")
    
    # Similar command
    similar_parser = subparsers.add_parser("similar", help="Find similar peptides")
    similar_parser.add_argument("peptide", help="Query peptide name")
    similar_parser.add_argument("-k", type=int, default=5, help="Number of results")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get peptide info")
    info_parser.add_argument("peptide", help="Peptide name")
    
    args = parser.parse_args()
    
    # Load atlas
    try:
        atlas = PeptideAtlas.load(args.data_dir, show_disclaimer=False)
    except FileNotFoundError:
        print(f"Error: Knowledge graph not found in {args.data_dir}")
        print("Run 'python scripts/build_kg.py' first to build the knowledge graph.")
        return 1
    
    if args.command == "stats":
        stats = atlas.stats()
        print(f"\nPeptide Atlas v{stats.version}")
        print(f"  Peptides: {stats.num_peptides}")
        print(f"  Targets: {stats.num_targets}")
        print(f"  Pathways: {stats.num_pathways}")
        print(f"  Effect Domains: {stats.num_effect_domains}")
        print(f"  Risks: {stats.num_risks}")
        print(f"  Total Edges: {stats.num_edges}")
        if stats.embedding_dim:
            print(f"  Embedding Dim: {stats.embedding_dim}")
    
    elif args.command == "list":
        if args.peptide_class:
            peptides = atlas.query_by_class(args.peptide_class)
        else:
            peptides = atlas.query_by_evidence(min_tier=args.min_evidence)
        
        print(f"\n{len(peptides)} peptides:\n")
        for p in peptides:
            print(f"  {p.canonical_name}")
            print(f"    Class: {p.peptide_class.value}")
            print(f"    Evidence: {p.evidence_tier.value}")
            print()
    
    elif args.command == "similar":
        if not atlas.has_embeddings:
            print("Error: Embeddings not loaded")
            print("Train embeddings first with 'python scripts/train_gnn.py'")
            return 1
        
        results = atlas.find_similar(args.peptide, k=args.k)
        
        print(f"\nPeptides similar to {args.peptide}:\n")
        for r in results:
            print(f"  {r.peptide.canonical_name}")
            print(f"    Similarity: {r.similarity:.3f}")
            print(f"    Class: {r.peptide.peptide_class.value}")
            print()
    
    elif args.command == "info":
        peptide = atlas.get_peptide(args.peptide)
        
        if not peptide:
            print(f"Peptide not found: {args.peptide}")
            return 1
        
        print(f"\n{peptide.canonical_name}")
        print(f"  Class: {peptide.peptide_class.value}")
        print(f"  Evidence Tier: {peptide.evidence_tier.value}")
        print(f"  Regulatory Status: {peptide.regulatory_status.value}")
        if peptide.description:
            print(f"  Description: {peptide.description}")
        if peptide.synonyms:
            print(f"  Synonyms: {', '.join(peptide.synonyms)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

