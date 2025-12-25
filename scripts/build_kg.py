#!/usr/bin/env python3
"""
Build the Peptide Atlas Knowledge Graph.

CRITICAL DISCLAIMER:
This script is for RESEARCH AND EDUCATIONAL PURPOSES ONLY.
No dosing, no protocols, no therapeutic recommendations.
"""

import argparse
from pathlib import Path

from peptide_atlas import print_disclaimer
from peptide_atlas.kg.builder import KnowledgeGraphBuilder
from peptide_atlas.data.validators import validate_knowledge_graph


def main():
    print_disclaimer()
    
    parser = argparse.ArgumentParser(
        description="Build the Peptide Atlas Knowledge Graph"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/processed/kg.json"),
        help="Output path for knowledge graph JSON",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate the knowledge graph after building",
    )
    args = parser.parse_args()
    
    print("\n=== Building Knowledge Graph ===\n")
    
    # Build
    builder = KnowledgeGraphBuilder()
    kg = builder.build()
    
    # Validate
    if args.validate:
        print("\n=== Validating ===\n")
        result = validate_knowledge_graph(kg)
        
        if not result.is_valid:
            print(f"ERROR: Validation failed with {len(result.errors)} errors")
            for error in result.errors:
                print(f"  - {error}")
            return 1
        
        print("Validation passed")
    
    # Save
    builder.save(args.output)
    
    print(f"\n=== Summary ===")
    print(f"Nodes: {kg.node_count}")
    print(f"Edges: {kg.edge_count}")
    print(f"Output: {args.output}")
    print()
    
    return 0


if __name__ == "__main__":
    exit(main())

