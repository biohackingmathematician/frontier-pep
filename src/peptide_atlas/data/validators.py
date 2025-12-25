"""
Data validation utilities for the Peptide Atlas.

Ensures data integrity and compliance with ethical constraints.

REMINDER: This project is for research and education only.
No dosing, no protocols, no therapeutic recommendations.
"""

from __future__ import annotations

import re
from typing import Optional

from loguru import logger
from pydantic import ValidationError

from peptide_atlas.constants import EvidenceTier
from peptide_atlas.data.schemas import (
    AssociatedWithEffectEdge,
    AssociatedWithRiskEdge,
    KnowledgeGraph,
    PeptideNode,
)


class ValidationResult:
    """Result of a validation check."""
    
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
    
    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0
    
    def add_error(self, message: str) -> None:
        self.errors.append(message)
        logger.error(f"Validation error: {message}")
    
    def add_warning(self, message: str) -> None:
        self.warnings.append(message)
        logger.warning(f"Validation warning: {message}")
    
    def __str__(self) -> str:
        if self.is_valid:
            return f"Valid (with {len(self.warnings)} warnings)"
        return f"Invalid: {len(self.errors)} errors, {len(self.warnings)} warnings"


# Patterns that indicate prohibited dosing information
DOSING_PATTERNS = [
    r"\d+\s*(mg|mcg|ug|μg|iu|IU|units?)\b",  # Dose amounts
    r"\d+x\s*(?:per|/)\s*(?:day|week|month)",  # Frequency
    r"\b(?:dose|dosage|dosing)\b",  # Dose-related terms
    r"\b(?:cycle|protocol|stack)\b",  # Protocol terms
    r"\b(?:inject|administer)\s+\d+",  # Administration with amounts
    r"\d+\s*(?:days?|weeks?|months?)\s+(?:on|off)",  # Cycling patterns
]


def check_no_dosing_info(text: Optional[str]) -> list[str]:
    """
    Check that text doesn't contain dosing information.
    
    Returns list of violations found.
    """
    if text is None:
        return []
    
    violations = []
    for pattern in DOSING_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            violations.append(f"Potential dosing info found: pattern '{pattern}' matched in '{text[:100]}...'")
    
    return violations


def validate_peptide(peptide: PeptideNode) -> ValidationResult:
    """Validate a single peptide entry."""
    result = ValidationResult()
    
    # Check required fields
    if not peptide.canonical_name:
        result.add_error("Peptide missing canonical_name")
    
    # Check evidence tier is set
    if peptide.evidence_tier == EvidenceTier.TIER_UNKNOWN:
        result.add_warning(f"Peptide '{peptide.canonical_name}' has UNKNOWN evidence tier")
    
    # Check for dosing information in description
    if peptide.description:
        violations = check_no_dosing_info(peptide.description)
        for v in violations:
            result.add_error(f"Peptide '{peptide.canonical_name}': {v}")
    
    # Check for dosing information in notes
    if peptide.notes:
        violations = check_no_dosing_info(peptide.notes)
        for v in violations:
            result.add_error(f"Peptide '{peptide.canonical_name}' notes: {v}")
    
    return result


def validate_knowledge_graph(kg: KnowledgeGraph) -> ValidationResult:
    """
    Validate an entire knowledge graph.
    
    Checks:
    - All peptides have required fields
    - No dosing information in any text fields
    - All effect/risk edges have evidence tiers
    - Referential integrity (edges reference valid nodes)
    """
    result = ValidationResult()
    
    # Collect all node IDs
    peptide_ids = {p.id for p in kg.peptides}
    target_ids = {t.id for t in kg.targets}
    pathway_ids = {p.id for p in kg.pathways}
    effect_ids = {e.id for e in kg.effect_domains}
    risk_ids = {r.id for r in kg.risks}
    
    # Validate each peptide
    for peptide in kg.peptides:
        peptide_result = validate_peptide(peptide)
        result.errors.extend(peptide_result.errors)
        result.warnings.extend(peptide_result.warnings)
    
    # Check for duplicate peptide names
    names = [p.canonical_name.lower() for p in kg.peptides]
    seen = set()
    for name in names:
        if name in seen:
            result.add_error(f"Duplicate peptide name: {name}")
        seen.add(name)
    
    # Validate binds edges
    for edge in kg.binds_edges:
        if edge.source_id not in peptide_ids:
            result.add_error(f"Binds edge source {edge.source_id} not found in peptides")
        if edge.target_id not in target_ids:
            result.add_error(f"Binds edge target {edge.target_id} not found in targets")
    
    # Validate modulates edges
    for edge in kg.modulates_edges:
        if edge.source_id not in peptide_ids:
            result.add_error(f"Modulates edge source {edge.source_id} not found in peptides")
        if edge.target_id not in pathway_ids:
            result.add_error(f"Modulates edge target {edge.target_id} not found in pathways")
    
    # Validate effect edges
    for edge in kg.effect_edges:
        if edge.source_id not in peptide_ids:
            result.add_error(f"Effect edge source {edge.source_id} not found in peptides")
        if edge.target_id not in effect_ids:
            result.add_error(f"Effect edge target {edge.target_id} not found in effect domains")
        # Check evidence tier is set
        if edge.evidence_tier == EvidenceTier.TIER_UNKNOWN:
            result.add_warning(f"Effect edge has UNKNOWN evidence tier")
    
    # Validate risk edges
    for edge in kg.risk_edges:
        if edge.source_id not in peptide_ids:
            result.add_error(f"Risk edge source {edge.source_id} not found in peptides")
        if edge.target_id not in risk_ids:
            result.add_error(f"Risk edge target {edge.target_id} not found in risks")
        # Check evidence tier is set
        if edge.evidence_tier == EvidenceTier.TIER_UNKNOWN:
            result.add_warning(f"Risk edge has UNKNOWN evidence tier")
    
    # Summary
    logger.info(f"Validation complete: {result}")
    
    return result


def validate_no_prohibited_content(kg: KnowledgeGraph) -> ValidationResult:
    """
    Specifically check for prohibited content (dosing, protocols).
    
    This is a stricter check that should pass before any export.
    """
    result = ValidationResult()
    
    # Check all peptide text fields
    for peptide in kg.peptides:
        for field_name in ["description", "notes"]:
            text = getattr(peptide, field_name)
            if text:
                violations = check_no_dosing_info(text)
                for v in violations:
                    result.add_error(f"PROHIBITED: {peptide.canonical_name}.{field_name}: {v}")
    
    # Check edge notes
    for edge in kg.effect_edges:
        if edge.notes:
            violations = check_no_dosing_info(edge.notes)
            for v in violations:
                result.add_error(f"PROHIBITED: Effect edge notes: {v}")
    
    for edge in kg.risk_edges:
        if edge.notes:
            violations = check_no_dosing_info(edge.notes)
            for v in violations:
                result.add_error(f"PROHIBITED: Risk edge notes: {v}")
    
    return result

