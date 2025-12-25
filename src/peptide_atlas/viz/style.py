"""
Visual styling for the Peptide Atlas.

REMINDER: This project is for research and education only.
"""

from peptide_atlas.constants import (
    EFFECT_DOMAIN_COLORS,
    EVIDENCE_TIER_COLORS,
    PEPTIDE_CLASS_COLORS,
    EffectDomain,
    EvidenceTier,
    PeptideClass,
)

# Standard disclaimer text
DISCLAIMER_TEXT = """
RESEARCH AND EDUCATIONAL USE ONLY

This visualization does NOT constitute medical advice.
No dosing information or therapeutic recommendations are provided.
Consult a healthcare professional for any medical decisions.
"""

# Dark theme colors
DARK_THEME = {
    "background": "#0a0a0f",
    "paper_bg": "#12121a",
    "text": "#e0e0e0",
    "text_secondary": "#a0a0a0",
    "grid": "#2a2a35",
    "accent": "#4a9eff",
}

# Light theme colors
LIGHT_THEME = {
    "background": "#ffffff",
    "paper_bg": "#f5f5f5",
    "text": "#1a1a1a",
    "text_secondary": "#666666",
    "grid": "#e0e0e0",
    "accent": "#2563eb",
}


def get_color_palette(
    color_by: str = "peptide_class",
) -> dict[str, str]:
    """
    Get color palette for visualization.
    
    Args:
        color_by: What to color by (peptide_class, evidence_tier, effect_domain)
        
    Returns:
        Dictionary mapping category to hex color
    """
    if color_by == "peptide_class":
        return {pc.value: color for pc, color in PEPTIDE_CLASS_COLORS.items()}
    elif color_by == "evidence_tier":
        return {et.value: color for et, color in EVIDENCE_TIER_COLORS.items()}
    elif color_by == "effect_domain":
        return {ed.value: color for ed, color in EFFECT_DOMAIN_COLORS.items()}
    else:
        return {pc.value: color for pc, color in PEPTIDE_CLASS_COLORS.items()}


def get_theme(name: str = "dark") -> dict[str, str]:
    """Get theme colors by name."""
    if name == "light":
        return LIGHT_THEME
    return DARK_THEME


def get_evidence_color(tier: EvidenceTier) -> str:
    """Get color for an evidence tier."""
    return EVIDENCE_TIER_COLORS.get(tier, "#7f7f7f")


def get_class_color(peptide_class: PeptideClass) -> str:
    """Get color for a peptide class."""
    return PEPTIDE_CLASS_COLORS.get(peptide_class, "#7f7f7f")


def format_hover_text(
    name: str,
    peptide_class: str,
    evidence_tier: str,
    description: str = "",
) -> str:
    """
    Format hover text for a peptide node.
    
    Explicitly excludes any dosing information.
    """
    text = f"<b>{name}</b><br>"
    text += f"Class: {peptide_class.replace('_', ' ').title()}<br>"
    text += f"Evidence: {evidence_tier.replace('_', ' ').replace('tier ', 'Tier ')}<br>"
    
    if description:
        # Truncate long descriptions
        if len(description) > 100:
            description = description[:97] + "..."
        text += f"<i>{description}</i><br>"
    
    text += "<br><span style='color:#ff6b6b;font-size:10px'>Research use only</span>"
    
    return text


# Node size scales
NODE_SIZE_SCALE = {
    "small": (8, 20),
    "medium": (12, 30),
    "large": (15, 40),
}


def get_node_size(
    value: float,
    scale: str = "medium",
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> float:
    """
    Map a value to a node size.
    
    Args:
        value: Value to map
        scale: Size scale preset
        min_val: Minimum value in data
        max_val: Maximum value in data
        
    Returns:
        Node size in pixels
    """
    size_range = NODE_SIZE_SCALE.get(scale, NODE_SIZE_SCALE["medium"])
    
    if max_val == min_val:
        return (size_range[0] + size_range[1]) / 2
    
    normalized = (value - min_val) / (max_val - min_val)
    return size_range[0] + normalized * (size_range[1] - size_range[0])

