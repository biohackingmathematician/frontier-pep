"""
Constants and enumerations for the Peptide Atlas.

REMINDER: This project is for research and education only.
No dosing, no protocols, no therapeutic recommendations.
"""

from enum import Enum


class EvidenceTier(str, Enum):
    """Evidence quality classification for peptide-effect relationships."""
    
    TIER_1_APPROVED = "tier_1_approved"  # Regulatory approval
    TIER_2_LATE_CLINICAL = "tier_2_late_clinical"  # Phase II/III RCT
    TIER_3_EARLY_CLINICAL = "tier_3_early_clinical"  # Phase I / early Phase II
    TIER_4_PRECLINICAL = "tier_4_preclinical"  # Animal studies only
    TIER_5_MECHANISTIC = "tier_5_mechanistic"  # In vitro / theoretical
    TIER_6_ANECDOTAL = "tier_6_anecdotal"  # Case reports, uncontrolled
    TIER_UNKNOWN = "tier_unknown"  # Insufficient data
    
    @property
    def confidence_score(self) -> float:
        """Numerical confidence score (1.0 = highest, 0.0 = lowest)."""
        scores = {
            self.TIER_1_APPROVED: 1.0,
            self.TIER_2_LATE_CLINICAL: 0.85,
            self.TIER_3_EARLY_CLINICAL: 0.65,
            self.TIER_4_PRECLINICAL: 0.45,
            self.TIER_5_MECHANISTIC: 0.25,
            self.TIER_6_ANECDOTAL: 0.15,
            self.TIER_UNKNOWN: 0.05,
        }
        return scores.get(self, 0.05)


class PeptideClass(str, Enum):
    """Primary peptide classification."""
    
    GHRH_ANALOG = "ghrh_analog"  # GHRH receptor agonists
    GHS_GHRELIN_MIMETIC = "ghs_ghrelin_mimetic"  # Ghrelin receptor agonists
    IGF_AXIS = "igf_axis"  # IGF-1 and related
    REGENERATIVE_REPAIR = "regenerative_repair"  # Tissue healing peptides
    THYMIC_IMMUNE = "thymic_immune"  # Thymic/immune modulators
    METABOLIC_INCRETIN = "metabolic_incretin"  # Metabolic peptides (non-GLP-1 focus)
    CNS_NEUROTROPHIC = "cns_neurotrophic"  # Neuroprotective peptides
    LONGEVITY_CELLULAR = "longevity_cellular"  # Aging/senescence related
    GLP1_REFERENCE = "glp1_reference"  # GLP-1 agonists (reference only, not focus)
    OTHER = "other"


class RegulatoryStatus(str, Enum):
    """Regulatory/approval status."""
    
    APPROVED = "approved"  # FDA/EMA approved for some indication
    INVESTIGATIONAL = "investigational"  # In clinical trials
    RESEARCH_ONLY = "research_only"  # Research chemical, no human approval
    DISCONTINUED = "discontinued"  # Development stopped
    UNKNOWN = "unknown"


class TargetType(str, Enum):
    """Molecular target types."""
    
    RECEPTOR = "receptor"
    ENZYME = "enzyme"
    TRANSPORTER = "transporter"
    ION_CHANNEL = "ion_channel"
    CELL_TYPE = "cell_type"
    TISSUE = "tissue"
    ORGAN = "organ"
    SIGNALING_COMPLEX = "signaling_complex"
    OTHER = "other"


class PathwayCategory(str, Enum):
    """Biological pathway categories."""
    
    GROWTH_ANABOLIC = "growth_anabolic"  # GH/IGF/mTOR
    METABOLIC = "metabolic"  # Glucose, lipid metabolism
    IMMUNE = "immune"  # Immune signaling
    REPAIR_REGENERATION = "repair_regeneration"  # Wound healing, ECM
    NEUROLOGICAL = "neurological"  # CNS pathways
    STRESS_RESPONSE = "stress_response"  # Cortisol, HPA axis
    AGING_LONGEVITY = "aging_longevity"  # Senescence, autophagy
    VASCULAR = "vascular"  # Angiogenesis, blood flow
    OTHER = "other"


class EffectDomain(str, Enum):
    """Effect/outcome domains."""
    
    ANABOLIC_BODY_COMPOSITION = "anabolic_body_composition"
    REGENERATION_REPAIR = "regeneration_repair"
    IMMUNE_RESILIENCE = "immune_resilience"
    METABOLIC_GLYCEMIC = "metabolic_glycemic"
    METABOLIC_LIPID = "metabolic_lipid"
    CNS_COGNITIVE = "cns_cognitive"
    CNS_MOOD = "cns_mood"
    CNS_NEUROPROTECTION = "cns_neuroprotection"
    LONGEVITY_AGING = "longevity_aging"
    PERFORMANCE_ENDURANCE = "performance_endurance"
    PERFORMANCE_STRENGTH = "performance_strength"
    RECOVERY = "recovery"
    SLEEP = "sleep"
    SKIN_HAIR = "skin_hair"
    SEXUAL_FUNCTION = "sexual_function"
    OTHER = "other"


class RiskCategory(str, Enum):
    """Adverse effect/risk categories."""
    
    GROWTH_PROLIFERATIVE = "growth_proliferative"  # Cancer risk, tissue growth
    METABOLIC = "metabolic"  # Glucose, insulin resistance
    CARDIOVASCULAR = "cardiovascular"  # Heart, blood pressure
    IMMUNE_OVERACTIVATION = "immune_overactivation"  # Autoimmunity, inflammation
    IMMUNE_SUPPRESSION = "immune_suppression"  # Infection risk
    CNS_PSYCHIATRIC = "cns_psychiatric"  # Mood, cognition adverse effects
    ENDOCRINE_SUPPRESSION = "endocrine_suppression"  # HPA/HPG suppression
    INJECTION_SITE = "injection_site"  # Local reactions
    FLUID_RETENTION = "fluid_retention"  # Edema, water retention
    MUSCULOSKELETAL = "musculoskeletal"  # Joint pain, CTS-like
    GASTROINTESTINAL = "gastrointestinal"  # Nausea, GI effects
    UNKNOWN_LONG_TERM = "unknown_long_term"  # Insufficient long-term data
    OTHER = "other"


class RiskSeverity(str, Enum):
    """Risk severity classification."""
    
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    POTENTIALLY_FATAL = "potentially_fatal"
    UNKNOWN = "unknown"


class Reversibility(str, Enum):
    """Risk reversibility."""
    
    REVERSIBLE = "reversible"
    PARTIALLY_REVERSIBLE = "partially_reversible"
    IRREVERSIBLE = "irreversible"
    UNKNOWN = "unknown"


class BindingType(str, Enum):
    """Receptor binding types."""
    
    AGONIST = "agonist"
    PARTIAL_AGONIST = "partial_agonist"
    ANTAGONIST = "antagonist"
    INVERSE_AGONIST = "inverse_agonist"
    ALLOSTERIC_POSITIVE = "allosteric_positive"
    ALLOSTERIC_NEGATIVE = "allosteric_negative"
    UNKNOWN = "unknown"


class EffectDirection(str, Enum):
    """Direction of effect."""
    
    BENEFICIAL = "beneficial"
    NEUTRAL = "neutral"
    HARMFUL = "harmful"
    AMBIGUOUS = "ambiguous"  # Context-dependent


class Confidence(str, Enum):
    """Confidence in a relationship."""
    
    HIGH = "high"  # Multiple high-quality sources
    MEDIUM = "medium"  # Some direct evidence
    LOW = "low"  # Limited or indirect evidence
    INFERRED = "inferred"  # Computationally inferred


class AdministrationRoute(str, Enum):
    """Routes of administration."""
    
    INJECTABLE_SC = "injectable_subcutaneous"
    INJECTABLE_IM = "injectable_intramuscular"
    INJECTABLE_IV = "injectable_intravenous"
    ORAL = "oral"
    INTRANASAL = "intranasal"
    SUBLINGUAL = "sublingual"
    TOPICAL = "topical"
    OTHER = "other"


class EvidenceSourceType(str, Enum):
    """Types of evidence sources."""
    
    RCT = "rct"  # Randomized controlled trial
    OBSERVATIONAL = "observational"
    CASE_SERIES = "case_series"
    CASE_REPORT = "case_report"
    PRECLINICAL_IN_VIVO = "preclinical_in_vivo"
    IN_VITRO = "in_vitro"
    MECHANISTIC_REVIEW = "mechanistic_review"
    REGULATORY_LABEL = "regulatory_label"
    DATABASE_ENTRY = "database_entry"


# Color palette for visualization (accessibility-considered)
EFFECT_DOMAIN_COLORS: dict[EffectDomain, str] = {
    EffectDomain.ANABOLIC_BODY_COMPOSITION: "#1f77b4",  # Blue
    EffectDomain.REGENERATION_REPAIR: "#2ca02c",  # Green
    EffectDomain.IMMUNE_RESILIENCE: "#9467bd",  # Purple
    EffectDomain.METABOLIC_GLYCEMIC: "#ff7f0e",  # Orange
    EffectDomain.METABOLIC_LIPID: "#ffbb78",  # Light orange
    EffectDomain.CNS_COGNITIVE: "#e377c2",  # Pink
    EffectDomain.CNS_MOOD: "#f7b6d2",  # Light pink
    EffectDomain.CNS_NEUROPROTECTION: "#c5b0d5",  # Light purple
    EffectDomain.LONGEVITY_AGING: "#17becf",  # Cyan
    EffectDomain.PERFORMANCE_ENDURANCE: "#7f7f7f",  # Gray
    EffectDomain.PERFORMANCE_STRENGTH: "#bcbd22",  # Yellow-green
    EffectDomain.RECOVERY: "#98df8a",  # Light green
    EffectDomain.SLEEP: "#aec7e8",  # Light blue
    EffectDomain.SKIN_HAIR: "#c49c94",  # Brown
    EffectDomain.SEXUAL_FUNCTION: "#f7b6d2",  # Light pink
    EffectDomain.OTHER: "#d3d3d3",  # Light gray
}

PEPTIDE_CLASS_COLORS: dict[PeptideClass, str] = {
    PeptideClass.GHRH_ANALOG: "#1f77b4",
    PeptideClass.GHS_GHRELIN_MIMETIC: "#2ca02c",
    PeptideClass.IGF_AXIS: "#ff7f0e",
    PeptideClass.REGENERATIVE_REPAIR: "#d62728",
    PeptideClass.THYMIC_IMMUNE: "#9467bd",
    PeptideClass.METABOLIC_INCRETIN: "#8c564b",
    PeptideClass.CNS_NEUROTROPHIC: "#e377c2",
    PeptideClass.LONGEVITY_CELLULAR: "#17becf",
    PeptideClass.GLP1_REFERENCE: "#7f7f7f",
    PeptideClass.OTHER: "#bcbd22",
}

EVIDENCE_TIER_COLORS: dict[EvidenceTier, str] = {
    EvidenceTier.TIER_1_APPROVED: "#1a9850",  # Dark green
    EvidenceTier.TIER_2_LATE_CLINICAL: "#66bd63",  # Green
    EvidenceTier.TIER_3_EARLY_CLINICAL: "#a6d96a",  # Light green
    EvidenceTier.TIER_4_PRECLINICAL: "#fee08b",  # Yellow
    EvidenceTier.TIER_5_MECHANISTIC: "#fdae61",  # Orange
    EvidenceTier.TIER_6_ANECDOTAL: "#f46d43",  # Red-orange
    EvidenceTier.TIER_UNKNOWN: "#d73027",  # Red
}

