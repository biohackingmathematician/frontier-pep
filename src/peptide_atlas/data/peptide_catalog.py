"""
Curated peptide catalog for the Frontier Peptide Atlas.

CRITICAL DISCLAIMER:
This catalog is for RESEARCH AND EDUCATIONAL PURPOSES ONLY.
- NO dosing information is provided or should be inferred.
- NO therapeutic recommendations are made.
- Inclusion does NOT imply safety, efficacy, or endorsement.
- Many compounds are experimental, off-label, or inadequately studied.
- Consult a qualified healthcare professional for any medical decisions.

Data sources: PubMed, DrugBank, ClinicalTrials.gov, regulatory labels.
"""

from peptide_atlas.constants import (
    AdministrationRoute,
    EvidenceTier,
    PeptideClass,
    RegulatoryStatus,
)
from peptide_atlas.data.schemas import PeptideNode


def get_curated_peptides() -> list[PeptideNode]:
    """
    Returns the curated list of peptides for the Atlas.
    
    NOTE: This is NOT a complete database. It is a research-focused subset.
    """
    
    peptides = []
    
    # =========================================================================
    # GH/GHRH AXIS PEPTIDES
    # =========================================================================
    
    peptides.extend([
        PeptideNode(
            canonical_name="Sermorelin",
            synonyms=["GRF 1-29", "Geref"],
            peptide_class=PeptideClass.GHRH_ANALOG,
            subclass="GHRH(1-29) analog",
            regulatory_status=RegulatoryStatus.APPROVED,  # Was FDA approved, now discontinued
            evidence_tier=EvidenceTier.TIER_2_LATE_CLINICAL,
            administration_routes=[AdministrationRoute.INJECTABLE_SC],
            description="Synthetic analog of GHRH(1-29). Stimulates pituitary GH release.",
            drugbank_id="DB00010",
        ),
        PeptideNode(
            canonical_name="Tesamorelin",
            synonyms=["Egrifta", "TH9507"],
            peptide_class=PeptideClass.GHRH_ANALOG,
            subclass="GHRH analog with trans-3-hexenoic acid",
            regulatory_status=RegulatoryStatus.APPROVED,  # FDA approved for HIV lipodystrophy
            evidence_tier=EvidenceTier.TIER_1_APPROVED,
            administration_routes=[AdministrationRoute.INJECTABLE_SC],
            description="GHRH analog approved for HIV-associated lipodystrophy.",
            drugbank_id="DB06285",
        ),
        PeptideNode(
            canonical_name="CJC-1295",
            synonyms=["DAC:GRF", "Drug Affinity Complex:GRF"],
            peptide_class=PeptideClass.GHRH_ANALOG,
            subclass="GHRH analog with DAC (extended half-life)",
            regulatory_status=RegulatoryStatus.RESEARCH_ONLY,
            evidence_tier=EvidenceTier.TIER_3_EARLY_CLINICAL,
            administration_routes=[AdministrationRoute.INJECTABLE_SC],
            description="Modified GHRH with drug affinity complex for extended duration.",
        ),
        PeptideNode(
            canonical_name="Modified GRF (1-29)",
            synonyms=["Mod GRF", "CJC-1295 without DAC", "tetrasubstituted GRF(1-29)"],
            peptide_class=PeptideClass.GHRH_ANALOG,
            subclass="Modified GHRH(1-29) without DAC",
            regulatory_status=RegulatoryStatus.RESEARCH_ONLY,
            evidence_tier=EvidenceTier.TIER_4_PRECLINICAL,
            administration_routes=[AdministrationRoute.INJECTABLE_SC],
            description="GHRH(1-29) with amino acid substitutions for stability.",
        ),
    ])
    
    # =========================================================================
    # GH SECRETAGOGUES (GHRELIN MIMETICS)
    # =========================================================================
    
    peptides.extend([
        PeptideNode(
            canonical_name="Ipamorelin",
            synonyms=["NNC 26-0161"],
            peptide_class=PeptideClass.GHS_GHRELIN_MIMETIC,
            subclass="Selective GHSR agonist",
            regulatory_status=RegulatoryStatus.RESEARCH_ONLY,
            evidence_tier=EvidenceTier.TIER_3_EARLY_CLINICAL,
            administration_routes=[AdministrationRoute.INJECTABLE_SC],
            description="Selective growth hormone secretagogue with minimal effect on cortisol/prolactin.",
        ),
        PeptideNode(
            canonical_name="GHRP-2",
            synonyms=["Pralmorelin", "KP-102"],
            peptide_class=PeptideClass.GHS_GHRELIN_MIMETIC,
            subclass="Hexapeptide GHSR agonist",
            regulatory_status=RegulatoryStatus.INVESTIGATIONAL,
            evidence_tier=EvidenceTier.TIER_3_EARLY_CLINICAL,
            administration_routes=[AdministrationRoute.INJECTABLE_SC, AdministrationRoute.INJECTABLE_IV],
            description="Synthetic hexapeptide GH secretagogue.",
        ),
        PeptideNode(
            canonical_name="GHRP-6",
            synonyms=["Growth Hormone Releasing Peptide 6"],
            peptide_class=PeptideClass.GHS_GHRELIN_MIMETIC,
            subclass="Hexapeptide GHSR agonist",
            regulatory_status=RegulatoryStatus.RESEARCH_ONLY,
            evidence_tier=EvidenceTier.TIER_3_EARLY_CLINICAL,
            administration_routes=[AdministrationRoute.INJECTABLE_SC],
            description="One of first synthetic GH secretagogues. Increases appetite.",
        ),
        PeptideNode(
            canonical_name="Hexarelin",
            synonyms=["Examorelin"],
            peptide_class=PeptideClass.GHS_GHRELIN_MIMETIC,
            subclass="Hexapeptide GHSR agonist",
            regulatory_status=RegulatoryStatus.INVESTIGATIONAL,
            evidence_tier=EvidenceTier.TIER_3_EARLY_CLINICAL,
            administration_routes=[AdministrationRoute.INJECTABLE_SC],
            description="Synthetic hexapeptide with strong GH release.",
        ),
        PeptideNode(
            canonical_name="MK-677",
            synonyms=["Ibutamoren", "L-163,191"],
            peptide_class=PeptideClass.GHS_GHRELIN_MIMETIC,
            subclass="Non-peptide GHSR agonist (oral)",
            regulatory_status=RegulatoryStatus.INVESTIGATIONAL,
            evidence_tier=EvidenceTier.TIER_2_LATE_CLINICAL,
            administration_routes=[AdministrationRoute.ORAL],
            description="Orally-active non-peptide ghrelin mimetic. Investigated for various indications.",
        ),
    ])
    
    # =========================================================================
    # IGF-1 / INSULIN AXIS
    # =========================================================================
    
    peptides.extend([
        PeptideNode(
            canonical_name="Mecasermin",
            synonyms=["Increlex", "rhIGF-1"],
            peptide_class=PeptideClass.IGF_AXIS,
            subclass="Recombinant human IGF-1",
            regulatory_status=RegulatoryStatus.APPROVED,  # FDA approved for IGF-1 deficiency
            evidence_tier=EvidenceTier.TIER_1_APPROVED,
            administration_routes=[AdministrationRoute.INJECTABLE_SC],
            description="Recombinant human IGF-1 approved for severe primary IGF-1 deficiency.",
            drugbank_id="DB01277",
        ),
        PeptideNode(
            canonical_name="IGF-1 LR3",
            synonyms=["Long R3 IGF-1", "LR3-IGF-1"],
            peptide_class=PeptideClass.IGF_AXIS,
            subclass="Modified IGF-1 with extended half-life",
            regulatory_status=RegulatoryStatus.RESEARCH_ONLY,
            evidence_tier=EvidenceTier.TIER_4_PRECLINICAL,
            administration_routes=[AdministrationRoute.INJECTABLE_SC],
            description="IGF-1 analog with N-terminal extension and Arg substitution. Research compound.",
        ),
        PeptideNode(
            canonical_name="MGF",
            synonyms=["Mechano Growth Factor", "IGF-1Ec"],
            peptide_class=PeptideClass.IGF_AXIS,
            subclass="IGF-1 splice variant",
            regulatory_status=RegulatoryStatus.RESEARCH_ONLY,
            evidence_tier=EvidenceTier.TIER_4_PRECLINICAL,
            administration_routes=[AdministrationRoute.INJECTABLE_SC],
            description="Splice variant of IGF-1 expressed in response to mechanical loading.",
        ),
        PeptideNode(
            canonical_name="PEG-MGF",
            synonyms=["Pegylated Mechano Growth Factor"],
            peptide_class=PeptideClass.IGF_AXIS,
            subclass="PEGylated MGF",
            regulatory_status=RegulatoryStatus.RESEARCH_ONLY,
            evidence_tier=EvidenceTier.TIER_5_MECHANISTIC,
            administration_routes=[AdministrationRoute.INJECTABLE_SC],
            description="PEGylated version of MGF for extended half-life. Research compound.",
        ),
    ])
    
    # =========================================================================
    # REGENERATIVE / TISSUE REPAIR
    # =========================================================================
    
    peptides.extend([
        PeptideNode(
            canonical_name="BPC-157",
            synonyms=["Body Protection Compound-157", "Bepecin", "PL 14736"],
            peptide_class=PeptideClass.REGENERATIVE_REPAIR,
            subclass="Gastric pentadecapeptide",
            regulatory_status=RegulatoryStatus.RESEARCH_ONLY,
            evidence_tier=EvidenceTier.TIER_4_PRECLINICAL,
            administration_routes=[AdministrationRoute.INJECTABLE_SC, AdministrationRoute.ORAL],
            description="Pentadecapeptide derived from gastric juice. Preclinical tissue repair research.",
        ),
        PeptideNode(
            canonical_name="TB-500",
            synonyms=["Thymosin Beta-4 Fragment"],
            peptide_class=PeptideClass.REGENERATIVE_REPAIR,
            subclass="Thymosin beta-4 active fragment",
            regulatory_status=RegulatoryStatus.RESEARCH_ONLY,
            evidence_tier=EvidenceTier.TIER_4_PRECLINICAL,
            administration_routes=[AdministrationRoute.INJECTABLE_SC],
            description="Active fragment of thymosin beta-4. Researched for tissue repair.",
        ),
        PeptideNode(
            canonical_name="Thymosin Beta-4",
            synonyms=["Tβ4", "TMSB4X"],
            peptide_class=PeptideClass.REGENERATIVE_REPAIR,
            subclass="Endogenous actin-sequestering peptide",
            regulatory_status=RegulatoryStatus.INVESTIGATIONAL,
            evidence_tier=EvidenceTier.TIER_3_EARLY_CLINICAL,
            administration_routes=[AdministrationRoute.INJECTABLE_SC],
            description="Endogenous 43-amino acid peptide involved in cell migration and wound healing.",
        ),
        PeptideNode(
            canonical_name="GHK-Cu",
            synonyms=["Copper peptide GHK", "Glycyl-L-histidyl-L-lysine:copper"],
            peptide_class=PeptideClass.REGENERATIVE_REPAIR,
            subclass="Copper-binding tripeptide",
            regulatory_status=RegulatoryStatus.RESEARCH_ONLY,
            evidence_tier=EvidenceTier.TIER_4_PRECLINICAL,
            administration_routes=[AdministrationRoute.TOPICAL, AdministrationRoute.INJECTABLE_SC],
            description="Naturally occurring tripeptide with copper. Used in skin care, researched for tissue repair.",
        ),
        PeptideNode(
            canonical_name="AOD9604",
            synonyms=["Anti-Obesity Drug 9604", "Tyr-hGH frag 177-191"],
            peptide_class=PeptideClass.REGENERATIVE_REPAIR,
            subclass="GH fragment 177-191 with tyrosine",
            regulatory_status=RegulatoryStatus.INVESTIGATIONAL,
            evidence_tier=EvidenceTier.TIER_3_EARLY_CLINICAL,
            administration_routes=[AdministrationRoute.INJECTABLE_SC, AdministrationRoute.ORAL],
            description="Modified fragment of GH C-terminus. Investigated for cartilage repair, metabolic effects.",
        ),
    ])
    
    # =========================================================================
    # THYMIC / IMMUNE PEPTIDES
    # =========================================================================
    
    peptides.extend([
        PeptideNode(
            canonical_name="Thymosin Alpha-1",
            synonyms=["Tα1", "Thymalfasin", "Zadaxin"],
            peptide_class=PeptideClass.THYMIC_IMMUNE,
            subclass="Thymic peptide",
            regulatory_status=RegulatoryStatus.APPROVED,  # Approved in some countries
            evidence_tier=EvidenceTier.TIER_2_LATE_CLINICAL,
            administration_routes=[AdministrationRoute.INJECTABLE_SC],
            description="28-amino acid thymic peptide. Approved in some countries for hepatitis, immunodeficiency.",
        ),
        PeptideNode(
            canonical_name="Thymalin",
            synonyms=["Thymic extract"],
            peptide_class=PeptideClass.THYMIC_IMMUNE,
            subclass="Thymic peptide complex",
            regulatory_status=RegulatoryStatus.APPROVED,  # Approved in Russia
            evidence_tier=EvidenceTier.TIER_3_EARLY_CLINICAL,
            administration_routes=[AdministrationRoute.INJECTABLE_IM],
            description="Thymic extract peptide complex used in some countries for immune modulation.",
        ),
        PeptideNode(
            canonical_name="LL-37",
            synonyms=["Cathelicidin", "CAP18", "hCAP18"],
            peptide_class=PeptideClass.THYMIC_IMMUNE,
            subclass="Human cathelicidin antimicrobial peptide",
            regulatory_status=RegulatoryStatus.RESEARCH_ONLY,
            evidence_tier=EvidenceTier.TIER_4_PRECLINICAL,
            administration_routes=[AdministrationRoute.INJECTABLE_SC, AdministrationRoute.TOPICAL],
            description="Endogenous antimicrobial peptide. Researched for infection and immune modulation.",
        ),
    ])
    
    # =========================================================================
    # CNS / NEUROTROPHIC PEPTIDES
    # =========================================================================
    
    peptides.extend([
        PeptideNode(
            canonical_name="Semax",
            synonyms=["ACTH 4-7 PGP"],
            peptide_class=PeptideClass.CNS_NEUROTROPHIC,
            subclass="ACTH fragment with Pro-Gly-Pro",
            regulatory_status=RegulatoryStatus.APPROVED,  # Approved in Russia/Ukraine
            evidence_tier=EvidenceTier.TIER_3_EARLY_CLINICAL,
            administration_routes=[AdministrationRoute.INTRANASAL],
            description="Synthetic ACTH analog approved in some countries for cognitive/neurological conditions.",
        ),
        PeptideNode(
            canonical_name="Selank",
            synonyms=["TP-7"],
            peptide_class=PeptideClass.CNS_NEUROTROPHIC,
            subclass="Tuftsin analog",
            regulatory_status=RegulatoryStatus.APPROVED,  # Approved in Russia
            evidence_tier=EvidenceTier.TIER_3_EARLY_CLINICAL,
            administration_routes=[AdministrationRoute.INTRANASAL],
            description="Synthetic tuftsin analog with anxiolytic and nootropic properties. Approved in Russia.",
        ),
        PeptideNode(
            canonical_name="Dihexa",
            synonyms=["N-hexanoic-Tyr-Ile-(6)aminohexanoic amide"],
            peptide_class=PeptideClass.CNS_NEUROTROPHIC,
            subclass="Angiotensin IV analog",
            regulatory_status=RegulatoryStatus.RESEARCH_ONLY,
            evidence_tier=EvidenceTier.TIER_4_PRECLINICAL,
            administration_routes=[AdministrationRoute.ORAL, AdministrationRoute.INJECTABLE_SC],
            description="Synthetic angiotensin IV analog. Preclinical research on cognitive enhancement.",
        ),
        PeptideNode(
            canonical_name="P21",
            synonyms=["P021"],
            peptide_class=PeptideClass.CNS_NEUROTROPHIC,
            subclass="CNTF-derived tetrapeptide",
            regulatory_status=RegulatoryStatus.RESEARCH_ONLY,
            evidence_tier=EvidenceTier.TIER_4_PRECLINICAL,
            administration_routes=[AdministrationRoute.INJECTABLE_SC],
            description="Ciliary neurotrophic factor-derived peptide. Preclinical neurogenesis research.",
        ),
        PeptideNode(
            canonical_name="Cerebrolysin",
            synonyms=["FPF 1070"],
            peptide_class=PeptideClass.CNS_NEUROTROPHIC,
            subclass="Porcine brain-derived peptide mixture",
            regulatory_status=RegulatoryStatus.APPROVED,  # Approved in some countries
            evidence_tier=EvidenceTier.TIER_2_LATE_CLINICAL,
            administration_routes=[AdministrationRoute.INJECTABLE_IV, AdministrationRoute.INJECTABLE_IM],
            description="Low molecular weight peptide preparation from porcine brain. Approved in some countries for stroke/dementia.",
        ),
    ])
    
    # =========================================================================
    # LONGEVITY / CELLULAR RESILIENCE
    # =========================================================================
    
    peptides.extend([
        PeptideNode(
            canonical_name="Epithalon",
            synonyms=["Epitalon", "Epithalone", "AEDG peptide"],
            peptide_class=PeptideClass.LONGEVITY_CELLULAR,
            subclass="Synthetic pineal tetrapeptide",
            regulatory_status=RegulatoryStatus.RESEARCH_ONLY,
            evidence_tier=EvidenceTier.TIER_4_PRECLINICAL,
            administration_routes=[AdministrationRoute.INJECTABLE_SC],
            description="Synthetic tetrapeptide based on epithalamin. Telomerase activation research.",
        ),
        PeptideNode(
            canonical_name="FOXO4-DRI",
            synonyms=["FOXO4 D-retro-inverso peptide"],
            peptide_class=PeptideClass.LONGEVITY_CELLULAR,
            subclass="Senolytic peptide",
            regulatory_status=RegulatoryStatus.RESEARCH_ONLY,
            evidence_tier=EvidenceTier.TIER_4_PRECLINICAL,
            administration_routes=[AdministrationRoute.INJECTABLE_SC],
            description="D-retro-inverso peptide targeting FOXO4-p53 interaction. Senescent cell research.",
        ),
        PeptideNode(
            canonical_name="SS-31",
            synonyms=["Elamipretide", "Bendavia", "MTP-131"],
            peptide_class=PeptideClass.LONGEVITY_CELLULAR,
            subclass="Mitochondria-targeted tetrapeptide",
            regulatory_status=RegulatoryStatus.INVESTIGATIONAL,
            evidence_tier=EvidenceTier.TIER_2_LATE_CLINICAL,
            administration_routes=[AdministrationRoute.INJECTABLE_SC, AdministrationRoute.INJECTABLE_IV],
            description="Mitochondria-targeted peptide. Phase 2/3 trials for mitochondrial diseases, heart failure.",
        ),
        PeptideNode(
            canonical_name="MOTS-c",
            synonyms=["Mitochondrial ORF of the 12S rRNA type-c"],
            peptide_class=PeptideClass.LONGEVITY_CELLULAR,
            subclass="Mitochondrial-derived peptide",
            regulatory_status=RegulatoryStatus.RESEARCH_ONLY,
            evidence_tier=EvidenceTier.TIER_4_PRECLINICAL,
            administration_routes=[AdministrationRoute.INJECTABLE_SC],
            description="Mitochondrial-derived peptide. Preclinical metabolic and aging research.",
        ),
        PeptideNode(
            canonical_name="Humanin",
            synonyms=["HN", "HNG"],
            peptide_class=PeptideClass.LONGEVITY_CELLULAR,
            subclass="Mitochondrial-derived peptide",
            regulatory_status=RegulatoryStatus.RESEARCH_ONLY,
            evidence_tier=EvidenceTier.TIER_4_PRECLINICAL,
            administration_routes=[AdministrationRoute.INJECTABLE_SC],
            description="Mitochondrial-derived cytoprotective peptide. Preclinical neuroprotection research.",
        ),
    ])
    
    # =========================================================================
    # METABOLIC / INCRETIN-ADJACENT (Non-GLP-1 focus)
    # =========================================================================
    
    peptides.extend([
        PeptideNode(
            canonical_name="Pramlintide",
            synonyms=["Symlin", "Amylin analog"],
            peptide_class=PeptideClass.METABOLIC_INCRETIN,
            subclass="Amylin analog",
            regulatory_status=RegulatoryStatus.APPROVED,
            evidence_tier=EvidenceTier.TIER_1_APPROVED,
            administration_routes=[AdministrationRoute.INJECTABLE_SC],
            description="Synthetic amylin analog approved for diabetes alongside insulin.",
            drugbank_id="DB01278",
        ),
        PeptideNode(
            canonical_name="Oxyntomodulin",
            synonyms=["OXM"],
            peptide_class=PeptideClass.METABOLIC_INCRETIN,
            subclass="Dual GLP-1/glucagon receptor agonist",
            regulatory_status=RegulatoryStatus.INVESTIGATIONAL,
            evidence_tier=EvidenceTier.TIER_3_EARLY_CLINICAL,
            administration_routes=[AdministrationRoute.INJECTABLE_SC],
            description="Endogenous peptide with dual receptor activity. Investigated for obesity/diabetes.",
        ),
    ])
    
    # =========================================================================
    # GLP-1 REFERENCE (Not central focus - included for context only)
    # =========================================================================
    
    peptides.extend([
        PeptideNode(
            canonical_name="Semaglutide",
            synonyms=["Ozempic", "Wegovy", "Rybelsus"],
            peptide_class=PeptideClass.GLP1_REFERENCE,
            subclass="GLP-1 receptor agonist",
            regulatory_status=RegulatoryStatus.APPROVED,
            evidence_tier=EvidenceTier.TIER_1_APPROVED,
            administration_routes=[AdministrationRoute.INJECTABLE_SC, AdministrationRoute.ORAL],
            description="GLP-1 agonist. REFERENCE ONLY - not central focus of this Atlas.",
            drugbank_id="DB13928",
            notes="Included as reference landmark. See GLP-1 literature for comprehensive coverage.",
        ),
        PeptideNode(
            canonical_name="Tirzepatide",
            synonyms=["Mounjaro", "Zepbound"],
            peptide_class=PeptideClass.GLP1_REFERENCE,
            subclass="Dual GLP-1/GIP receptor agonist",
            regulatory_status=RegulatoryStatus.APPROVED,
            evidence_tier=EvidenceTier.TIER_1_APPROVED,
            administration_routes=[AdministrationRoute.INJECTABLE_SC],
            description="Dual GLP-1/GIP agonist. REFERENCE ONLY - not central focus of this Atlas.",
            drugbank_id="DB15171",
            notes="Included as reference landmark. See incretin literature for comprehensive coverage.",
        ),
    ])
    
    return peptides


def get_peptide_count_by_class() -> dict[PeptideClass, int]:
    """Returns count of peptides per class."""
    peptides = get_curated_peptides()
    counts: dict[PeptideClass, int] = {}
    for p in peptides:
        counts[p.peptide_class] = counts.get(p.peptide_class, 0) + 1
    return counts


def get_peptide_count_by_evidence_tier() -> dict[EvidenceTier, int]:
    """Returns count of peptides per evidence tier."""
    peptides = get_curated_peptides()
    counts: dict[EvidenceTier, int] = {}
    for p in peptides:
        counts[p.evidence_tier] = counts.get(p.evidence_tier, 0) + 1
    return counts

