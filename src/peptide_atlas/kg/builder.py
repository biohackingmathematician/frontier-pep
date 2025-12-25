"""
Knowledge Graph builder for the Peptide Atlas.

Constructs a heterogeneous graph from curated peptide data,
adding targets, pathways, effects, and risks with appropriate relationships.

REMINDER: This is for research and education only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

import networkx as nx
from loguru import logger

from peptide_atlas.constants import (
    BindingType,
    Confidence,
    EffectDirection,
    EffectDomain,
    EvidenceTier,
    PathwayCategory,
    RiskCategory,
    RiskSeverity,
    TargetType,
)
from peptide_atlas.data.peptide_catalog import get_curated_peptides
from peptide_atlas.data.schemas import (
    AssociatedWithEffectEdge,
    AssociatedWithRiskEdge,
    BindsEdge,
    EffectDomainNode,
    KnowledgeGraph,
    ModulatesEdge,
    PathwayNode,
    PeptideNode,
    RiskNode,
    TargetNode,
)


class KnowledgeGraphBuilder:
    """Builds the peptide knowledge graph from curated data."""
    
    def __init__(self):
        self.kg = KnowledgeGraph()
        self._target_cache: dict[str, UUID] = {}
        self._pathway_cache: dict[str, UUID] = {}
        self._effect_cache: dict[EffectDomain, UUID] = {}
        self._risk_cache: dict[str, UUID] = {}
    
    def build(self) -> KnowledgeGraph:
        """Build the complete knowledge graph."""
        logger.info("Building Peptide Atlas Knowledge Graph...")
        
        # Step 1: Add all peptides
        peptides = get_curated_peptides()
        self.kg.peptides = peptides
        logger.info(f"Added {len(peptides)} peptides")
        
        # Step 2: Add core targets
        self._add_core_targets()
        logger.info(f"Added {len(self.kg.targets)} targets")
        
        # Step 3: Add core pathways
        self._add_core_pathways()
        logger.info(f"Added {len(self.kg.pathways)} pathways")
        
        # Step 4: Add effect domains
        self._add_effect_domains()
        logger.info(f"Added {len(self.kg.effect_domains)} effect domains")
        
        # Step 5: Add risks
        self._add_core_risks()
        logger.info(f"Added {len(self.kg.risks)} risks")
        
        # Step 6: Add relationships
        self._add_peptide_relationships()
        logger.info(
            f"Added relationships: "
            f"{len(self.kg.binds_edges)} binds, "
            f"{len(self.kg.modulates_edges)} modulates, "
            f"{len(self.kg.effect_edges)} effects, "
            f"{len(self.kg.risk_edges)} risks"
        )
        
        logger.info(
            f"Knowledge Graph complete: "
            f"{self.kg.node_count} nodes, {self.kg.edge_count} edges"
        )
        
        return self.kg
    
    def _add_core_targets(self) -> None:
        """Add core molecular targets."""
        targets_data = [
            # Receptors
            ("GHRHR", TargetType.RECEPTOR, "Growth Hormone Releasing Hormone Receptor"),
            ("GHSR", TargetType.RECEPTOR, "Growth Hormone Secretagogue Receptor (Ghrelin Receptor)"),
            ("IGF1R", TargetType.RECEPTOR, "Insulin-like Growth Factor 1 Receptor"),
            ("INSR", TargetType.RECEPTOR, "Insulin Receptor"),
            ("GLP1R", TargetType.RECEPTOR, "Glucagon-like Peptide-1 Receptor"),
            ("GIPR", TargetType.RECEPTOR, "Gastric Inhibitory Polypeptide Receptor"),
            ("CALCR", TargetType.RECEPTOR, "Calcitonin Receptor (Amylin)"),
            ("AT4", TargetType.RECEPTOR, "Angiotensin IV Receptor"),
            
            # Tissues
            ("Pituitary", TargetType.ORGAN, "Pituitary gland"),
            ("Liver", TargetType.ORGAN, "Liver"),
            ("Skeletal Muscle", TargetType.TISSUE, "Skeletal muscle tissue"),
            ("Thymus", TargetType.ORGAN, "Thymus gland"),
            ("CNS", TargetType.ORGAN, "Central Nervous System"),
            ("Bone", TargetType.TISSUE, "Bone tissue"),
            ("Tendon", TargetType.TISSUE, "Tendon tissue"),
            ("Cartilage", TargetType.TISSUE, "Cartilage tissue"),
            ("Skin", TargetType.TISSUE, "Skin tissue"),
            ("Vasculature", TargetType.TISSUE, "Blood vessels"),
            ("Mitochondria", TargetType.OTHER, "Mitochondria"),
            ("Immune Cells", TargetType.CELL_TYPE, "Immune cells (T-cells, NK cells, etc.)"),
        ]
        
        for name, ttype, desc in targets_data:
            target = TargetNode(
                id=uuid4(),
                name=name,
                target_type=ttype,
                description=desc,
            )
            self.kg.targets.append(target)
            self._target_cache[name] = target.id
    
    def _add_core_pathways(self) -> None:
        """Add core signaling pathways."""
        pathways_data = [
            ("GH/IGF-1 Axis", PathwayCategory.GROWTH_ANABOLIC, 
             "Growth hormone and IGF-1 signaling cascade"),
            ("PI3K-Akt-mTOR", PathwayCategory.GROWTH_ANABOLIC,
             "Phosphoinositide 3-kinase/Akt/mTOR pathway"),
            ("JAK-STAT", PathwayCategory.GROWTH_ANABOLIC,
             "Janus kinase/Signal transducer and activator of transcription"),
            ("MAPK/ERK", PathwayCategory.GROWTH_ANABOLIC,
             "Mitogen-activated protein kinase / Extracellular signal-regulated kinase"),
            ("Ghrelin Signaling", PathwayCategory.METABOLIC,
             "Ghrelin receptor signaling pathway"),
            ("Incretin Signaling", PathwayCategory.METABOLIC,
             "GLP-1 and GIP receptor signaling"),
            ("ECM Remodeling", PathwayCategory.REPAIR_REGENERATION,
             "Extracellular matrix remodeling"),
            ("Angiogenesis", PathwayCategory.REPAIR_REGENERATION,
             "Blood vessel formation"),
            ("Wound Healing", PathwayCategory.REPAIR_REGENERATION,
             "Wound healing and tissue repair"),
            ("T-Cell Maturation", PathwayCategory.IMMUNE,
             "Thymic T-cell development and maturation"),
            ("Innate Immunity", PathwayCategory.IMMUNE,
             "Innate immune response pathways"),
            ("BDNF/TrkB", PathwayCategory.NEUROLOGICAL,
             "Brain-derived neurotrophic factor signaling"),
            ("Neuroprotection", PathwayCategory.NEUROLOGICAL,
             "Neuronal survival and protection pathways"),
            ("Autophagy", PathwayCategory.AGING_LONGEVITY,
             "Cellular autophagy pathway"),
            ("Telomerase", PathwayCategory.AGING_LONGEVITY,
             "Telomerase activity and telomere maintenance"),
            ("Mitochondrial Function", PathwayCategory.AGING_LONGEVITY,
             "Mitochondrial biogenesis and function"),
            ("Senescence", PathwayCategory.AGING_LONGEVITY,
             "Cellular senescence pathways"),
        ]
        
        for name, category, desc in pathways_data:
            pathway = PathwayNode(
                id=uuid4(),
                name=name,
                category=category,
                description=desc,
            )
            self.kg.pathways.append(pathway)
            self._pathway_cache[name] = pathway.id
    
    def _add_effect_domains(self) -> None:
        """Add all effect domains."""
        for effect in EffectDomain:
            node = EffectDomainNode(
                id=uuid4(),
                name=effect.value.replace("_", " ").title(),
                category=effect,
            )
            self.kg.effect_domains.append(node)
            self._effect_cache[effect] = node.id
    
    def _add_core_risks(self) -> None:
        """Add core risk categories."""
        risks_data = [
            ("Fluid Retention / Edema", RiskCategory.FLUID_RETENTION, RiskSeverity.MILD),
            ("Carpal Tunnel-like Symptoms", RiskCategory.MUSCULOSKELETAL, RiskSeverity.MILD),
            ("Joint Pain", RiskCategory.MUSCULOSKELETAL, RiskSeverity.MILD),
            ("Insulin Resistance", RiskCategory.METABOLIC, RiskSeverity.MODERATE),
            ("Hypoglycemia Risk", RiskCategory.METABOLIC, RiskSeverity.MODERATE),
            ("Theoretical Proliferative Risk", RiskCategory.GROWTH_PROLIFERATIVE, RiskSeverity.UNKNOWN),
            ("HPA Axis Suppression", RiskCategory.ENDOCRINE_SUPPRESSION, RiskSeverity.MODERATE),
            ("Increased Appetite", RiskCategory.METABOLIC, RiskSeverity.MILD),
            ("Nausea", RiskCategory.GASTROINTESTINAL, RiskSeverity.MILD),
            ("Injection Site Reactions", RiskCategory.INJECTION_SITE, RiskSeverity.MILD),
            ("Unknown Long-term Effects", RiskCategory.UNKNOWN_LONG_TERM, RiskSeverity.UNKNOWN),
            ("Immune Overactivation", RiskCategory.IMMUNE_OVERACTIVATION, RiskSeverity.MODERATE),
            ("Blood Pressure Changes", RiskCategory.CARDIOVASCULAR, RiskSeverity.MODERATE),
            ("Mood/Anxiety Effects", RiskCategory.CNS_PSYCHIATRIC, RiskSeverity.MILD),
        ]
        
        for name, category, severity in risks_data:
            risk = RiskNode(
                id=uuid4(),
                name=name,
                category=category,
                severity_typical=severity,
            )
            self.kg.risks.append(risk)
            self._risk_cache[name] = risk.id
    
    def _get_target_id(self, name: str) -> Optional[UUID]:
        return self._target_cache.get(name)
    
    def _get_pathway_id(self, name: str) -> Optional[UUID]:
        return self._pathway_cache.get(name)
    
    def _get_effect_id(self, effect: EffectDomain) -> Optional[UUID]:
        return self._effect_cache.get(effect)
    
    def _get_risk_id(self, name: str) -> Optional[UUID]:
        return self._risk_cache.get(name)
    
    def _add_peptide_relationships(self) -> None:
        """Add relationships for each peptide based on class and known mechanisms."""
        
        for peptide in self.kg.peptides:
            self._add_relationships_for_peptide(peptide)
    
    def _add_relationships_for_peptide(self, peptide: PeptideNode) -> None:
        """Add class-specific relationships for a peptide."""
        
        # GHRH Analogs
        if peptide.peptide_class.value == "ghrh_analog":
            self._add_ghrh_analog_relationships(peptide)
        
        # GH Secretagogues
        elif peptide.peptide_class.value == "ghs_ghrelin_mimetic":
            self._add_ghs_relationships(peptide)
        
        # IGF Axis
        elif peptide.peptide_class.value == "igf_axis":
            self._add_igf_relationships(peptide)
        
        # Regenerative/Repair
        elif peptide.peptide_class.value == "regenerative_repair":
            self._add_regenerative_relationships(peptide)
        
        # Thymic/Immune
        elif peptide.peptide_class.value == "thymic_immune":
            self._add_immune_relationships(peptide)
        
        # CNS/Neurotrophic
        elif peptide.peptide_class.value == "cns_neurotrophic":
            self._add_cns_relationships(peptide)
        
        # Longevity
        elif peptide.peptide_class.value == "longevity_cellular":
            self._add_longevity_relationships(peptide)
        
        # Metabolic
        elif peptide.peptide_class.value == "metabolic_incretin":
            self._add_metabolic_relationships(peptide)
        
        # GLP-1 Reference
        elif peptide.peptide_class.value == "glp1_reference":
            self._add_glp1_reference_relationships(peptide)
    
    def _add_ghrh_analog_relationships(self, peptide: PeptideNode) -> None:
        """Add relationships for GHRH analogs."""
        # Binds GHRHR
        if ghrhr_id := self._get_target_id("GHRHR"):
            self.kg.binds_edges.append(BindsEdge(
                source_id=peptide.id,
                target_id=ghrhr_id,
                binding_type=BindingType.AGONIST,
                confidence=Confidence.HIGH,
            ))
        
        # Targets pituitary
        if pit_id := self._get_target_id("Pituitary"):
            self.kg.binds_edges.append(BindsEdge(
                source_id=peptide.id,
                target_id=pit_id,
                binding_type=BindingType.AGONIST,
                confidence=Confidence.HIGH,
            ))
        
        # Modulates GH/IGF-1 axis
        if ghigf_id := self._get_pathway_id("GH/IGF-1 Axis"):
            self.kg.modulates_edges.append(ModulatesEdge(
                source_id=peptide.id,
                target_id=ghigf_id,
                direction="activates",
                magnitude="strong",
                confidence=Confidence.HIGH,
            ))
        
        # Effects
        for effect in [EffectDomain.ANABOLIC_BODY_COMPOSITION, EffectDomain.RECOVERY]:
            if effect_id := self._get_effect_id(effect):
                self.kg.effect_edges.append(AssociatedWithEffectEdge(
                    source_id=peptide.id,
                    target_id=effect_id,
                    direction=EffectDirection.BENEFICIAL,
                    evidence_tier=peptide.evidence_tier,
                    confidence=Confidence.MEDIUM,
                ))
        
        # Risks
        for risk_name in ["Fluid Retention / Edema", "Joint Pain", "Unknown Long-term Effects"]:
            if risk_id := self._get_risk_id(risk_name):
                self.kg.risk_edges.append(AssociatedWithRiskEdge(
                    source_id=peptide.id,
                    target_id=risk_id,
                    evidence_tier=peptide.evidence_tier,
                    confidence=Confidence.MEDIUM,
                ))
    
    def _add_ghs_relationships(self, peptide: PeptideNode) -> None:
        """Add relationships for GH secretagogues."""
        # Binds GHSR
        if ghsr_id := self._get_target_id("GHSR"):
            self.kg.binds_edges.append(BindsEdge(
                source_id=peptide.id,
                target_id=ghsr_id,
                binding_type=BindingType.AGONIST,
                confidence=Confidence.HIGH,
            ))
        
        # Modulates Ghrelin Signaling
        if ghrelin_id := self._get_pathway_id("Ghrelin Signaling"):
            self.kg.modulates_edges.append(ModulatesEdge(
                source_id=peptide.id,
                target_id=ghrelin_id,
                direction="activates",
                magnitude="strong",
                confidence=Confidence.HIGH,
            ))
        
        # Modulates GH/IGF-1 axis
        if ghigf_id := self._get_pathway_id("GH/IGF-1 Axis"):
            self.kg.modulates_edges.append(ModulatesEdge(
                source_id=peptide.id,
                target_id=ghigf_id,
                direction="activates",
                magnitude="moderate",
                confidence=Confidence.HIGH,
            ))
        
        # Effects
        for effect in [EffectDomain.ANABOLIC_BODY_COMPOSITION, EffectDomain.SLEEP, EffectDomain.RECOVERY]:
            if effect_id := self._get_effect_id(effect):
                self.kg.effect_edges.append(AssociatedWithEffectEdge(
                    source_id=peptide.id,
                    target_id=effect_id,
                    direction=EffectDirection.BENEFICIAL,
                    evidence_tier=peptide.evidence_tier,
                    confidence=Confidence.MEDIUM,
                ))
        
        # Risks
        for risk_name in ["Increased Appetite", "Fluid Retention / Edema", 
                          "Insulin Resistance", "Unknown Long-term Effects"]:
            if risk_id := self._get_risk_id(risk_name):
                self.kg.risk_edges.append(AssociatedWithRiskEdge(
                    source_id=peptide.id,
                    target_id=risk_id,
                    evidence_tier=peptide.evidence_tier,
                    confidence=Confidence.MEDIUM,
                ))
    
    def _add_igf_relationships(self, peptide: PeptideNode) -> None:
        """Add relationships for IGF-axis peptides."""
        # Binds IGF1R
        if igf1r_id := self._get_target_id("IGF1R"):
            self.kg.binds_edges.append(BindsEdge(
                source_id=peptide.id,
                target_id=igf1r_id,
                binding_type=BindingType.AGONIST,
                confidence=Confidence.HIGH,
            ))
        
        # Modulates PI3K-Akt-mTOR
        if mtor_id := self._get_pathway_id("PI3K-Akt-mTOR"):
            self.kg.modulates_edges.append(ModulatesEdge(
                source_id=peptide.id,
                target_id=mtor_id,
                direction="activates",
                magnitude="strong",
                confidence=Confidence.HIGH,
            ))
        
        # Effects
        for effect in [EffectDomain.ANABOLIC_BODY_COMPOSITION, 
                       EffectDomain.PERFORMANCE_STRENGTH,
                       EffectDomain.REGENERATION_REPAIR]:
            if effect_id := self._get_effect_id(effect):
                self.kg.effect_edges.append(AssociatedWithEffectEdge(
                    source_id=peptide.id,
                    target_id=effect_id,
                    direction=EffectDirection.BENEFICIAL,
                    evidence_tier=peptide.evidence_tier,
                    confidence=Confidence.MEDIUM,
                ))
        
        # Risks
        for risk_name in ["Hypoglycemia Risk", "Theoretical Proliferative Risk", 
                          "Unknown Long-term Effects"]:
            if risk_id := self._get_risk_id(risk_name):
                self.kg.risk_edges.append(AssociatedWithRiskEdge(
                    source_id=peptide.id,
                    target_id=risk_id,
                    evidence_tier=peptide.evidence_tier,
                    confidence=Confidence.MEDIUM,
                ))
    
    def _add_regenerative_relationships(self, peptide: PeptideNode) -> None:
        """Add relationships for regenerative peptides."""
        # Target tissues
        for tissue in ["Tendon", "Cartilage", "Skeletal Muscle", "Skin", "Vasculature"]:
            if tissue_id := self._get_target_id(tissue):
                self.kg.binds_edges.append(BindsEdge(
                    source_id=peptide.id,
                    target_id=tissue_id,
                    binding_type=BindingType.AGONIST,
                    confidence=Confidence.MEDIUM,
                ))
        
        # Modulates wound healing / ECM
        for pathway in ["ECM Remodeling", "Wound Healing", "Angiogenesis"]:
            if pathway_id := self._get_pathway_id(pathway):
                self.kg.modulates_edges.append(ModulatesEdge(
                    source_id=peptide.id,
                    target_id=pathway_id,
                    direction="activates",
                    magnitude="moderate",
                    confidence=Confidence.MEDIUM,
                ))
        
        # Effects
        for effect in [EffectDomain.REGENERATION_REPAIR, EffectDomain.RECOVERY]:
            if effect_id := self._get_effect_id(effect):
                self.kg.effect_edges.append(AssociatedWithEffectEdge(
                    source_id=peptide.id,
                    target_id=effect_id,
                    direction=EffectDirection.BENEFICIAL,
                    evidence_tier=peptide.evidence_tier,
                    confidence=Confidence.MEDIUM,
                ))
        
        # Risks (mostly unknown for these)
        if risk_id := self._get_risk_id("Unknown Long-term Effects"):
            self.kg.risk_edges.append(AssociatedWithRiskEdge(
                source_id=peptide.id,
                target_id=risk_id,
                evidence_tier=peptide.evidence_tier,
                confidence=Confidence.HIGH,
            ))
    
    def _add_immune_relationships(self, peptide: PeptideNode) -> None:
        """Add relationships for immune/thymic peptides."""
        # Target thymus and immune cells
        for target in ["Thymus", "Immune Cells"]:
            if target_id := self._get_target_id(target):
                self.kg.binds_edges.append(BindsEdge(
                    source_id=peptide.id,
                    target_id=target_id,
                    binding_type=BindingType.AGONIST,
                    confidence=Confidence.MEDIUM,
                ))
        
        # Modulates immune pathways
        for pathway in ["T-Cell Maturation", "Innate Immunity"]:
            if pathway_id := self._get_pathway_id(pathway):
                self.kg.modulates_edges.append(ModulatesEdge(
                    source_id=peptide.id,
                    target_id=pathway_id,
                    direction="modulates",
                    magnitude="moderate",
                    confidence=Confidence.MEDIUM,
                ))
        
        # Effects
        if effect_id := self._get_effect_id(EffectDomain.IMMUNE_RESILIENCE):
            self.kg.effect_edges.append(AssociatedWithEffectEdge(
                source_id=peptide.id,
                target_id=effect_id,
                direction=EffectDirection.BENEFICIAL,
                evidence_tier=peptide.evidence_tier,
                confidence=Confidence.MEDIUM,
            ))
        
        # Risks
        for risk_name in ["Immune Overactivation", "Unknown Long-term Effects"]:
            if risk_id := self._get_risk_id(risk_name):
                self.kg.risk_edges.append(AssociatedWithRiskEdge(
                    source_id=peptide.id,
                    target_id=risk_id,
                    evidence_tier=peptide.evidence_tier,
                    confidence=Confidence.MEDIUM,
                ))
    
    def _add_cns_relationships(self, peptide: PeptideNode) -> None:
        """Add relationships for CNS/neurotrophic peptides."""
        # Target CNS
        if cns_id := self._get_target_id("CNS"):
            self.kg.binds_edges.append(BindsEdge(
                source_id=peptide.id,
                target_id=cns_id,
                binding_type=BindingType.AGONIST,
                confidence=Confidence.MEDIUM,
            ))
        
        # Modulates neurotrophic pathways
        for pathway in ["BDNF/TrkB", "Neuroprotection"]:
            if pathway_id := self._get_pathway_id(pathway):
                self.kg.modulates_edges.append(ModulatesEdge(
                    source_id=peptide.id,
                    target_id=pathway_id,
                    direction="activates",
                    magnitude="moderate",
                    confidence=Confidence.MEDIUM,
                ))
        
        # Effects
        for effect in [EffectDomain.CNS_COGNITIVE, EffectDomain.CNS_NEUROPROTECTION,
                       EffectDomain.CNS_MOOD]:
            if effect_id := self._get_effect_id(effect):
                self.kg.effect_edges.append(AssociatedWithEffectEdge(
                    source_id=peptide.id,
                    target_id=effect_id,
                    direction=EffectDirection.BENEFICIAL,
                    evidence_tier=peptide.evidence_tier,
                    confidence=Confidence.LOW,
                ))
        
        # Risks
        for risk_name in ["Mood/Anxiety Effects", "Unknown Long-term Effects"]:
            if risk_id := self._get_risk_id(risk_name):
                self.kg.risk_edges.append(AssociatedWithRiskEdge(
                    source_id=peptide.id,
                    target_id=risk_id,
                    evidence_tier=peptide.evidence_tier,
                    confidence=Confidence.MEDIUM,
                ))
    
    def _add_longevity_relationships(self, peptide: PeptideNode) -> None:
        """Add relationships for longevity peptides."""
        # Target mitochondria
        if mito_id := self._get_target_id("Mitochondria"):
            self.kg.binds_edges.append(BindsEdge(
                source_id=peptide.id,
                target_id=mito_id,
                binding_type=BindingType.AGONIST,
                confidence=Confidence.MEDIUM,
            ))
        
        # Modulates longevity pathways
        for pathway in ["Autophagy", "Telomerase", "Mitochondrial Function", "Senescence"]:
            if pathway_id := self._get_pathway_id(pathway):
                self.kg.modulates_edges.append(ModulatesEdge(
                    source_id=peptide.id,
                    target_id=pathway_id,
                    direction="modulates",
                    magnitude="unknown",
                    confidence=Confidence.LOW,
                ))
        
        # Effects
        if effect_id := self._get_effect_id(EffectDomain.LONGEVITY_AGING):
            self.kg.effect_edges.append(AssociatedWithEffectEdge(
                source_id=peptide.id,
                target_id=effect_id,
                direction=EffectDirection.AMBIGUOUS,  # Often speculative
                evidence_tier=peptide.evidence_tier,
                confidence=Confidence.LOW,
            ))
        
        # Risks
        if risk_id := self._get_risk_id("Unknown Long-term Effects"):
            self.kg.risk_edges.append(AssociatedWithRiskEdge(
                source_id=peptide.id,
                target_id=risk_id,
                evidence_tier=peptide.evidence_tier,
                confidence=Confidence.HIGH,
            ))
    
    def _add_metabolic_relationships(self, peptide: PeptideNode) -> None:
        """Add relationships for metabolic peptides."""
        # Binds relevant receptors
        if calcr_id := self._get_target_id("CALCR"):
            self.kg.binds_edges.append(BindsEdge(
                source_id=peptide.id,
                target_id=calcr_id,
                binding_type=BindingType.AGONIST,
                confidence=Confidence.HIGH,
            ))
        
        # Effects
        for effect in [EffectDomain.METABOLIC_GLYCEMIC, EffectDomain.ANABOLIC_BODY_COMPOSITION]:
            if effect_id := self._get_effect_id(effect):
                self.kg.effect_edges.append(AssociatedWithEffectEdge(
                    source_id=peptide.id,
                    target_id=effect_id,
                    direction=EffectDirection.BENEFICIAL,
                    evidence_tier=peptide.evidence_tier,
                    confidence=Confidence.MEDIUM,
                ))
        
        # Risks
        for risk_name in ["Nausea", "Hypoglycemia Risk"]:
            if risk_id := self._get_risk_id(risk_name):
                self.kg.risk_edges.append(AssociatedWithRiskEdge(
                    source_id=peptide.id,
                    target_id=risk_id,
                    evidence_tier=peptide.evidence_tier,
                    confidence=Confidence.MEDIUM,
                ))
    
    def _add_glp1_reference_relationships(self, peptide: PeptideNode) -> None:
        """Add relationships for GLP-1 reference peptides."""
        # Binds GLP1R
        if glp1r_id := self._get_target_id("GLP1R"):
            self.kg.binds_edges.append(BindsEdge(
                source_id=peptide.id,
                target_id=glp1r_id,
                binding_type=BindingType.AGONIST,
                confidence=Confidence.HIGH,
            ))
        
        # Modulates incretin signaling
        if incretin_id := self._get_pathway_id("Incretin Signaling"):
            self.kg.modulates_edges.append(ModulatesEdge(
                source_id=peptide.id,
                target_id=incretin_id,
                direction="activates",
                magnitude="strong",
                confidence=Confidence.HIGH,
            ))
        
        # Effects
        for effect in [EffectDomain.METABOLIC_GLYCEMIC, EffectDomain.ANABOLIC_BODY_COMPOSITION]:
            if effect_id := self._get_effect_id(effect):
                self.kg.effect_edges.append(AssociatedWithEffectEdge(
                    source_id=peptide.id,
                    target_id=effect_id,
                    direction=EffectDirection.BENEFICIAL,
                    evidence_tier=peptide.evidence_tier,
                    confidence=Confidence.HIGH,
                ))
        
        # Risks
        for risk_name in ["Nausea", "Injection Site Reactions"]:
            if risk_id := self._get_risk_id(risk_name):
                self.kg.risk_edges.append(AssociatedWithRiskEdge(
                    source_id=peptide.id,
                    target_id=risk_id,
                    evidence_tier=peptide.evidence_tier,
                    confidence=Confidence.HIGH,
                ))
    
    def to_networkx(self) -> nx.MultiDiGraph:
        """Convert KnowledgeGraph to NetworkX MultiDiGraph."""
        G = nx.MultiDiGraph()
        
        # Add nodes
        for peptide in self.kg.peptides:
            G.add_node(
                str(peptide.id),
                node_type="peptide",
                name=peptide.canonical_name,
                peptide_class=peptide.peptide_class.value,
                evidence_tier=peptide.evidence_tier.value,
            )
        
        for target in self.kg.targets:
            G.add_node(
                str(target.id),
                node_type="target",
                name=target.name,
                target_type=target.target_type.value,
            )
        
        for pathway in self.kg.pathways:
            G.add_node(
                str(pathway.id),
                node_type="pathway",
                name=pathway.name,
                category=pathway.category.value,
            )
        
        for effect in self.kg.effect_domains:
            G.add_node(
                str(effect.id),
                node_type="effect_domain",
                name=effect.name,
                category=effect.category.value,
            )
        
        for risk in self.kg.risks:
            G.add_node(
                str(risk.id),
                node_type="risk",
                name=risk.name,
                category=risk.category.value,
                severity=risk.severity_typical.value,
            )
        
        # Add edges
        for edge in self.kg.binds_edges:
            G.add_edge(
                str(edge.source_id),
                str(edge.target_id),
                edge_type="binds",
                binding_type=edge.binding_type.value,
                confidence=edge.confidence.value,
            )
        
        for edge in self.kg.modulates_edges:
            G.add_edge(
                str(edge.source_id),
                str(edge.target_id),
                edge_type="modulates",
                direction=edge.direction,
                confidence=edge.confidence.value,
            )
        
        for edge in self.kg.effect_edges:
            G.add_edge(
                str(edge.source_id),
                str(edge.target_id),
                edge_type="associated_with_effect",
                direction=edge.direction.value,
                evidence_tier=edge.evidence_tier.value,
                confidence=edge.confidence.value,
            )
        
        for edge in self.kg.risk_edges:
            G.add_edge(
                str(edge.source_id),
                str(edge.target_id),
                edge_type="associated_with_risk",
                evidence_tier=edge.evidence_tier.value,
                confidence=edge.confidence.value,
            )
        
        return G
    
    def save(self, path: Path) -> None:
        """Save knowledge graph to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.kg.model_dump(mode="json"), f, indent=2, default=str)
        logger.info(f"Saved knowledge graph to {path}")
    
    @classmethod
    def load(cls, path: Path) -> KnowledgeGraph:
        """Load knowledge graph from JSON."""
        with open(path) as f:
            data = json.load(f)
        return KnowledgeGraph.model_validate(data)


def build_knowledge_graph() -> KnowledgeGraph:
    """Main entry point for building the knowledge graph."""
    builder = KnowledgeGraphBuilder()
    return builder.build()

