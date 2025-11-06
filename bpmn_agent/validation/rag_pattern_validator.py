"""
RAG Pattern Validator for Phase 4

Validates that RAG patterns applied during generation are correctly reflected
in the generated BPMN XML.
"""

import logging
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field

from bpmn_agent.models.graph import ProcessGraph, GraphNode, GraphEdge
from bpmn_agent.models.knowledge_base import BPMNPattern, KnowledgeBase, DomainType

logger = logging.getLogger(__name__)


@dataclass
class PatternComplianceFinding:
    """Finding from pattern compliance validation."""
    pattern_id: str
    pattern_name: str
    structure_compliance: float  # 0.0 - 1.0
    element_compliance: float
    relation_compliance: float
    overall_score: float
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class RAGValidationResult:
    """Result of RAG pattern validation."""
    findings: List[PatternComplianceFinding] = field(default_factory=list)
    overall_compliance_score: float = 1.0
    patterns_validated: int = 0
    patterns_passed: int = 0


class RAGPatternValidator:
    """
    Validates that applied RAG patterns are correctly reflected in XML.
    
    Follows Strategy Pattern (like BaseLLMClient in project).
    Uses Dependency Injection for testability.
    Implements Graceful Degradation (works without KB).
    """
    
    def __init__(
        self,
        kb: Optional[KnowledgeBase] = None,
        pattern_bridge: Optional[object] = None
    ):
        """
        Initialize RAG pattern validator.
        
        Args:
            kb: Knowledge base (optional, will load default if None)
            pattern_bridge: Pattern matching bridge (optional)
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Graceful degradation: funciona sin KB
        try:
            if kb is None:
                from bpmn_agent.knowledge.loader import PatternLibraryLoader
                loader = PatternLibraryLoader()
                self.kb = loader.load_all_patterns()
            else:
                self.kb = kb
            
            if pattern_bridge is None:
                from bpmn_agent.knowledge.pattern_matching_bridge import AdvancedPatternMatchingBridge
                self.pattern_bridge = AdvancedPatternMatchingBridge(self.kb)
            else:
                self.pattern_bridge = pattern_bridge
            
            self.enabled = True
            self.logger.info("RAGPatternValidator initialized with KB")
        except Exception as e:
            self.logger.warning(f"RAGPatternValidator initialized without KB: {e}")
            self.enabled = False
            self.kb = None
            self.pattern_bridge = None
    
    def validate_pattern_compliance(
        self,
        xml_content: str,
        patterns_applied: List[str],
        graph: Optional[ProcessGraph] = None,
        domain: Optional[str] = None
    ) -> RAGValidationResult:
        """
        Validate that applied patterns are correctly reflected.
        
        Args:
            xml_content: Generated BPMN XML
            patterns_applied: List of pattern IDs that were applied
            graph: Optional process graph for deeper analysis
            domain: Optional domain for context
            
        Returns:
            RAGValidationResult with compliance findings
        """
        if not self.enabled or not patterns_applied:
            return RAGValidationResult(
                findings=[],
                overall_compliance_score=1.0,
                patterns_validated=0,
                patterns_passed=0
            )
        
        findings = []
        
        for pattern_id in patterns_applied:
            try:
                pattern = self._get_pattern_by_id(pattern_id)
                if not pattern:
                    self.logger.warning(f"Pattern {pattern_id} not found in KB, skipping validation")
                    continue
                
                finding = self._validate_single_pattern(
                    xml_content, pattern, graph
                )
                findings.append(finding)
            except Exception as e:
                self.logger.error(f"Error validating pattern {pattern_id}: {e}")
                continue
        
        # Calculate overall score
        if findings:
            overall_score = sum(f.overall_score for f in findings) / len(findings)
            patterns_passed = sum(1 for f in findings if f.overall_score >= 0.8)
        else:
            overall_score = 1.0
            patterns_passed = 0
        
        return RAGValidationResult(
            findings=findings,
            overall_compliance_score=overall_score,
            patterns_validated=len(findings),
            patterns_passed=patterns_passed
        )
    
    def _get_pattern_by_id(self, pattern_id: str) -> Optional[BPMNPattern]:
        """Get pattern by ID from knowledge base."""
        if not self.kb:
            return None
        
        # Try direct lookup first
        pattern = self.kb.patterns.get(pattern_id)
        if pattern:
            return pattern
        
        # Try searching by name (case-insensitive)
        pattern_id_lower = pattern_id.lower()
        for pid, p in self.kb.patterns.items():
            if p.name.lower() == pattern_id_lower or pid.lower() == pattern_id_lower:
                return p
        
        return None
    
    def _validate_single_pattern(
        self,
        xml_content: str,
        pattern: BPMNPattern,
        graph: Optional[ProcessGraph]
    ) -> PatternComplianceFinding:
        """Validate a single pattern's compliance."""
        issues = []
        suggestions = []
        
        # 1. Validate structure compliance
        structure_score = self._validate_pattern_structure(xml_content, pattern, issues, suggestions)
        
        # 2. Validate element compliance
        element_score = self._validate_pattern_elements(xml_content, pattern, issues, suggestions)
        
        # 3. Validate relation compliance (if graph available)
        if graph:
            relation_score = self._validate_pattern_relations(graph, pattern, issues, suggestions)
        else:
            relation_score = 1.0  # Assume OK if no graph
        
        # Overall score (weighted average)
        overall_score = (
            structure_score * 0.4 +
            element_score * 0.4 +
            relation_score * 0.2
        )
        
        return PatternComplianceFinding(
            pattern_id=pattern.id,
            pattern_name=pattern.name,
            structure_compliance=structure_score,
            element_compliance=element_score,
            relation_compliance=relation_score,
            overall_score=overall_score,
            issues=issues,
            suggestions=suggestions
        )
    
    def _validate_pattern_structure(
        self,
        xml_content: str,
        pattern: BPMNPattern,
        issues: List[str],
        suggestions: List[str]
    ) -> float:
        """Validate pattern structure is present in XML."""
        # Check if pattern-specific elements are mentioned
        pattern_keywords = list(pattern.tags) + [pattern.name.lower()]
        
        # Add keywords from description (first 5 words)
        desc_words = pattern.description.lower().split()[:5]
        pattern_keywords.extend([w for w in desc_words if len(w) > 3])
        
        # Check XML content for keywords
        xml_lower = xml_content.lower()
        found_keywords = sum(1 for kw in pattern_keywords if kw.lower() in xml_lower)
        
        if found_keywords == 0:
            issues.append(f"Pattern '{pattern.name}' structure not found in XML")
            suggestions.append(f"Ensure pattern '{pattern.name}' elements are present")
            return 0.0
        
        # Score based on keyword coverage
        score = min(1.0, found_keywords / max(len(pattern_keywords), 1))
        
        if score < 0.5:
            issues.append(f"Pattern '{pattern.name}' has low keyword coverage ({score:.2f})")
            suggestions.append(f"Increase presence of pattern-specific elements: {', '.join(pattern_keywords[:3])}")
        
        return score
    
    def _validate_pattern_elements(
        self,
        xml_content: str,
        pattern: BPMNPattern,
        issues: List[str],
        suggestions: List[str]
    ) -> float:
        """Validate pattern-specific elements are present."""
        try:
            xml_doc = ET.fromstring(xml_content)
        except ET.ParseError as e:
            issues.append(f"XML parsing error: {e}")
            return 0.0
        
        # Check for expected BPMN elements based on pattern structure
        if pattern.graph_structure and pattern.graph_structure.nodes:
            expected_elements = set(pattern.graph_structure.nodes)
            
            # Extract actual elements from XML
            actual_elements = set()
            for elem in xml_doc.iter():
                tag_name = elem.tag.split('}')[-1]  # Remove namespace
                if tag_name in ['task', 'userTask', 'serviceTask', 'scriptTask', 
                               'exclusiveGateway', 'parallelGateway', 'inclusiveGateway',
                               'startEvent', 'endEvent', 'intermediateCatchEvent']:
                    elem_id = elem.get('id', '')
                    if elem_id:
                        actual_elements.add(elem_id)
            
            # Compare expected vs actual
            if expected_elements:
                # Check if pattern structure elements are represented
                # This is a simplified check - in reality, we'd need to map
                # pattern node IDs to actual XML element IDs
                found_count = 0
                for expected in expected_elements:
                    # Check if any element matches expected pattern
                    if any(expected.lower() in elem.lower() for elem in actual_elements):
                        found_count += 1
                
                score = found_count / len(expected_elements) if expected_elements else 1.0
                
                if score < 0.7:
                    issues.append(f"Pattern '{pattern.name}' elements not fully present ({score:.2f} coverage)")
                    suggestions.append(f"Ensure all pattern elements are generated: {', '.join(list(expected_elements)[:3])}")
                
                return score
        
        # If no structure defined, check tags/keywords
        return self._validate_pattern_structure(xml_content, pattern, issues, suggestions)
    
    def _validate_pattern_relations(
        self,
        graph: ProcessGraph,
        pattern: BPMNPattern,
        issues: List[str],
        suggestions: List[str]
    ) -> float:
        """Validate pattern relations match graph structure."""
        if not pattern.structure or not pattern.structure.edges:
            return 1.0  # No relations to validate
        
        # Build graph structure from pattern
        pattern_edges = set(pattern.structure.edges)
        
        # Build actual graph edges
        actual_edges = set()
        for edge in graph.edges:
            source_id = edge.source_id if hasattr(edge, 'source_id') else getattr(edge, 'source', None)
            target_id = edge.target_id if hasattr(edge, 'target_id') else getattr(edge, 'target', None)
            
            if source_id and target_id:
                edge_str = f"{source_id}->{target_id}"
                actual_edges.add(edge_str)
        
        # Compare pattern edges with actual edges
        # This is simplified - real validation would need to map pattern node IDs
        # to actual graph node IDs
        
        if not pattern_edges:
            return 1.0
        
        # Check if pattern edge structure is represented
        # Count matching edge patterns
        matches = 0
        for pattern_edge in pattern_edges:
            # Check if pattern edge structure matches any actual edge
            # Simplified: check if edge pattern keywords appear
            if any(keyword in str(actual_edges) for keyword in pattern_edge.split('->')):
                matches += 1
        
        score = matches / len(pattern_edges) if pattern_edges else 1.0
        
        if score < 0.7:
            issues.append(f"Pattern '{pattern.name}' relations not fully matched ({score:.2f} coverage)")
            suggestions.append(f"Ensure pattern relations are correctly implemented: {', '.join(list(pattern_edges)[:2])}")
        
        return score
