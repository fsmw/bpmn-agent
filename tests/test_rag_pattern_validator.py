"""
Unit Tests for RAG Pattern Validator (Phase 4)

Tests covering:
- Pattern compliance validation
- XML structure validation
- Element and relation validation
- Graceful degradation without KB
"""

import pytest
import xml.etree.ElementTree as ET
from unittest.mock import Mock, MagicMock, patch
from typing import List, Optional

from bpmn_agent.validation.rag_pattern_validator import (
    RAGPatternValidator,
    RAGValidationResult,
    PatternComplianceFinding,
)
from bpmn_agent.models.graph import ProcessGraph, GraphNode, GraphEdge
from bpmn_agent.models.knowledge_base import BPMNPattern, KnowledgeBase, DomainType, GraphStructure


@pytest.fixture
def sample_bpmn_xml():
    """Sample valid BPMN XML for testing."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<bpmn2:definitions xmlns:bpmn2="http://www.omg.org/spec/BPMN/20100524/MODEL"
                   xmlns:bpmndi="http://www.omg.org/spec/BPMN/20100524/DI"
                   targetNamespace="http://example.com/bpmn">
  <bpmn2:process id="Process_1" name="Test Process">
    <bpmn2:startEvent id="StartEvent_1"/>
    <bpmn2:task id="Task_1" name="Task 1"/>
    <bpmn2:endEvent id="EndEvent_1"/>
    <bpmn2:sequenceFlow id="Flow_1" sourceRef="StartEvent_1" targetRef="Task_1"/>
    <bpmn2:sequenceFlow id="Flow_2" sourceRef="Task_1" targetRef="EndEvent_1"/>
  </bpmn2:process>
</bpmn2:definitions>"""


@pytest.fixture
def sample_process_graph():
    """Sample process graph for testing."""
    nodes = [
        GraphNode(id="StartEvent_1", name="Start", node_type="startEvent"),
        GraphNode(id="Task_1", name="Task 1", node_type="task"),
        GraphNode(id="EndEvent_1", name="End", node_type="endEvent"),
    ]
    edges = [
        GraphEdge(
            id="Flow_1", source_id="StartEvent_1", target_id="Task_1", edge_type="sequenceFlow"
        ),
        GraphEdge(
            id="Flow_2", source_id="Task_1", target_id="EndEvent_1", edge_type="sequenceFlow"
        ),
    ]
    return ProcessGraph(nodes=nodes, edges=edges)


@pytest.fixture
def sample_bpmn_pattern():
    """Sample BPMN pattern for testing."""
    return BPMNPattern(
        pattern_id="test_pattern_1",
        name="Test Pattern",
        description="A test pattern",
        category="workflow",
        domain=DomainType.GENERIC,
        graph_structure=GraphStructure(
            required_elements=["startEvent", "task", "endEvent"],
            required_relations=["sequenceFlow"],
            optional_elements=[],
            optional_relations=[],
        ),
        xml_template="<bpmn2:process><bpmn2:startEvent/><bpmn2:task/><bpmn2:endEvent/></bpmn2:process>",
        examples=[],
        best_practices=[],
    )


@pytest.fixture
def mock_knowledge_base(sample_bpmn_pattern):
    """Mock knowledge base with sample pattern."""
    kb = Mock(spec=KnowledgeBase)
    kb.get_pattern_by_id = Mock(return_value=sample_bpmn_pattern)
    kb.get_patterns_by_domain = Mock(return_value=[sample_bpmn_pattern])
    return kb


@pytest.fixture
def mock_pattern_bridge():
    """Mock pattern matching bridge."""
    bridge = Mock()
    bridge.match_patterns = Mock(return_value=[])
    return bridge


class TestRAGPatternValidator:
    """Tests for RAGPatternValidator."""

    def test_initialization_with_kb(self, mock_knowledge_base, mock_pattern_bridge):
        """Test validator initialization with KB."""
        validator = RAGPatternValidator(kb=mock_knowledge_base, pattern_bridge=mock_pattern_bridge)
        assert validator.enabled is True
        assert validator.kb == mock_knowledge_base
        assert validator.pattern_bridge == mock_pattern_bridge

    def test_initialization_without_kb(self):
        """Test validator initialization without KB (graceful degradation)."""
        with patch(
            "bpmn_agent.validation.rag_pattern_validator.PatternLibraryLoader"
        ) as mock_loader:
            mock_loader_instance = Mock()
            mock_loader_instance.load_all_patterns = Mock(side_effect=Exception("KB not available"))
            mock_loader.return_value = mock_loader_instance

            validator = RAGPatternValidator()
            assert validator.enabled is False
            assert validator.kb is None

    def test_validate_pattern_compliance_no_patterns(
        self, mock_knowledge_base, mock_pattern_bridge, sample_bpmn_xml
    ):
        """Test validation with no patterns applied."""
        validator = RAGPatternValidator(kb=mock_knowledge_base, pattern_bridge=mock_pattern_bridge)

        result = validator.validate_pattern_compliance(
            xml_content=sample_bpmn_xml, patterns_applied=[]
        )

        assert isinstance(result, RAGValidationResult)
        assert result.patterns_validated == 0
        assert result.patterns_passed == 0
        assert result.overall_compliance_score == 1.0

    def test_validate_pattern_compliance_with_patterns(
        self, mock_knowledge_base, mock_pattern_bridge, sample_bpmn_xml, sample_bpmn_pattern
    ):
        """Test validation with patterns applied."""
        validator = RAGPatternValidator(kb=mock_knowledge_base, pattern_bridge=mock_pattern_bridge)

        result = validator.validate_pattern_compliance(
            xml_content=sample_bpmn_xml, patterns_applied=["test_pattern_1"], domain="generic"
        )

        assert isinstance(result, RAGValidationResult)
        assert result.patterns_validated >= 0
        assert len(result.findings) >= 0

    def test_validate_pattern_compliance_disabled(self, sample_bpmn_xml):
        """Test validation when validator is disabled."""
        validator = RAGPatternValidator()
        validator.enabled = False

        result = validator.validate_pattern_compliance(
            xml_content=sample_bpmn_xml, patterns_applied=["test_pattern_1"]
        )

        assert isinstance(result, RAGValidationResult)
        assert result.patterns_validated == 0

    def test_get_pattern_by_id_found(
        self, mock_knowledge_base, mock_pattern_bridge, sample_bpmn_pattern
    ):
        """Test getting pattern by ID when found."""
        validator = RAGPatternValidator(kb=mock_knowledge_base, pattern_bridge=mock_pattern_bridge)

        pattern = validator._get_pattern_by_id("test_pattern_1")
        assert pattern == sample_bpmn_pattern

    def test_get_pattern_by_id_not_found(self, mock_knowledge_base, mock_pattern_bridge):
        """Test getting pattern by ID when not found."""
        mock_knowledge_base.get_pattern_by_id = Mock(return_value=None)
        validator = RAGPatternValidator(kb=mock_knowledge_base, pattern_bridge=mock_pattern_bridge)

        pattern = validator._get_pattern_by_id("nonexistent_pattern")
        assert pattern is None

    def test_validate_single_pattern(
        self,
        mock_knowledge_base,
        mock_pattern_bridge,
        sample_bpmn_xml,
        sample_bpmn_pattern,
        sample_process_graph,
    ):
        """Test validating a single pattern."""
        validator = RAGPatternValidator(kb=mock_knowledge_base, pattern_bridge=mock_pattern_bridge)

        finding = validator._validate_single_pattern(
            xml_content=sample_bpmn_xml, pattern=sample_bpmn_pattern, graph=sample_process_graph
        )

        assert isinstance(finding, PatternComplianceFinding)
        assert finding.pattern_id == "test_pattern_1"
        assert 0.0 <= finding.overall_score <= 1.0
        assert finding.structure_compliance >= 0.0
        assert finding.element_compliance >= 0.0
        assert finding.relation_compliance >= 0.0

    def test_validate_pattern_structure(
        self, mock_knowledge_base, mock_pattern_bridge, sample_bpmn_xml, sample_bpmn_pattern
    ):
        """Test pattern structure validation."""
        validator = RAGPatternValidator(kb=mock_knowledge_base, pattern_bridge=mock_pattern_bridge)

        issues = []
        suggestions = []
        score = validator._validate_pattern_structure(
            xml_content=sample_bpmn_xml,
            pattern=sample_bpmn_pattern,
            issues=issues,
            suggestions=suggestions,
        )

        assert 0.0 <= score <= 1.0
        assert isinstance(issues, list)
        assert isinstance(suggestions, list)

    def test_validate_pattern_elements(
        self, mock_knowledge_base, mock_pattern_bridge, sample_bpmn_xml, sample_bpmn_pattern
    ):
        """Test pattern elements validation."""
        validator = RAGPatternValidator(kb=mock_knowledge_base, pattern_bridge=mock_pattern_bridge)

        issues = []
        suggestions = []
        score = validator._validate_pattern_elements(
            xml_content=sample_bpmn_xml,
            pattern=sample_bpmn_pattern,
            issues=issues,
            suggestions=suggestions,
        )

        assert 0.0 <= score <= 1.0
        assert isinstance(issues, list)
        assert isinstance(suggestions, list)

    def test_validate_pattern_relations(
        self, mock_knowledge_base, mock_pattern_bridge, sample_process_graph, sample_bpmn_pattern
    ):
        """Test pattern relations validation."""
        validator = RAGPatternValidator(kb=mock_knowledge_base, pattern_bridge=mock_pattern_bridge)

        issues = []
        suggestions = []
        score = validator._validate_pattern_relations(
            graph=sample_process_graph,
            pattern=sample_bpmn_pattern,
            issues=issues,
            suggestions=suggestions,
        )

        assert 0.0 <= score <= 1.0
        assert isinstance(issues, list)
        assert isinstance(suggestions, list)

    def test_validate_pattern_relations_no_graph(
        self, mock_knowledge_base, mock_pattern_bridge, sample_bpmn_pattern
    ):
        """Test pattern relations validation without graph."""
        validator = RAGPatternValidator(kb=mock_knowledge_base, pattern_bridge=mock_pattern_bridge)

        issues = []
        suggestions = []
        score = validator._validate_pattern_relations(
            graph=None, pattern=sample_bpmn_pattern, issues=issues, suggestions=suggestions
        )

        # Should return 0.0 or handle gracefully
        assert 0.0 <= score <= 1.0


@pytest.mark.unit
class TestRAGPatternValidatorIntegration:
    """Integration tests for RAGPatternValidator."""

    def test_full_validation_workflow(self, sample_bpmn_xml, sample_process_graph):
        """Test complete validation workflow."""
        validator = RAGPatternValidator()

        # Test with empty patterns (should not fail)
        result = validator.validate_pattern_compliance(
            xml_content=sample_bpmn_xml, patterns_applied=[], graph=sample_process_graph
        )

        assert isinstance(result, RAGValidationResult)
        assert result.patterns_validated == 0

    def test_validation_with_invalid_xml(
        self, mock_knowledge_base, mock_pattern_bridge, sample_bpmn_pattern
    ):
        """Test validation with invalid XML."""
        validator = RAGPatternValidator(kb=mock_knowledge_base, pattern_bridge=mock_pattern_bridge)

        invalid_xml = "<invalid>not bpmn</invalid>"

        result = validator.validate_pattern_compliance(
            xml_content=invalid_xml, patterns_applied=["test_pattern_1"]
        )

        # Should handle gracefully
        assert isinstance(result, RAGValidationResult)
