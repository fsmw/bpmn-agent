"""
Unit Tests for Validation Integration Layer (Phase 4)

Tests covering:
- Integration layer initialization
- Unified validation orchestration
- Result combination
- Graceful degradation
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import List, Optional

from bpmn_agent.validation.integration_layer import (
    ValidationIntegrationLayer,
    UnifiedValidationResult,
)
from bpmn_agent.validation.enhanced_xsd_validation import (
    EnhancedXSDValidator,
    XSDValidationResult,
    XSDValidationError,
    XSDValidationErrorLevel,
    ValidationErrorCategory,
)
from bpmn_agent.validation.rag_pattern_validator import (
    RAGPatternValidator,
    RAGValidationResult,
    PatternComplianceFinding,
)
from bpmn_agent.validation.rag_feedback_loop import RAGFeedbackLoop
from bpmn_agent.models.graph import ProcessGraph, GraphNode, GraphEdge
from bpmn_agent.models.extraction import ExtractionResultWithErrors
from bpmn_agent.models.knowledge_base import DomainType


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
def sample_xsd_result():
    """Sample XSD validation result."""
    return XSDValidationResult(
        is_valid=True, quality_score=0.85, errors=[], warnings=[], suggestions=[]
    )


@pytest.fixture
def sample_rag_result():
    """Sample RAG validation result."""
    return RAGValidationResult(
        findings=[
            PatternComplianceFinding(
                pattern_id="pattern_1",
                pattern_name="Test Pattern",
                structure_compliance=0.9,
                element_compliance=0.8,
                relation_compliance=0.85,
                overall_score=0.85,
                issues=[],
                suggestions=[],
            )
        ],
        overall_compliance_score=0.85,
        patterns_validated=1,
        patterns_passed=1,
    )


@pytest.fixture
def mock_xsd_validator(sample_xsd_result):
    """Mock XSD validator."""
    validator = Mock(spec=EnhancedXSDValidator)
    validator.validate = Mock(return_value=sample_xsd_result)
    return validator


@pytest.fixture
def mock_rag_validator(sample_rag_result):
    """Mock RAG validator."""
    validator = Mock(spec=RAGPatternValidator)
    validator.validate_pattern_compliance = Mock(return_value=sample_rag_result)
    return validator


@pytest.fixture
def mock_feedback_loop():
    """Mock feedback loop."""
    loop = Mock(spec=RAGFeedbackLoop)
    loop.record_validation_findings = Mock()
    loop.get_feedback_summary = Mock(return_value={})
    return loop


class TestValidationIntegrationLayer:
    """Tests for ValidationIntegrationLayer."""

    def test_initialization_with_defaults(self):
        """Test initialization with default validators."""
        layer = ValidationIntegrationLayer()
        assert layer.xsd_validator is not None
        assert isinstance(layer.xsd_validator, EnhancedXSDValidator)

    def test_initialization_with_custom_validators(
        self, mock_xsd_validator, mock_rag_validator, mock_feedback_loop
    ):
        """Test initialization with custom validators."""
        layer = ValidationIntegrationLayer(
            xsd_validator=mock_xsd_validator,
            rag_validator=mock_rag_validator,
            feedback_loop=mock_feedback_loop,
            enable_rag=True,
        )

        assert layer.xsd_validator == mock_xsd_validator
        assert layer.rag_validator == mock_rag_validator
        assert layer.feedback_loop == mock_feedback_loop
        assert layer.rag_enabled is True

    def test_initialization_without_rag(self, mock_xsd_validator):
        """Test initialization without RAG."""
        layer = ValidationIntegrationLayer(xsd_validator=mock_xsd_validator, enable_rag=False)

        assert layer.xsd_validator == mock_xsd_validator
        assert layer.rag_enabled is False
        assert layer.rag_validator is None
        assert layer.feedback_loop is None

    def test_validate_xsd_only(self, mock_xsd_validator, sample_bpmn_xml, sample_xsd_result):
        """Test validation with XSD only."""
        layer = ValidationIntegrationLayer(xsd_validator=mock_xsd_validator, enable_rag=False)

        result = layer.validate(xml_content=sample_bpmn_xml)

        assert isinstance(result, UnifiedValidationResult)
        assert result.xsd_result == sample_xsd_result
        assert result.rag_result is None
        mock_xsd_validator.validate.assert_called_once()

    def test_validate_with_rag(
        self,
        mock_xsd_validator,
        mock_rag_validator,
        mock_feedback_loop,
        sample_bpmn_xml,
        sample_xsd_result,
        sample_rag_result,
        sample_process_graph,
    ):
        """Test validation with RAG enabled."""
        layer = ValidationIntegrationLayer(
            xsd_validator=mock_xsd_validator,
            rag_validator=mock_rag_validator,
            feedback_loop=mock_feedback_loop,
            enable_rag=True,
        )

        result = layer.validate(
            xml_content=sample_bpmn_xml,
            graph=sample_process_graph,
            patterns_applied=["pattern_1"],
            domain=DomainType.GENERIC,
        )

        assert isinstance(result, UnifiedValidationResult)
        assert result.xsd_result == sample_xsd_result
        assert result.rag_result == sample_rag_result
        assert result.patterns_applied == ["pattern_1"]
        mock_rag_validator.validate_pattern_compliance.assert_called_once()

    def test_validate_without_patterns_applied(
        self, mock_xsd_validator, mock_rag_validator, sample_bpmn_xml, sample_xsd_result
    ):
        """Test validation without patterns applied (RAG should not run)."""
        layer = ValidationIntegrationLayer(
            xsd_validator=mock_xsd_validator, rag_validator=mock_rag_validator, enable_rag=True
        )

        result = layer.validate(xml_content=sample_bpmn_xml, patterns_applied=None)

        assert isinstance(result, UnifiedValidationResult)
        assert result.rag_result is None
        mock_rag_validator.validate_pattern_compliance.assert_not_called()

    def test_validate_records_feedback(
        self,
        mock_xsd_validator,
        mock_rag_validator,
        mock_feedback_loop,
        sample_bpmn_xml,
        sample_xsd_result,
    ):
        """Test that validation records feedback."""
        layer = ValidationIntegrationLayer(
            xsd_validator=mock_xsd_validator,
            rag_validator=mock_rag_validator,
            feedback_loop=mock_feedback_loop,
            enable_rag=True,
        )

        layer.validate(
            xml_content=sample_bpmn_xml, patterns_applied=["pattern_1"], domain=DomainType.GENERIC
        )

        mock_feedback_loop.record_validation_findings.assert_called_once()

    def test_validate_handles_rag_failure(
        self, mock_xsd_validator, mock_rag_validator, sample_bpmn_xml, sample_xsd_result
    ):
        """Test that validation handles RAG validator failure gracefully."""
        mock_rag_validator.validate_pattern_compliance = Mock(side_effect=Exception("RAG error"))

        layer = ValidationIntegrationLayer(
            xsd_validator=mock_xsd_validator, rag_validator=mock_rag_validator, enable_rag=True
        )

        result = layer.validate(xml_content=sample_bpmn_xml, patterns_applied=["pattern_1"])

        # Should still return result with XSD validation
        assert isinstance(result, UnifiedValidationResult)
        assert result.xsd_result == sample_xsd_result
        assert result.rag_result is None

    def test_combine_results(self, mock_xsd_validator, sample_xsd_result, sample_rag_result):
        """Test result combination."""
        layer = ValidationIntegrationLayer(xsd_validator=mock_xsd_validator, enable_rag=False)

        unified = layer._combine_results(
            xsd_result=sample_xsd_result,
            rag_result=sample_rag_result,
            patterns_applied=["pattern_1"],
            domain=DomainType.GENERIC,
        )

        assert isinstance(unified, UnifiedValidationResult)
        assert unified.xsd_result == sample_xsd_result
        assert unified.rag_result == sample_rag_result
        assert unified.patterns_applied == ["pattern_1"]
        assert unified.domain == DomainType.GENERIC

    def test_combine_results_no_rag(self, mock_xsd_validator, sample_xsd_result):
        """Test result combination without RAG."""
        layer = ValidationIntegrationLayer(xsd_validator=mock_xsd_validator, enable_rag=False)

        unified = layer._combine_results(
            xsd_result=sample_xsd_result, rag_result=None, patterns_applied=[], domain=None
        )

        assert isinstance(unified, UnifiedValidationResult)
        assert unified.xsd_result == sample_xsd_result
        assert unified.rag_result is None

    def test_get_validation_summary(self, mock_xsd_validator, sample_xsd_result, sample_rag_result):
        """Test getting validation summary."""
        layer = ValidationIntegrationLayer(xsd_validator=mock_xsd_validator, enable_rag=False)

        unified = UnifiedValidationResult(
            xsd_result=sample_xsd_result,
            rag_result=sample_rag_result,
            overall_valid=True,
            overall_quality_score=0.85,
        )

        summary = layer.get_validation_summary(unified)

        assert isinstance(summary, dict)
        assert "is_valid" in summary
        assert "quality_score" in summary

    def test_get_feedback_summary(self, mock_xsd_validator, mock_feedback_loop):
        """Test getting feedback summary."""
        layer = ValidationIntegrationLayer(
            xsd_validator=mock_xsd_validator, feedback_loop=mock_feedback_loop, enable_rag=True
        )

        summary = layer.get_feedback_summary()

        assert summary is not None
        mock_feedback_loop.get_feedback_summary.assert_called_once()

    def test_get_feedback_summary_no_feedback_loop(self, mock_xsd_validator):
        """Test getting feedback summary without feedback loop."""
        layer = ValidationIntegrationLayer(xsd_validator=mock_xsd_validator, enable_rag=False)

        summary = layer.get_feedback_summary()

        assert summary is None


@pytest.mark.unit
class TestValidationIntegrationLayerIntegration:
    """Integration tests for ValidationIntegrationLayer."""

    def test_full_validation_workflow(self, sample_bpmn_xml, sample_process_graph):
        """Test complete validation workflow."""
        layer = ValidationIntegrationLayer(enable_rag=False)

        result = layer.validate(xml_content=sample_bpmn_xml, graph=sample_process_graph)

        assert isinstance(result, UnifiedValidationResult)
        assert result.xsd_result is not None

    def test_validation_with_extraction_result(self, sample_bpmn_xml, sample_xsd_result):
        """Test validation with extraction result."""
        mock_xsd_validator = Mock(spec=EnhancedXSDValidator)
        mock_xsd_validator.validate = Mock(return_value=sample_xsd_result)

        layer = ValidationIntegrationLayer(xsd_validator=mock_xsd_validator, enable_rag=False)

        extraction_result = ExtractionResultWithErrors(entities=[], relations=[], errors=[])

        result = layer.validate(xml_content=sample_bpmn_xml, extraction_result=extraction_result)

        assert isinstance(result, UnifiedValidationResult)
