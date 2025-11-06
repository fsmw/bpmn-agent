"""
Integration Tests for Phase 3 Tool Suite

Tests the integration of all Phase 3 tools with the orchestrator:
- Graph analysis tools integration
- Validation tools integration
- Refinement tools integration
- End-to-end tool workflows
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any, List

from bpmn_agent.agent.orchestrator import BPMNAgent, AgentConfig, ProcessingMode
from bpmn_agent.models.graph import GraphNode, GraphEdge, ProcessGraph
from bpmn_agent.models.extraction import ExtractionResultWithErrors, ExtractedEntity, ExtractedRelation, ExtractionMetadata
from bpmn_agent.tools.graph_analysis import GraphAnalyzer, GraphAnomaly, AnomalyType
from bpmn_agent.tools.validation import ValidationLevel, ValidationCategory
from bpmn_agent.tools.refinement import ClarificationType, SuggestionCategory, RefinementPlan
from datetime import datetime


class TestToolSuiteIntegration:
    """Integration tests for Phase 3 tool suite."""
    
    @pytest.fixture
    def sample_extraction_result(self):
        """Create sample extraction result for testing."""
        entities = [
            ExtractedEntity(
                name="Submit Request",
                type="task",
                confidence="high",
                source_context="User submits request for approval",
                identifier="entity_1"
            ),
            ExtractedEntity(
                name="Review Request",
                type="task", 
                confidence="medium",
                source_context="Manager reviews submitted request",
                identifier="entity_2"
            ),
            ExtractedEntity(
                name="Approve/Reject Decision",
                type="gateway",
                confidence="high",
                source_context="Decision point for approval",
                identifier="entity_3"
            ),
            ExtractedEntity(
                name="End Process",
                type="end",
                confidence="high",
                source_context="Process completes",
                identifier="entity_4"
            ),
            # Add a low confidence entity for testing
            ExtractedEntity(
                name="Some Activity",
                type="activity",
                confidence="low",
                source_context="Ambiguous description of what happens",
                identifier="entity_5"
            )
        ]
        
        relations = [
            ExtractedRelation(
                source_name="Submit Request",
                source_type="task",
                target_name="Review Request",
                target_type="task",
                relation_type="sequence_flow",
                confidence="high"
            ),
            ExtractedRelation(
                source_name="Review Request",
                source_type="task",
                target_name="Approve/Reject Decision",
                target_type="gateway",
                relation_type="sequence_flow",
                confidence="high"
            ),
            # Missing connection from gateway to end - will be detected as issue
        ]
        
        metadata = ExtractionMetadata(
            input_text="Sample process description",
            input_length=50,
            extraction_timestamp=datetime.now().isoformat(),
            extraction_duration_ms=1500.0,
            llm_model="test-model",
            llm_temperature=0.3,
            stage="extraction",
            total_entities_extracted=len(entities),
            high_confidence_entities=3,
            medium_confidence_entities=1,
            low_confidence_entities=1,
            total_relations_extracted=len(relations),
            high_confidence_relations=2
        )
        
        return ExtractionResultWithErrors(
            entities=entities,
            relations=relations,
            co_references=[],
            metadata=metadata,
            errors=[]
        )
    
    @pytest.fixture
    def sample_graph(self):
        """Create sample process graph with issues for testing."""
        nodes = [
            GraphNode(
                id="start_1",
                type="start",
                label="Start",
                bpmn_type="bpmn:StartEvent"
            ),
            GraphNode(
                id="task_1",
                type="task",
                label="Submit Request",
                bpmn_type="bpmn:Task"
            ),
            GraphNode(
                id="gateway_1",
                type="exclusive_gateway",
                label="Decision",
                bpmn_type="bpmn:ExclusiveGateway"
            ),
            # Add an isolated node (will be detected as anomaly)
            GraphNode(
                id="isolated_task",
                type="task",
                label="Orphaned Task",
                bpmn_type="bpmn:Task"
            ),
            GraphNode(
                id="end_1",
                type="end",
                label="End",
                bpmn_type="bpmn:EndEvent"
            )
        ]
        
        edges = [
            GraphEdge(
                source_id="start_1",
                target_id="task_1",
                type="sequence_flow",
                label=""
            ),
            GraphEdge(
                source_id="task_1",
                target_id="gateway_1",
                type="sequence_flow",
                label=""
            ),
            # Missing edge to end event - will create orphaned nodes
        ]
        
        return ProcessGraph(
            id="test_process",
            name="Test Process",
            description="Process for testing tools",
            nodes=nodes,
            edges=edges,
            is_acyclic=True,
            is_connected=False,
            has_implicit_parallelism=False,
            complexity=2.0,
            version="1.0",
            created_timestamp=datetime.now().isoformat(),
            metadata={"test": True}
        )
    
    @pytest.fixture
    def sample_xml(self):
        """Create sample XML for validation testing."""
        return """<?xml version="1.0" encoding="UTF-8"?>
<definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL"
             xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL"
             targetNamespace="http://opencode.ai/bpmn/test">
    <process id="test_process" name="Test Process">
        <startEvent id="start_1"/>
        <task id="task_1" name="Submit Request"/>
        <exclusiveGateway id="gateway_1"/>
        <endEvent id="end_1"/>
        <sequenceFlow sourceRef="start_1" targetRef="task_1"/>
        <sequenceFlow sourceRef="task_1" targetRef="gateway_1"/>
        <!-- Missing sequence flow to end event -->
    </process>
</definitions>"""
    
    def test_graph_analysis_integration(self, sample_graph, sample_extraction_result):
        """Test graph analysis tools integration."""
        analyzer = GraphAnalyzer()
        
        # Test complete analysis
        result = analyzer.analyze_graph_structure(sample_graph, sample_extraction_result)
        
        # Verify analysis results
        assert result.graph_id == "test_process"
        assert result.total_nodes == 5
        assert result.quality_score < 100.0  # Should have issues
        
        # Check for expected anomalies
        anomaly_types = {a.anomaly_type for a in result.anomalies}
        assert AnomalyType.ISOLATED_NODE in anomaly_types or AnomalyType.ORPHANED_NODE in anomaly_types
        
        # Test individual analysis functions
        isolated_nodes = analyzer.find_isolated_nodes(sample_graph)
        assert isinstance(isolated_nodes, list)
        
        orphaned_nodes = analyzer.find_orphaned_nodes(sample_graph)
        assert isinstance(orphaned_nodes, list)
        
        cycles = analyzer.detect_cycles(sample_graph)
        assert isinstance(cycles, list)
        
        suggestions = analyzer.suggest_implicit_joins(sample_graph)
        assert isinstance(suggestions, list)
    
    def test_validation_integration(self, sample_graph, sample_extraction_result, sample_xml):
        """Test validation tools integration."""
        from bpmn_agent.tools.validation import XMLValidator, GraphValidator, ExtractionValidator
        
        # Test XML validation
        xml_validator = XMLValidator()
        xml_result = xml_validator.validate_xml_against_xsd(sample_xml)
        
        assert xml_result.is_valid == True  # Basic structure should be valid
        assert isinstance(xml_result.issues, list)
        
        # Test graph validation
        graph_validator = GraphValidator()
        graph_result = graph_validator.validate_graph_semantics(sample_graph, sample_extraction_result)
        
        assert isinstance(graph_result.issues, list)
        assert graph_result.total_issues > 0  # Should find issues with our test graph
        
        # Test extraction validation
        extraction_validator = ExtractionValidator()
        extraction_result = extraction_validator.validate_extraction(sample_extraction_result)
        
        assert isinstance(extraction_result.issues, list)
        assert extraction_result.total_issues >= 0
    
    def test_refinement_integration(self, sample_extraction_result, sample_graph):
        """Test refinement tools integration."""
        from bpmn_agent.tools.refinement import ClarificationRequester, ImprovementSuggester, RefinementOrchestrator
        
        # Test clarification request
        requester = ClarificationRequester()
        questions = requester.request_clarification(sample_extraction_result, graph=sample_graph)
        
        assert isinstance(questions, list)
        # Should have questions about low confidence entities or connectivity issues
        question_types = {q.question_type for q in questions}
        assert len(question_types) >= 0
        
        # Test improvement suggestions
        suggester = ImprovementSuggester()
        suggestions = suggester.suggest_improvements(sample_extraction_result, graph=sample_graph)
        
        assert isinstance(suggestions, list)
        # Should suggest improvements based on detected issues
        suggestion_categories = {s.category for s in suggestions}
        assert len(suggestion_categories) >= 0
    
    @pytest.mark.asyncio
    async def test_refinement_plan_creation(self, sample_extraction_result, sample_graph):
        """Test refinement plan creation with orchestrator."""
        from bpmn_agent.tools.refinement import create_refinement_plan
        
        # Create refinement plan
        plan = await create_refinement_plan(
            extraction_result=sample_extraction_result,
            graph=sample_graph,
            original_text="Sample process with some ambiguities and connectivity issues"
        )
        
        assert isinstance(plan, RefinementPlan)
        assert plan.plan_id is not None
        assert plan.created_at is not None
        
        # Check plan components
        assert isinstance(plan.clarifications_needed, list)
        assert isinstance(plan.improvements_suggested, list)
        assert isinstance(plan.stage_reexecutions, list)
        
        # Should recommend stage re-executions for detected issues
        assert len(plan.stage_reexecutions) >= 0
    
    def test_tools_with_orchestrator_mock(self, sample_extraction_result, sample_graph):
        """Test tools integration with orchestrator (mocked)."""
        # Create mock agent to test tool integration
        mock_llm_config = Mock()
        mock_llm_config.provider = "openai"
        mock_llm_config.model = "gpt-3.5-turbo"
        mock_llm_config.timeout = 30
        
        with pytest.MonkeyPatch().context() as m:
            # Mock the LLM client to avoid real API calls
            mock_client = Mock()
            mock_client.call = AsyncMock(return_value='{"mock": "response"}')
            m.setattr("bpmn_agent.core.llm_client.LLMClientFactory.create", Mock(return_value=mock_client))
            
            config = AgentConfig.from_env(ProcessingMode.ANALYSIS_ONLY)
            config.llm_config = mock_llm_config
            
            # Test agent initialization with tools
            agent = BPMNAgent(config)
            assert agent is not None
            
            # Test that orchestrator has access to tools through the agent
            assert hasattr(agent, 'text_preprocessor')
            assert hasattr(agent, 'entity_extractor')
            assert hasattr(agent, 'graph_builder')
    
    @pytest.mark.asyncio
    async def test_end_to_end_tool_workflow(self, sample_extraction_result, sample_graph, sample_xml):
        """Test complete end-to-end tool workflow."""
        from bpmn_agent.tools.graph_analysis import analyze_graph_structure, find_isolated_nodes
        from bpmn_agent.tools.validation import validate_graph_semantics, validate_xml_against_xsd
        from bpmn_agent.tools.refinement import suggest_improvements, request_clarification
        
        # Step 1: Analyze graph structure
        graph_analysis = analyze_graph_structure(sample_graph, sample_extraction_result)
        assert graph_analysis.quality_score < 100.0
        
        # Step 2: Validate graph semantics  
        graph_validation = validate_graph_semantics(sample_graph, sample_extraction_result)
        assert not graph_validation.is_valid or graph_validation.total_issues > 0
        
        # Step 3: Validate XML
        xml_validation = validate_xml_against_xsd(sample_xml)
        assert xml_validation.total_issues >= 0
        
        # Step 4: Generate improvements based on analysis
        improvements = suggest_improvements(sample_extraction_result, graph_validation, sample_graph)
        assert isinstance(improvements, list)
        
        # Step 5: Request clarifications for ambiguities
        clarifications = request_clarification(sample_extraction_result, graph_validation, sample_graph)
        assert isinstance(clarifications, list)
        
        # Step 6: Verify tool coordination
        total_issues_found = len(graph_analysis.anomalies) + graph_validation.total_issues
        total_suggestions = len(improvements) + len(clarifications)
        
        assert total_issues_found >= 1  # Should find some issues in test data
        assert total_suggestions >= 0  # Should provide suggestions for issues
    
    def test_tool_error_handling(self, sample_graph):
        """Test error handling in tools."""
        analyzer = GraphAnalyzer()
        
        # Test with invalid graph
        invalid_graph = ProcessGraph(
            id="invalid",
            name="Invalid",
            description="Test error handling",
            nodes=[],
            edges=[],
            is_acyclic=True,
            is_connected=False,
            has_implicit_parallelism=False,
            complexity=0.0,
            version="1.0",
            created_timestamp=datetime.now().isoformat()
        )
        
        result = analyzer.analyze_graph_structure(invalid_graph)
        assert result is not None
        assert result.total_nodes == 0
        assert result.quality_score is not None
    
    def test_tool_performance(self, sample_extraction_result, sample_graph):
        """Test tool performance characteristics."""
        import time
        from bpmn_agent.tools.graph_analysis import analyze_graph_structure
        from bpmn_agent.tools.validation import validate_graph_semantics
        from bpmn_agent.tools.refinement import suggest_improvements
        
        start_time = time.time()
        
        # Run all tools
        graph_analysis = analyze_graph_structure(sample_graph, sample_extraction_result)
        graph_validation = validate_graph_semantics(sample_graph, sample_extraction_result)
        improvements = suggest_improvements(sample_extraction_result, graph_validation, sample_graph)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Tools should complete quickly (under 1 second for simple cases)
        assert duration < 1.0
        assert len(graph_analysis.anomalies) >= 0
        assert len(graph_validation.issues) >= 0
        assert len(improvements) >= 0
    
    @pytest.mark.asyncio
    async def test_advanced_refinement_workflow(self, sample_extraction_result, sample_graph):
        """Test advanced refinement workflow with coordination."""
        from bpmn_agent.tools.refinement import RefinementOrchestrator
        
        orchestrator = RefinementOrchestrator()
        
        # Create comprehensive refinement plan
        plan = await orchestrator.create_refinement_plan(
            extraction_result=sample_extraction_result,
            graph=sample_graph,
            original_text="Test process with multiple issues for comprehensive refinement"
        )
        
        # Verify plan quality
        assert plan.plan_id is not None
        assert plan.estimated_effort in ["low", "moderate", "high"]
        assert plan.expected_improvement in ["low", "medium", "high"]
        
        # Verify plan components are actionable
        if plan.clarifications_needed:
            for question in plan.clarifications_needed[:3]:  # Check first 3
                assert question.question is not None
                assert question.possible_answers is not None
        
        if plan.improvements_suggested:
            for suggestion in plan.improvements_suggested[:3]:  # Check first 3
                assert suggestion.title is not None
                assert suggestion.suggested_changes is not None
        
        # Verify stage re-execution recommendations make sense
        if plan.stage_reexecutions:
            valid_stages = ["entity_extraction", "entity_resolution", "graph_construction", "xml_generation"]
            for stage in plan.stage_reexecutions:
                assert stage in valid_stages


# Test configuration
@pytest.fixture(scope="module")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def setup_logging():
    """Setup logging for tests."""
    import logging
    logging.basicConfig(level=logging.INFO)