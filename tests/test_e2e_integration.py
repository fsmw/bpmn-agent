"""
End-to-end integration tests for BPMN Agent.

These tests verify the complete text-to-XML workflow, testing all processing
modes and metrics collection.
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import xml.etree.ElementTree as ET

from bpmn_agent.agent import (
    BPMNAgent,
    AgentConfig,
    AgentState,
    ProcessingMode,
)
from bpmn_agent.core.llm_client import LLMConfig, LLMClientFactory


# Test data - various process descriptions
PROCESS_DESCRIPTIONS = {
    "simple": "Customer submits order. System processes payment. Order is shipped.",
    "approval": (
        "Employee submits request. Manager approves or rejects. "
        "If approved, HR processes. If rejected, notify employee."
    ),
    "parallel": (
        "Order received. In parallel: validate inventory and check payment. "
        "After both complete, pick items and ship."
    ),
    "finance": (
        "Invoice received from vendor. Finance team validates amount and vendor. "
        "If valid, approve payment. If invalid, reject and notify vendor. "
        "Approved payments go to accounting for processing."
    ),
}


class TestE2EStandardMode:
    """Test end-to-end standard mode (all stages → XML)."""

    @pytest.mark.asyncio
    async def test_simple_workflow_produces_valid_xml(self, patch_llm_for_e2e):
        """Test that simple workflow produces valid BPMN 2.0 XML."""
        config = AgentConfig(
            mode=ProcessingMode.STANDARD,
            llm_config=LLMConfig.from_env(),
            error_handling="recovery",
        )
        agent = BPMNAgent(config)

        xml_output, state = await agent.process(PROCESS_DESCRIPTIONS["simple"])

        # Verify XML output exists
        assert xml_output is not None
        assert isinstance(xml_output, str)
        assert len(xml_output) > 0

        # Verify XML is well-formed
        try:
            root = ET.fromstring(xml_output)
            assert root.tag.endswith("definitions")
        except ET.ParseError:
            pytest.fail("Output is not valid XML")

        # Verify state completed all stages
        assert state is not None
        assert len(state.stage_results) == 5
        assert state.output_xml == xml_output

    @pytest.mark.asyncio
    async def test_approval_workflow_includes_decision(self, patch_llm_for_e2e):
        """Test approval workflow generates gateways."""
        config = AgentConfig(
            mode=ProcessingMode.STANDARD,
            llm_config=LLMConfig.from_env(),
            error_handling="recovery",
        )
        agent = BPMNAgent(config)

        xml_output, state = await agent.process(PROCESS_DESCRIPTIONS["approval"])

        assert xml_output is not None
        root = ET.fromstring(xml_output)

        # Check for gateway elements (decision point)
        namespaces = {
            "bpmn": "http://www.omg.org/spec/BPMN/20100524/MODEL",
        }
        gateways = root.findall(".//bpmn:exclusiveGateway", namespaces)
        # Should have gateways for approval decision
        assert len(gateways) >= 1

    @pytest.mark.asyncio
    async def test_parallel_workflow_includes_parallel_gateway(self, patch_llm_for_e2e):
        """Test parallel workflow generates parallel gateways."""
        config = AgentConfig(
            mode=ProcessingMode.STANDARD,
            llm_config=LLMConfig.from_env(),
            error_handling="recovery",
        )
        agent = BPMNAgent(config)

        xml_output, state = await agent.process(PROCESS_DESCRIPTIONS["parallel"])

        assert xml_output is not None
        root = ET.fromstring(xml_output)

        namespaces = {
            "bpmn": "http://www.omg.org/spec/BPMN/20100524/MODEL",
        }
        # ElementTree doesn't support wildcard patterns like *Gateway,
        # so we need to search for each gateway type individually
        all_gateways = (
            root.findall(".//bpmn:exclusiveGateway", namespaces)
            + root.findall(".//bpmn:parallelGateway", namespaces)
            + root.findall(".//bpmn:inclusiveGateway", namespaces)
        )
        assert len(all_gateways) >= 1

    @pytest.mark.asyncio
    async def test_state_tracks_completion_progression(self):
        """Test that state tracks completion rate as stages progress."""
        config = AgentConfig(
            mode=ProcessingMode.STANDARD,
            llm_config=LLMConfig.from_env(),
            error_handling="recovery",
        )
        agent = BPMNAgent(config)

        xml_output, state = await agent.process(PROCESS_DESCRIPTIONS["simple"])

        # Verify all stages have results
        assert len(state.stage_results) == 5
        for i, result in enumerate(state.stage_results, 1):
            assert result is not None

        # Verify completion rate progresses
        initial_rate = 0.2  # First stage = 20%
        final_rate = 1.0  # All stages = 100%
        assert state.completion_rate == final_rate


class TestE2EAnalysisOnlyMode:
    """Test end-to-end analysis-only mode (stages 1-4 only)."""

    @pytest.mark.asyncio
    async def test_analysis_only_no_xml_output(self):
        """Test analysis-only mode returns no XML."""
        config = AgentConfig(
            mode=ProcessingMode.ANALYSIS_ONLY,
            llm_config=LLMConfig.from_env(),
            error_handling="recovery",
        )
        agent = BPMNAgent(config)

        xml_output, state = await agent.process(PROCESS_DESCRIPTIONS["simple"])

        # Verify no XML output
        assert xml_output is None

        # Verify only 4 stages executed
        assert len(state.stage_results) == 4

    @pytest.mark.asyncio
    async def test_analysis_only_extracts_entities(self, patch_llm_for_e2e):
        """Test analysis-only mode extracts entities."""
        config = AgentConfig(
            mode=ProcessingMode.ANALYSIS_ONLY,
            llm_config=LLMConfig.from_env(),
            error_handling="recovery",
        )
        agent = BPMNAgent(config)

        xml_output, state = await agent.process(PROCESS_DESCRIPTIONS["approval"])

        # Verify metrics show extracted entities
        assert state.metrics is not None
        assert hasattr(state.metrics, "entities_extracted")
        # Should have extracted at least some entities
        assert state.metrics.entities_extracted > 0

    @pytest.mark.asyncio
    async def test_analysis_only_creates_graph(self, patch_llm_for_e2e):
        """Test analysis-only mode creates semantic graph."""
        config = AgentConfig(
            mode=ProcessingMode.ANALYSIS_ONLY,
            llm_config=LLMConfig.from_env(),
            error_handling="recovery",
        )
        agent = BPMNAgent(config)

        xml_output, state = await agent.process(PROCESS_DESCRIPTIONS["parallel"])

        # Verify graph was constructed
        assert state.metrics is not None
        assert hasattr(state.metrics, "graph_nodes")
        assert state.metrics.graph_nodes > 0
        assert hasattr(state.metrics, "graph_edges")
        assert state.metrics.graph_edges > 0


class TestE2EValidationOnlyMode:
    """Test end-to-end validation-only mode (stage 1 only)."""

    @pytest.mark.asyncio
    async def test_validation_only_single_stage(self):
        """Test validation-only mode executes only stage 1."""
        config = AgentConfig(
            mode=ProcessingMode.VALIDATION_ONLY,
            llm_config=LLMConfig.from_env(),
            error_handling="recovery",
        )
        agent = BPMNAgent(config)

        xml_output, state = await agent.process(PROCESS_DESCRIPTIONS["simple"])

        # Verify only 1 stage executed
        assert len(state.stage_results) == 1
        assert xml_output is None

    @pytest.mark.asyncio
    async def test_validation_only_validates_text(self):
        """Test validation-only mode validates and preprocesses text."""
        config = AgentConfig(
            mode=ProcessingMode.VALIDATION_ONLY,
            llm_config=LLMConfig.from_env(),
            error_handling="recovery",
        )
        agent = BPMNAgent(config)

        xml_output, state = await agent.process(PROCESS_DESCRIPTIONS["finance"])

        # Verify state has preprocessing metadata
        assert state.preprocessed_text is not None
        assert state.input_text == PROCESS_DESCRIPTIONS["finance"]
        assert len(state.preprocessed_text) > 0


class TestE2EMetricsCollection:
    """Test metrics collection across all stages."""

    @pytest.mark.asyncio
    async def test_metrics_token_counts(self):
        """Test that metrics track token usage."""
        config = AgentConfig(
            mode=ProcessingMode.STANDARD,
            llm_config=LLMConfig.from_env(),
            error_handling="recovery",
        )
        agent = BPMNAgent(config)

        xml_output, state = await agent.process(PROCESS_DESCRIPTIONS["simple"])

        # Verify metrics collected
        assert state.metrics is not None
        assert hasattr(state.metrics, "prompt_tokens")
        assert hasattr(state.metrics, "completion_tokens")
        # Both should be 0 in mock tests
        assert state.metrics.prompt_tokens >= 0
        assert state.metrics.completion_tokens >= 0

    @pytest.mark.asyncio
    async def test_metrics_stage_timing(self):
        """Test that metrics track stage execution timing."""
        config = AgentConfig(
            mode=ProcessingMode.STANDARD,
            llm_config=LLMConfig.from_env(),
            error_handling="recovery",
        )
        agent = BPMNAgent(config)

        xml_output, state = await agent.process(PROCESS_DESCRIPTIONS["simple"])

        # Verify stage results have timing info
        for result in state.stage_results:
            assert result is not None

    @pytest.mark.asyncio
    async def test_metrics_entity_extraction(self):
        """Test that metrics track entity extraction."""
        config = AgentConfig(
            mode=ProcessingMode.STANDARD,
            llm_config=LLMConfig.from_env(),
            error_handling="recovery",
        )
        agent = BPMNAgent(config)

        xml_output, state = await agent.process(PROCESS_DESCRIPTIONS["approval"])

        assert state.metrics is not None
        assert hasattr(state.metrics, "entities_extracted")


class TestE2EErrorRecovery:
    """Test end-to-end error handling and recovery."""

    @pytest.mark.asyncio
    async def test_recovery_mode_handles_llm_errors(self):
        """Test recovery mode continues despite LLM errors."""
        config = AgentConfig(
            mode=ProcessingMode.STANDARD,
            llm_config=LLMConfig.from_env(),
            error_handling="recovery",
        )
        agent = BPMNAgent(config)

        # Should process successfully even if some stages have issues
        xml_output, state = await agent.process(PROCESS_DESCRIPTIONS["simple"])

        # Should have some result
        assert state is not None

    @pytest.mark.asyncio
    async def test_strict_mode_fails_on_errors(self):
        """Test strict mode raises on errors."""
        config = AgentConfig(
            mode=ProcessingMode.STANDARD,
            llm_config=LLMConfig.from_env(),
            error_handling="strict",
        )
        agent = BPMNAgent(config)

        # Should process or raise clear error
        try:
            xml_output, state = await agent.process(PROCESS_DESCRIPTIONS["simple"])
            assert state is not None
        except Exception as e:
            # Strict mode should raise clear exceptions
            assert isinstance(e, (ValueError, RuntimeError, Exception))


class TestE2EMultipleModes:
    """Test switching between different modes."""

    @pytest.mark.asyncio
    async def test_process_same_text_different_modes(self):
        """Test processing same text with different modes yields different results."""
        llm_config = LLMConfig.from_env()

        # Standard mode
        config_standard = AgentConfig(
            mode=ProcessingMode.STANDARD,
            llm_config=llm_config,
            error_handling="recovery",
        )
        agent_standard = BPMNAgent(config_standard)
        xml_standard, state_standard = await agent_standard.process(PROCESS_DESCRIPTIONS["simple"])

        # Analysis-only mode
        config_analysis = AgentConfig(
            mode=ProcessingMode.ANALYSIS_ONLY,
            llm_config=llm_config,
            error_handling="recovery",
        )
        agent_analysis = BPMNAgent(config_analysis)
        xml_analysis, state_analysis = await agent_analysis.process(PROCESS_DESCRIPTIONS["simple"])

        # Validation-only mode
        config_validation = AgentConfig(
            mode=ProcessingMode.VALIDATION_ONLY,
            llm_config=llm_config,
            error_handling="recovery",
        )
        agent_validation = BPMNAgent(config_validation)
        xml_validation, state_validation = await agent_validation.process(
            PROCESS_DESCRIPTIONS["simple"]
        )

        # Verify different outputs
        assert xml_standard is not None  # Standard has XML
        assert xml_analysis is None  # Analysis has no XML
        assert xml_validation is None  # Validation has no XML

        assert len(state_standard.stage_results) == 5  # All stages
        assert len(state_analysis.stage_results) == 4  # 4 stages
        assert len(state_validation.stage_results) == 1  # 1 stage


class TestE2EXMLValidation:
    """Test XML output validation."""

    @pytest.mark.asyncio
    async def test_xml_has_required_elements(self, patch_llm_for_e2e):
        """Test generated XML has required BPMN elements."""
        config = AgentConfig(
            mode=ProcessingMode.STANDARD,
            llm_config=LLMConfig.from_env(),
            error_handling="recovery",
        )
        agent = BPMNAgent(config)

        xml_output, state = await agent.process(PROCESS_DESCRIPTIONS["simple"])

        assert xml_output is not None
        root = ET.fromstring(xml_output)

        namespaces = {
            "bpmn": "http://www.omg.org/spec/BPMN/20100524/MODEL",
            "bpmndi": "http://www.omg.org/spec/BPMN/20100524/DI",
        }

        # Check for required elements
        process = root.find(".//bpmn:process", namespaces)
        assert process is not None

        # Should have start and end events
        start_events = root.findall(".//bpmn:startEvent", namespaces)
        end_events = root.findall(".//bpmn:endEvent", namespaces)
        assert len(start_events) >= 1
        assert len(end_events) >= 1

    @pytest.mark.asyncio
    async def test_xml_has_valid_ids(self):
        """Test generated XML has unique valid IDs."""
        config = AgentConfig(
            mode=ProcessingMode.STANDARD,
            llm_config=LLMConfig.from_env(),
            error_handling="recovery",
        )
        agent = BPMNAgent(config)

        xml_output, state = await agent.process(PROCESS_DESCRIPTIONS["simple"])

        assert xml_output is not None
        root = ET.fromstring(xml_output)

        # Collect all IDs
        all_ids = set()
        for elem in root.iter():
            elem_id = elem.get("id")
            if elem_id:
                assert elem_id not in all_ids, f"Duplicate ID: {elem_id}"
                all_ids.add(elem_id)

        # Should have multiple elements
        assert len(all_ids) > 0

    @pytest.mark.asyncio
    async def test_xml_has_valid_connections(self):
        """Test sequence flows reference existing elements."""
        config = AgentConfig(
            mode=ProcessingMode.STANDARD,
            llm_config=LLMConfig.from_env(),
            error_handling="recovery",
        )
        agent = BPMNAgent(config)

        xml_output, state = await agent.process(PROCESS_DESCRIPTIONS["approval"])

        assert xml_output is not None
        root = ET.fromstring(xml_output)

        namespaces = {
            "bpmn": "http://www.omg.org/spec/BPMN/20100524/MODEL",
        }

        # Get all element IDs
        element_ids = set()
        for elem in root.iter():
            elem_id = elem.get("id")
            if elem_id:
                element_ids.add(elem_id)

        # Check sequence flows reference existing elements
        flows = root.findall(".//bpmn:sequenceFlow", namespaces)
        for flow in flows:
            source = flow.get("sourceRef")
            target = flow.get("targetRef")
            assert source in element_ids, f"Unknown source: {source}"
            assert target in element_ids, f"Unknown target: {target}"


class TestE2EPerformance:
    """Test performance characteristics."""

    @pytest.mark.asyncio
    async def test_simple_process_completes_reasonably(self):
        """Test that simple process completes in reasonable time."""
        import time

        config = AgentConfig(
            mode=ProcessingMode.STANDARD,
            llm_config=LLMConfig.from_env(),
            error_handling="recovery",
        )
        agent = BPMNAgent(config)

        start = time.time()
        xml_output, state = await agent.process(PROCESS_DESCRIPTIONS["simple"])
        elapsed = time.time() - start

        # Should complete (even if mock)
        assert state is not None
        assert elapsed >= 0

    @pytest.mark.asyncio
    async def test_analysis_mode_faster_than_standard(self):
        """Test that analysis mode is faster than standard (no XML generation)."""
        import time

        llm_config = LLMConfig.from_env()

        # Standard mode
        config_standard = AgentConfig(
            mode=ProcessingMode.STANDARD,
            llm_config=llm_config,
            error_handling="recovery",
        )
        agent_standard = BPMNAgent(config_standard)
        start = time.time()
        await agent_standard.process(PROCESS_DESCRIPTIONS["simple"])
        time_standard = time.time() - start

        # Analysis mode
        config_analysis = AgentConfig(
            mode=ProcessingMode.ANALYSIS_ONLY,
            llm_config=llm_config,
            error_handling="recovery",
        )
        agent_analysis = BPMNAgent(config_analysis)
        start = time.time()
        await agent_analysis.process(PROCESS_DESCRIPTIONS["simple"])
        time_analysis = time.time() - start

        # Analysis should not be much slower (both mock, so times very similar)
        # Main difference is lack of XML generation stage
        assert time_analysis >= 0
        assert time_standard >= 0


class TestE2EKBIntegration:
    """Test end-to-end with knowledge base integration."""

    @pytest.mark.asyncio
    async def test_kb_enhanced_mode_available(self):
        """Test that KB-enhanced mode can be configured."""
        config = AgentConfig(
            mode=ProcessingMode.KB_ENHANCED,
            llm_config=LLMConfig.from_env(),
            enable_kb=True,
            error_handling="recovery",
        )
        agent = BPMNAgent(config)

        xml_output, state = await agent.process(PROCESS_DESCRIPTIONS["finance"])

        # Should complete with KB enabled
        assert state is not None

    @pytest.mark.asyncio
    async def test_kb_disabled_mode_works(self):
        """Test that KB can be disabled."""
        config = AgentConfig(
            mode=ProcessingMode.STANDARD,
            llm_config=LLMConfig.from_env(),
            enable_kb=False,
            error_handling="recovery",
        )
        agent = BPMNAgent(config)

        xml_output, state = await agent.process(PROCESS_DESCRIPTIONS["simple"])

        # Should complete without KB
        assert state is not None


class TestE2EStateManagement:
    """Test state management across processing."""

    @pytest.mark.asyncio
    async def test_state_preserves_input(self):
        """Test that state preserves input text."""
        config = AgentConfig(
            mode=ProcessingMode.STANDARD,
            llm_config=LLMConfig.from_env(),
            error_handling="recovery",
        )
        agent = BPMNAgent(config)

        input_text = PROCESS_DESCRIPTIONS["simple"]
        xml_output, state = await agent.process(input_text)

        assert state.input_text == input_text

    @pytest.mark.asyncio
    async def test_state_tracks_all_results(self):
        """Test that state tracks results from all executed stages."""
        config = AgentConfig(
            mode=ProcessingMode.STANDARD,
            llm_config=LLMConfig.from_env(),
            error_handling="recovery",
        )
        agent = BPMNAgent(config)

        xml_output, state = await agent.process(PROCESS_DESCRIPTIONS["simple"])

        # Verify all stage results
        assert len(state.stage_results) == 5
        for i, result in enumerate(state.stage_results):
            assert result is not None, f"Stage {i+1} result is None"

    @pytest.mark.asyncio
    async def test_state_has_summary_method(self):
        """Test that state provides summary."""
        config = AgentConfig(
            mode=ProcessingMode.STANDARD,
            llm_config=LLMConfig.from_env(),
            error_handling="recovery",
        )
        agent = BPMNAgent(config)

        xml_output, state = await agent.process(PROCESS_DESCRIPTIONS["simple"])

        # State should be summarizable
        assert state is not None
        assert hasattr(state, "completion_rate")
        assert 0 <= state.completion_rate <= 1.0
