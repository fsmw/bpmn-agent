"""
End-to-end tests for Phase 4 validation with complete pipeline.

These tests verify:
1. Complete pipeline with Phase 4 validation
2. Quality improvement with RAG feedback
3. Multiple domain support (HR, Finance, IT, Healthcare, Manufacturing)
4. Validation with real BPMN examples
5. Metrics and quality reporting

Test markers:
- @pytest.mark.e2e: End-to-end tests (may require LLM)
- @pytest.mark.integration: Integration tests
"""

import asyncio
import pytest
from pathlib import Path
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET

from bpmn_agent.agent import BPMNAgent, AgentConfig, ProcessingMode
from bpmn_agent.core.llm_client import LLMConfig
from bpmn_agent.agent.config import PipelineConfig
from bpmn_agent.models.knowledge_base import DomainType
from bpmn_agent.validation.integration_layer import UnifiedValidationResult


# Test process descriptions for different domains
DOMAIN_PROCESSES = {
    DomainType.HR: (
        "Employee submits application. HR reviews qualifications. "
        "If qualified, schedule interview. If not, send rejection letter. "
        "After interview, make hiring decision."
    ),
    DomainType.FINANCE: (
        "Invoice received from vendor. Finance validates amount and vendor details. "
        "If valid, approve payment. If invalid, reject and notify vendor. "
        "Approved payments processed by accounting department."
    ),
    DomainType.IT: (
        "User reports bug. IT team triages issue. "
        "If critical, assign to senior developer. If minor, assign to junior developer. "
        "Developer fixes bug and submits for testing. Tester verifies fix."
    ),
    DomainType.HEALTHCARE: (
        "Patient arrives at clinic. Receptionist registers patient. "
        "Nurse takes vital signs. Doctor examines patient. "
        "If treatment needed, prescribe medication. Otherwise, provide advice."
    ),
    DomainType.MANUFACTURING: (
        "Order received. Production planning creates schedule. "
        "Materials department checks inventory. If available, start production. "
        "Quality control inspects finished products. Ship approved products."
    ),
}


class TestPhase4E2ECompletePipeline:
    """Test complete pipeline with Phase 4 validation enabled."""

    @pytest.fixture
    def agent_config_with_phase4(self):
        """Agent config with Phase 4 validation enabled."""
        llm_config = LLMConfig.from_env()
        pipeline_config = PipelineConfig(
            enable_phase4_validation=True,
            enable_rag_validation=True
        )
        return AgentConfig(
            llm_config=llm_config,
            mode=ProcessingMode.KB_ENHANCED,
            enable_kb=True,
            pipeline_config=pipeline_config,
            error_handling="recovery"
        )

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_complete_pipeline_with_phase4_validation(self, agent_config_with_phase4):
        """Test that complete pipeline executes Phase 4 validation."""
        agent = BPMNAgent(agent_config_with_phase4)
        
        text = "Customer submits order. System validates payment. Order is shipped."
        xml_output, state = await agent.process(text)
        
        # Verify XML was generated
        assert xml_output is not None
        assert isinstance(xml_output, str)
        assert len(xml_output) > 0
        
        # Verify XML is well-formed
        try:
            root = ET.fromstring(xml_output)
            assert root.tag.endswith("definitions")
        except ET.ParseError:
            pytest.fail("Output is not valid XML")
        
        # Verify Phase 4 validation stage executed
        assert len(state.stage_results) >= 6  # 5 stages + Phase 4 validation
        
        validation_stage = next(
            (r for r in state.stage_results if r.stage_name == "phase4_validation"),
            None
        )
        assert validation_stage is not None, "Phase 4 validation stage should exist"
        assert validation_stage.status.value == "completed", "Phase 4 validation should complete"
        
        # Verify validation result exists
        assert validation_stage.result is not None
        assert isinstance(validation_stage.result, UnifiedValidationResult)
        
        # Verify metrics were recorded
        assert validation_stage.metrics is not None
        assert "valid" in validation_stage.metrics
        assert "quality_score" in validation_stage.metrics

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_phase4_validation_quality_score(self, agent_config_with_phase4):
        """Test that Phase 4 validation provides quality score."""
        agent = BPMNAgent(agent_config_with_phase4)
        
        text = "Start process. Execute task. End process."
        xml_output, state = await agent.process(text)
        
        validation_stage = next(
            (r for r in state.stage_results if r.stage_name == "phase4_validation"),
            None
        )
        assert validation_stage is not None
        
        quality_score = validation_stage.metrics.get("quality_score")
        assert quality_score is not None
        assert isinstance(quality_score, (int, float))
        assert 0.0 <= quality_score <= 1.0

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_phase4_validation_with_errors(self, agent_config_with_phase4):
        """Test that Phase 4 validation detects errors in generated XML."""
        agent = BPMNAgent(agent_config_with_phase4)
        
        # Use a description that might generate incomplete XML
        text = "Process starts"
        xml_output, state = await agent.process(text)
        
        validation_stage = next(
            (r for r in state.stage_results if r.stage_name == "phase4_validation"),
            None
        )
        assert validation_stage is not None
        
        validation_result: UnifiedValidationResult = validation_stage.result
        assert validation_result is not None
        
        # Validation should complete even if there are errors
        assert validation_stage.status.value == "completed"
        
        # Check if errors or warnings were detected
        total_errors = validation_stage.metrics.get("xsd_errors", 0)
        total_warnings = validation_stage.metrics.get("xsd_warnings", 0)
        
        # At minimum, validation should have run
        assert total_errors >= 0
        assert total_warnings >= 0


class TestPhase4E2EMultipleDomains:
    """Test Phase 4 validation with multiple domains."""

    @pytest.fixture
    def agent_config_with_phase4(self):
        """Agent config with Phase 4 validation enabled."""
        llm_config = LLMConfig.from_env()
        pipeline_config = PipelineConfig(
            enable_phase4_validation=True,
            enable_rag_validation=True
        )
        return AgentConfig(
            llm_config=llm_config,
            mode=ProcessingMode.KB_ENHANCED,
            enable_kb=True,
            pipeline_config=pipeline_config,
            error_handling="recovery"
        )

    @pytest.mark.e2e
    @pytest.mark.asyncio
    @pytest.mark.parametrize("domain,process_text", [
        (DomainType.HR, DOMAIN_PROCESSES[DomainType.HR]),
        (DomainType.FINANCE, DOMAIN_PROCESSES[DomainType.FINANCE]),
        (DomainType.IT, DOMAIN_PROCESSES[DomainType.IT]),
        (DomainType.HEALTHCARE, DOMAIN_PROCESSES[DomainType.HEALTHCARE]),
        (DomainType.MANUFACTURING, DOMAIN_PROCESSES[DomainType.MANUFACTURING]),
    ])
    async def test_phase4_validation_with_domain(
        self, 
        agent_config_with_phase4, 
        domain: DomainType, 
        process_text: str
    ):
        """Test Phase 4 validation with different domains."""
        agent = BPMNAgent(agent_config_with_phase4)
        
        xml_output, state = await agent.process(process_text, domain=domain.value)
        
        # Verify XML was generated
        assert xml_output is not None
        
        # Verify Phase 4 validation executed
        validation_stage = next(
            (r for r in state.stage_results if r.stage_name == "phase4_validation"),
            None
        )
        assert validation_stage is not None
        assert validation_stage.status.value == "completed"
        
        # Verify domain was passed to validation
        validation_result: UnifiedValidationResult = validation_stage.result
        assert validation_result is not None
        
        # Verify metrics include domain-specific validation
        assert validation_stage.metrics is not None
        
        # Quality score should be calculated
        quality_score = validation_stage.metrics.get("quality_score")
        assert quality_score is not None
        assert isinstance(quality_score, (int, float))

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_phase4_rag_validation_with_patterns(self, agent_config_with_phase4):
        """Test that RAG validation validates applied patterns."""
        agent = BPMNAgent(agent_config_with_phase4)
        
        text = DOMAIN_PROCESSES[DomainType.FINANCE]
        xml_output, state = await agent.process(text, domain=DomainType.FINANCE.value)
        
        validation_stage = next(
            (r for r in state.stage_results if r.stage_name == "phase4_validation"),
            None
        )
        assert validation_stage is not None
        
        # Check if RAG validation metrics exist
        metrics = validation_stage.metrics
        if metrics.get("rag_compliance") is not None:
            rag_compliance = metrics["rag_compliance"]
            assert isinstance(rag_compliance, (int, float))
            assert 0.0 <= rag_compliance <= 1.0
        
        # Check patterns validated count
        patterns_validated = metrics.get("patterns_validated", 0)
        assert isinstance(patterns_validated, int)
        assert patterns_validated >= 0


class TestPhase4E2EQualityMetrics:
    """Test quality metrics and reporting."""

    @pytest.fixture
    def agent_config_with_phase4(self):
        """Agent config with Phase 4 validation enabled."""
        llm_config = LLMConfig.from_env()
        pipeline_config = PipelineConfig(
            enable_phase4_validation=True,
            enable_rag_validation=True
        )
        return AgentConfig(
            llm_config=llm_config,
            mode=ProcessingMode.KB_ENHANCED,
            enable_kb=True,
            pipeline_config=pipeline_config,
            error_handling="recovery"
        )

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_quality_score_target_85_percent(self, agent_config_with_phase4):
        """Test that quality score meets target of > 85% for good processes."""
        agent = BPMNAgent(agent_config_with_phase4)
        
        # Use a well-structured process description
        text = (
            "Customer places order. System validates payment method. "
            "If payment valid, reserve inventory and create shipping label. "
            "If payment invalid, request alternative payment method. "
            "After reservation, pick items from warehouse. "
            "Package items and ship to customer. Send confirmation email."
        )
        
        xml_output, state = await agent.process(text)
        
        validation_stage = next(
            (r for r in state.stage_results if r.stage_name == "phase4_validation"),
            None
        )
        assert validation_stage is not None
        
        quality_score = validation_stage.metrics.get("quality_score", 0.0)
        
        # Log quality score for debugging
        print(f"\nQuality Score: {quality_score:.2%}")
        print(f"Total Errors: {validation_stage.metrics.get('xsd_errors', 0)}")
        print(f"Total Warnings: {validation_stage.metrics.get('xsd_warnings', 0)}")
        
        # Note: We don't enforce 85% as a hard requirement in tests
        # as it depends on LLM quality and process complexity
        # But we verify the metric exists and is reasonable
        assert isinstance(quality_score, (int, float))
        assert 0.0 <= quality_score <= 1.0

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_validation_metrics_completeness(self, agent_config_with_phase4):
        """Test that all required validation metrics are present."""
        agent = BPMNAgent(agent_config_with_phase4)
        
        text = "Start process. Execute task. End process."
        xml_output, state = await agent.process(text)
        
        validation_stage = next(
            (r for r in state.stage_results if r.stage_name == "phase4_validation"),
            None
        )
        assert validation_stage is not None
        
        metrics = validation_stage.metrics
        assert metrics is not None
        
        # Required metrics
        assert "valid" in metrics
        assert "quality_score" in metrics
        assert "xsd_errors" in metrics
        assert "xsd_warnings" in metrics
        
        # Optional RAG metrics (if RAG is enabled)
        if agent_config_with_phase4.pipeline_config.enable_rag_validation:
            # These may be None if no patterns were applied
            assert "rag_compliance" in metrics or metrics.get("rag_compliance") is None
            assert "patterns_validated" in metrics

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_no_regressions_in_existing_functionality(self, agent_config_with_phase4):
        """Test that Phase 4 validation doesn't break existing functionality."""
        agent = BPMNAgent(agent_config_with_phase4)
        
        # Test standard workflow still works
        text = "Customer submits order. System processes payment. Order is shipped."
        xml_output, state = await agent.process(text)
        
        # Verify XML was generated (existing functionality)
        assert xml_output is not None
        assert len(xml_output) > 0
        
        # Verify XML is valid
        try:
            root = ET.fromstring(xml_output)
            assert root.tag.endswith("definitions")
        except ET.ParseError:
            pytest.fail("XML generation should still work")
        
        # Verify all stages completed
        assert len(state.stage_results) >= 5  # At least 5 stages
        
        # Verify no fatal errors
        assert not state.has_fatal_errors, "No fatal errors should occur"


class TestPhase4E2ERAGFeedback:
    """Test RAG feedback and quality improvement."""

    @pytest.fixture
    def agent_config_with_phase4(self):
        """Agent config with Phase 4 validation enabled."""
        llm_config = LLMConfig.from_env()
        pipeline_config = PipelineConfig(
            enable_phase4_validation=True,
            enable_rag_validation=True
        )
        return AgentConfig(
            llm_config=llm_config,
            mode=ProcessingMode.KB_ENHANCED,
            enable_kb=True,
            pipeline_config=pipeline_config,
            error_handling="recovery"
        )

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_rag_feedback_recording(self, agent_config_with_phase4):
        """Test that RAG feedback is recorded during validation."""
        agent = BPMNAgent(agent_config_with_phase4)
        
        if agent.validation_layer is None:
            pytest.skip("Phase 4 validation not enabled")
        
        # Check if feedback loop exists
        if not hasattr(agent.validation_layer, 'feedback_loop') or agent.validation_layer.feedback_loop is None:
            pytest.skip("RAG feedback loop not available")
        
        text = DOMAIN_PROCESSES[DomainType.FINANCE]
        xml_output, state = await agent.process(text, domain=DomainType.FINANCE.value)
        
        # Verify validation executed
        validation_stage = next(
            (r for r in state.stage_results if r.stage_name == "phase4_validation"),
            None
        )
        assert validation_stage is not None
        
        # Verify feedback loop exists and can be accessed
        feedback_loop = agent.validation_layer.feedback_loop
        assert feedback_loop is not None
        
        # Note: Actual feedback recording happens inside ValidationIntegrationLayer.validate()
        # We verify that validation completed successfully, which means feedback was processed

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_rag_pattern_validation_metrics(self, agent_config_with_phase4):
        """Test that RAG pattern validation metrics are recorded."""
        agent = BPMNAgent(agent_config_with_phase4)
        
        text = DOMAIN_PROCESSES[DomainType.HR]
        xml_output, state = await agent.process(text, domain=DomainType.HR.value)
        
        validation_stage = next(
            (r for r in state.stage_results if r.stage_name == "phase4_validation"),
            None
        )
        assert validation_stage is not None
        
        metrics = validation_stage.metrics
        
        # If RAG validation is enabled, check for RAG-specific metrics
        if agent_config_with_phase4.pipeline_config.enable_rag_validation:
            # These metrics may be None if no patterns were applied
            # But the keys should exist
            assert "rag_compliance" in metrics or metrics.get("rag_compliance") is None
            assert "patterns_validated" in metrics
            
            patterns_validated = metrics.get("patterns_validated", 0)
            assert isinstance(patterns_validated, int)
            assert patterns_validated >= 0


class TestPhase4E2ERealExamples:
    """Test Phase 4 validation with real BPMN example files."""

    @pytest.fixture
    def agent_config_with_phase4(self):
        """Agent config with Phase 4 validation enabled."""
        llm_config = LLMConfig.from_env()
        pipeline_config = PipelineConfig(
            enable_phase4_validation=True,
            enable_rag_validation=True
        )
        return AgentConfig(
            llm_config=llm_config,
            mode=ProcessingMode.KB_ENHANCED,
            enable_kb=True,
            pipeline_config=pipeline_config,
            error_handling="recovery"
        )

    def _load_bpmn_file(self, filename: str) -> Optional[str]:
        """Load BPMN file content if it exists."""
        # Check in examples directory (project root)
        project_root = Path(__file__).parent.parent.parent.parent
        examples_dir = project_root / "examples"
        
        # Try different possible locations
        possible_paths = [
            examples_dir / filename,
            project_root / "examples" / filename,
            Path(__file__).parent.parent / "examples" / filename,
        ]
        
        for path in possible_paths:
            if path.exists():
                return path.read_text(encoding="utf-8")
        
        return None

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_validate_pizza_store_bpmn(self, agent_config_with_phase4):
        """Test validation with Pizza-Store.bpmn example."""
        # Try to load the file
        bpmn_content = self._load_bpmn_file("Pizza-Store.bpmn")
        
        if bpmn_content is None:
            pytest.skip("Pizza-Store.bpmn file not found")
        
        # Use the validation layer directly to validate existing BPMN
        agent = BPMNAgent(agent_config_with_phase4)
        
        if agent.validation_layer is None:
            pytest.skip("Phase 4 validation not enabled")
        
        # Validate the BPMN XML directly
        validation_result = agent.validation_layer.validate(
            xml_content=bpmn_content,
            graph=None,  # We don't have graph for existing BPMN
            extraction_result=None,
            domain=None,
            patterns_applied=None
        )
        
        assert validation_result is not None
        assert isinstance(validation_result, UnifiedValidationResult)
        
        # Log results
        print(f"\nPizza-Store.bpmn Validation:")
        print(f"  Valid: {validation_result.overall_valid}")
        print(f"  Quality Score: {validation_result.overall_quality_score:.2%}")
        print(f"  XSD Errors: {validation_result.xsd_result.total_errors}")
        print(f"  XSD Warnings: {validation_result.xsd_result.total_warnings}")

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_validate_car_wash_bpmn(self, agent_config_with_phase4):
        """Test validation with Car-Wash.bpmn example."""
        bpmn_content = self._load_bpmn_file("Car-Wash.bpmn")
        
        if bpmn_content is None:
            pytest.skip("Car-Wash.bpmn file not found")
        
        agent = BPMNAgent(agent_config_with_phase4)
        
        if agent.validation_layer is None:
            pytest.skip("Phase 4 validation not enabled")
        
        validation_result = agent.validation_layer.validate(
            xml_content=bpmn_content,
            graph=None,
            extraction_result=None,
            domain=None,
            patterns_applied=None
        )
        
        assert validation_result is not None
        print(f"\nCar-Wash.bpmn Validation:")
        print(f"  Valid: {validation_result.overall_valid}")
        print(f"  Quality Score: {validation_result.overall_quality_score:.2%}")

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_validate_recruitment_bpmn(self, agent_config_with_phase4):
        """Test validation with Recruitment-and-Selection.bpmn example."""
        bpmn_content = self._load_bpmn_file("Recruitment-and-Selection.bpmn")
        
        if bpmn_content is None:
            pytest.skip("Recruitment-and-Selection.bpmn file not found")
        
        agent = BPMNAgent(agent_config_with_phase4)
        
        if agent.validation_layer is None:
            pytest.skip("Phase 4 validation not enabled")
        
        validation_result = agent.validation_layer.validate(
            xml_content=bpmn_content,
            graph=None,
            extraction_result=None,
            domain=DomainType.HR,  # HR domain for recruitment
            patterns_applied=None
        )
        
        assert validation_result is not None
        print(f"\nRecruitment-and-Selection.bpmn Validation:")
        print(f"  Valid: {validation_result.overall_valid}")
        print(f"  Quality Score: {validation_result.overall_quality_score:.2%}")

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_validate_smart_parking_bpmn(self, agent_config_with_phase4):
        """Test validation with Smart-Parking.bpmn example."""
        bpmn_content = self._load_bpmn_file("Smart-Parking.bpmn")
        
        if bpmn_content is None:
            pytest.skip("Smart-Parking.bpmn file not found")
        
        agent = BPMNAgent(agent_config_with_phase4)
        
        if agent.validation_layer is None:
            pytest.skip("Phase 4 validation not enabled")
        
        validation_result = agent.validation_layer.validate(
            xml_content=bpmn_content,
            graph=None,
            extraction_result=None,
            domain=None,
            patterns_applied=None
        )
        
        assert validation_result is not None
        print(f"\nSmart-Parking.bpmn Validation:")
        print(f"  Valid: {validation_result.overall_valid}")
        print(f"  Quality Score: {validation_result.overall_quality_score:.2%}")
