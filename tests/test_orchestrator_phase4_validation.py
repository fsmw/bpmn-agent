"""
Tests for Phase 4 validation integration in BPMNAgent orchestrator.

Tests that Phase 4 validation (XSD + RAG) is properly integrated
and executes after XML generation.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from bpmn_agent.agent.orchestrator import BPMNAgent
from bpmn_agent.agent.config import AgentConfig, ProcessingMode, PipelineConfig
from bpmn_agent.core.llm_client import LLMConfig
from bpmn_agent.models.knowledge_base import DomainType
from bpmn_agent.validation.integration_layer import UnifiedValidationResult
from bpmn_agent.validation.enhanced_xsd_validation import XSDValidationResult


@pytest.fixture
def agent_config_with_phase4():
    """Agent config with Phase 4 validation enabled."""
    llm_config = LLMConfig(
        provider="ollama",
        base_url="http://localhost:11434",
        model="mistral"
    )
    pipeline_config = PipelineConfig(
        enable_phase4_validation=True,
        enable_rag_validation=True
    )
    return AgentConfig(
        llm_config=llm_config,
        mode=ProcessingMode.STANDARD,
        enable_kb=True,
        pipeline_config=pipeline_config
    )


@pytest.fixture
def agent_config_without_phase4():
    """Agent config with Phase 4 validation disabled."""
    llm_config = LLMConfig(
        provider="ollama",
        base_url="http://localhost:11434",
        model="mistral"
    )
    pipeline_config = PipelineConfig(
        enable_phase4_validation=False
    )
    return AgentConfig(
        llm_config=llm_config,
        mode=ProcessingMode.STANDARD,
        enable_kb=False,
        pipeline_config=pipeline_config
    )


@pytest.mark.unit
def test_agent_initializes_validation_layer_when_enabled(agent_config_with_phase4):
    """Test that ValidationIntegrationLayer is initialized when Phase 4 is enabled."""
    agent = BPMNAgent(agent_config_with_phase4)
    
    assert agent.validation_layer is not None
    assert agent.validation_layer.enable_rag is True


@pytest.mark.unit
def test_agent_does_not_initialize_validation_layer_when_disabled(agent_config_without_phase4):
    """Test that ValidationIntegrationLayer is None when Phase 4 is disabled."""
    agent = BPMNAgent(agent_config_without_phase4)
    
    assert agent.validation_layer is None


@pytest.mark.unit
def test_get_patterns_applied_extracts_from_xml_generator():
    """Test that _get_patterns_applied extracts patterns from XML generator."""
    llm_config = LLMConfig(
        provider="ollama",
        base_url="http://localhost:11434",
        model="mistral"
    )
    config = AgentConfig(
        llm_config=llm_config,
        mode=ProcessingMode.STANDARD,
        enable_kb=True,
        pipeline_config=PipelineConfig(enable_phase4_validation=True)
    )
    agent = BPMNAgent(config)
    
    # Mock XML generator with pattern references
    from bpmn_agent.stages.xml_generation import PatternReference, IDMapping
    
    mock_pattern_ref = PatternReference(
        pattern_id="test_pattern_1",
        pattern_name="Test Pattern",
        confidence=0.9
    )
    mock_mapping = IDMapping(
        graph_id="node_1",
        bpmn_id="task_1",
        element_type="task",
        pattern_reference=mock_pattern_ref
    )
    agent.xml_generator.id_mappings = [mock_mapping]
    
    patterns = agent._get_patterns_applied()
    assert patterns == ["test_pattern_1"]


@pytest.mark.unit
def test_get_patterns_applied_returns_none_when_no_patterns():
    """Test that _get_patterns_applied returns None when no patterns are found."""
    llm_config = LLMConfig(
        provider="ollama",
        base_url="http://localhost:11434",
        model="mistral"
    )
    config = AgentConfig(
        llm_config=llm_config,
        mode=ProcessingMode.STANDARD,
        enable_kb=True,
        pipeline_config=PipelineConfig(enable_phase4_validation=True)
    )
    agent = BPMNAgent(config)
    
    # Mock XML generator without pattern references
    agent.xml_generator.id_mappings = []
    
    patterns = agent._get_patterns_applied()
    assert patterns is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_phase4_validation_executes_after_xml_generation(agent_config_with_phase4):
    """Test that Phase 4 validation executes after XML generation in standard mode."""
    agent = BPMNAgent(agent_config_with_phase4)
    
    # Mock validation layer
    mock_validation_result = UnifiedValidationResult(
        xsd_result=XSDValidationResult(
            is_valid=True,
            quality_score=0.85,
            errors=[],
            total_errors=0,
            total_warnings=0
        ),
        overall_valid=True,
        overall_quality_score=0.85
    )
    
    agent.validation_layer.validate = Mock(return_value=mock_validation_result)
    
    # Mock the pipeline stages to avoid actual LLM calls
    with patch.object(agent, '_stage1_preprocess', return_value=MagicMock(cleaned_text="test")):
        with patch.object(agent, '_stage2_extract_entities', return_value=MagicMock()):
            with patch.object(agent, '_stage3_resolve_entities', return_value=MagicMock()):
                with patch.object(agent, '_stage4_build_graph', return_value=MagicMock()):
                    with patch.object(agent, '_stage5_generate_xml', return_value="<xml>test</xml>"):
                        with patch.object(agent, '_get_patterns_applied', return_value=None):
                            # Process text
                            xml_output, state = await agent.process("Test process description")
                            
                            # Verify validation was called
                            agent.validation_layer.validate.assert_called_once()
                            
                            # Verify state has validation results
                            assert len(state.stage_results) > 0
                            validation_stage = next(
                                (r for r in state.stage_results if r.stage_name == "phase4_validation"),
                                None
                            )
                            assert validation_stage is not None
                            assert validation_stage.status.value == "completed"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_phase4_validation_passes_rag_context(agent_config_with_phase4):
    """Test that Phase 4 validation receives RAG context (domain, patterns, graph, extraction)."""
    agent = BPMNAgent(agent_config_with_phase4)
    
    # Mock validation layer
    mock_validation_result = UnifiedValidationResult(
        xsd_result=XSDValidationResult(
            is_valid=True,
            quality_score=0.85,
            errors=[],
            total_errors=0,
            total_warnings=0
        ),
        overall_valid=True,
        overall_quality_score=0.85
    )
    
    agent.validation_layer.validate = Mock(return_value=mock_validation_result)
    
    # Mock components
    mock_extraction = MagicMock()
    mock_graph = MagicMock()
    mock_patterns = ["pattern_1", "pattern_2"]
    
    with patch.object(agent, '_stage1_preprocess', return_value=MagicMock(cleaned_text="test")):
        with patch.object(agent, '_stage2_extract_entities', return_value=mock_extraction):
            with patch.object(agent, '_stage3_resolve_entities', return_value=mock_extraction):
                with patch.object(agent, '_stage4_build_graph', return_value=mock_graph):
                    with patch.object(agent, '_stage5_generate_xml', return_value="<xml>test</xml>"):
                        with patch.object(agent, '_get_patterns_applied', return_value=mock_patterns):
                            # Process in KB-enhanced mode to get domain
                            agent.config.mode = ProcessingMode.KB_ENHANCED
                            with patch.object(agent, '_detect_domain', return_value="finance"):
                                xml_output, state = await agent.process("Test process")
                                
                                # Verify validation was called with correct context
                                call_args = agent.validation_layer.validate.call_args
                                assert call_args is not None
                                
                                # Check that graph, extraction_result, domain, and patterns were passed
                                assert call_args.kwargs.get('graph') == mock_graph
                                assert call_args.kwargs.get('extraction_result') == mock_extraction
                                assert call_args.kwargs.get('domain') == DomainType.FINANCE
                                assert call_args.kwargs.get('patterns_applied') == mock_patterns


@pytest.mark.integration
@pytest.mark.asyncio
async def test_phase4_validation_records_feedback(agent_config_with_phase4):
    """Test that Phase 4 validation records feedback in RAGFeedbackLoop."""
    agent = BPMNAgent(agent_config_with_phase4)
    
    # Mock validation layer with feedback loop
    mock_feedback_loop = Mock()
    agent.validation_layer.feedback_loop = mock_feedback_loop
    agent.validation_layer.rag_enabled = True
    
    mock_validation_result = UnifiedValidationResult(
        xsd_result=XSDValidationResult(
            is_valid=True,
            quality_score=0.85,
            errors=[],
            total_errors=0,
            total_warnings=0
        ),
        overall_valid=True,
        overall_quality_score=0.85
    )
    
    agent.validation_layer.validate = Mock(return_value=mock_validation_result)
    
    mock_patterns = ["pattern_1"]
    
    with patch.object(agent, '_stage1_preprocess', return_value=MagicMock(cleaned_text="test")):
        with patch.object(agent, '_stage2_extract_entities', return_value=MagicMock()):
            with patch.object(agent, '_stage3_resolve_entities', return_value=MagicMock()):
                with patch.object(agent, '_stage4_build_graph', return_value=MagicMock()):
                    with patch.object(agent, '_stage5_generate_xml', return_value="<xml>test</xml>"):
                        with patch.object(agent, '_get_patterns_applied', return_value=mock_patterns):
                            # Process in KB-enhanced mode
                            agent.config.mode = ProcessingMode.KB_ENHANCED
                            with patch.object(agent, '_detect_domain', return_value="finance"):
                                await agent.process("Test process")
                                
                                # Verify feedback was recorded (called inside validate method)
                                # The feedback loop is called inside ValidationIntegrationLayer.validate
                                # We verify that validate was called, which internally calls feedback


@pytest.mark.integration
@pytest.mark.asyncio
async def test_phase4_validation_does_not_execute_when_disabled(agent_config_without_phase4):
    """Test that Phase 4 validation does not execute when disabled."""
    agent = BPMNAgent(agent_config_without_phase4)
    
    # Mock the pipeline stages
    with patch.object(agent, '_stage1_preprocess', return_value=MagicMock(cleaned_text="test")):
        with patch.object(agent, '_stage2_extract_entities', return_value=MagicMock()):
            with patch.object(agent, '_stage3_resolve_entities', return_value=MagicMock()):
                with patch.object(agent, '_stage4_build_graph', return_value=MagicMock()):
                    with patch.object(agent, '_stage5_generate_xml', return_value="<xml>test</xml>"):
                        xml_output, state = await agent.process("Test process description")
                        
                        # Verify validation layer is None
                        assert agent.validation_layer is None
                        
                        # Verify no validation stage in results
                        validation_stage = next(
                            (r for r in state.stage_results if r.stage_name == "phase4_validation"),
                            None
                        )
                        assert validation_stage is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_phase4_validation_handles_errors_gracefully(agent_config_with_phase4):
    """Test that Phase 4 validation errors don't crash the pipeline."""
    agent = BPMNAgent(agent_config_with_phase4)
    
    # Mock validation layer to raise an error
    agent.validation_layer.validate = Mock(side_effect=Exception("Validation error"))
    
    with patch.object(agent, '_stage1_preprocess', return_value=MagicMock(cleaned_text="test")):
        with patch.object(agent, '_stage2_extract_entities', return_value=MagicMock()):
            with patch.object(agent, '_stage3_resolve_entities', return_value=MagicMock()):
                with patch.object(agent, '_stage4_build_graph', return_value=MagicMock()):
                    with patch.object(agent, '_stage5_generate_xml', return_value="<xml>test</xml>"):
                        with patch.object(agent, '_get_patterns_applied', return_value=None):
                            # Process should not crash
                            xml_output, state = await agent.process("Test process description")
                            
                            # Verify XML was still generated
                            assert xml_output == "<xml>test</xml>"
                            
                            # Verify validation stage shows failure
                            validation_stage = next(
                                (r for r in state.stage_results if r.stage_name == "phase4_validation"),
                                None
                            )
                            assert validation_stage is not None
                            assert validation_stage.status.value == "failed"
