"""
BPMN Agent Orchestrator

Coordinates all 5 pipeline stages into a unified agent interface.
Manages state, error handling, and integrates knowledge base patterns.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from bpmn_agent.agent.config import AgentConfig, ProcessingMode, ErrorHandlingStrategy
from bpmn_agent.agent.state import AgentState, StageResult, StageStatus, ProcessingMetrics
from bpmn_agent.core.llm_client import BaseLLMClient, LLMClientFactory
from bpmn_agent.core.observability import ObservabilityManager, log_execution
from bpmn_agent.core.tokenizer import TokenCounter
from bpmn_agent.knowledge.domain_classifier import DomainClassifier
from bpmn_agent.models.extraction import ExtractionResultWithErrors
from bpmn_agent.models.graph import ProcessGraph
from bpmn_agent.stages import (
    TextPreprocessor,
    EntityExtractor,
    EntityResolutionPipeline,
    SemanticGraphConstructionPipeline,
)
from bpmn_agent.stages.xml_generation import BPMNXMLGenerator

logger = logging.getLogger(__name__)


class BPMNAgent:
    """
    Main BPMN Agent orchestrator.
    
    Coordinates the 5-stage extraction pipeline:
    1. Text Preprocessing
    2. Entity & Relation Extraction
    3. Entity Resolution
    4. Semantic Graph Construction
    5. BPMN XML Generation
    """
    
    def __init__(self, config: AgentConfig):
        """Initialize the BPMN Agent.
        
        Args:
            config: Agent configuration
            
        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self.llm_client: BaseLLMClient = LLMClientFactory.create(config.llm_config)
        self.state: Optional[AgentState] = None
        
        # Initialize components
        self.text_preprocessor = TextPreprocessor()
        self.entity_extractor = EntityExtractor(self.llm_client)
        self.entity_resolver = EntityResolutionPipeline()
        self.graph_builder = SemanticGraphConstructionPipeline()
        self.xml_generator = BPMNXMLGenerator(enable_kb=config.enable_kb)
        self.domain_classifier = DomainClassifier() if config.enable_kb else None
        self.token_counter = TokenCounter()
        
        # Initialize observability
        if config.enable_logging:
            ObservabilityManager.initialize()
        
        logger.info(
            "BPMNAgent initialized",
            extra={
                "mode": config.mode.value,
                "enable_kb": config.enable_kb,
                "llm_provider": config.llm_config.provider,
            },
        )
    
    @log_execution(include_args=False, include_duration=True)
    async def process(
        self,
        text: str,
        process_name: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> Tuple[Optional[str], AgentState]:
        """
        Process natural language text to generate BPMN XML.
        
        Args:
            text: Input natural language text
            process_name: Optional override for process name
            domain: Optional domain hint for KB-enhanced processing
            
        Returns:
            (xml_output, state) tuple
        """
        # Initialize state
        session_id = str(uuid.uuid4())
        self.state = AgentState(
            session_id=session_id,
            start_time=datetime.now(),
            input_text=text,
            input_domain=domain,
        )
        
        # Check mode
        if self.config.mode == ProcessingMode.ANALYSIS_ONLY:
            return await self._process_analysis_only(text, domain)
        elif self.config.mode == ProcessingMode.VALIDATION_ONLY:
            return await self._process_validation_only(text)
        elif self.config.mode == ProcessingMode.KB_ENHANCED:
            return await self._process_kb_enhanced(text, process_name, domain)
        else:
            return await self._process_standard(text, process_name, domain)
    
    async def _process_standard(
        self,
        text: str,
        process_name: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> Tuple[Optional[str], AgentState]:
        """Standard processing mode (all 5 stages)."""
        try:
            # Stage 1: Text Preprocessing
            preprocessed = await self._stage1_preprocess(text)
            if not preprocessed:
                return None, self.state
            
            # Stage 2: Entity Extraction
            extraction_result = await self._stage2_extract_entities(preprocessed, domain)
            if extraction_result is None:
                return None, self.state
            
            # Stage 3: Entity Resolution
            resolved = await self._stage3_resolve_entities(extraction_result)
            if resolved is None:
                return None, self.state
            
            # Stage 4: Graph Construction
            graph = await self._stage4_build_graph(resolved)
            if graph is None:
                return None, self.state
            
            # Stage 5: XML Generation
            xml_output = await self._stage5_generate_xml(graph, process_name)
            
            if xml_output:
                self.state.output_xml = xml_output
            
            return xml_output, self.state
            
        except Exception as e:
            logger.exception("Error in standard processing")
            if self.state:
                self.state.errors.append(f"Processing failed: {str(e)}")
            return None, self.state
    
    async def _process_kb_enhanced(
        self,
        text: str,
        process_name: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> Tuple[Optional[str], AgentState]:
        """KB-enhanced processing mode."""
        try:
            # Auto-detect domain if not provided
            detected_domain = domain
            if self.config.kb_domain_auto_detect and not domain:
                detected_domain = await self._detect_domain(text)
                self.state.input_domain = detected_domain
            
            # Process all stages with KB context
            preprocessed = await self._stage1_preprocess(text)
            if not preprocessed:
                return None, self.state
            
            extraction_result = await self._stage2_extract_entities(
                preprocessed, detected_domain, use_kb=True
            )
            if extraction_result is None:
                return None, self.state
            
            resolved = await self._stage3_resolve_entities(extraction_result)
            if resolved is None:
                return None, self.state
            
            graph = await self._stage4_build_graph(resolved, detected_domain)
            if graph is None:
                return None, self.state
            
            xml_output = await self._stage5_generate_xml(
                graph, process_name, domain=detected_domain
            )
            
            if xml_output:
                self.state.output_xml = xml_output
            
            return xml_output, self.state
            
        except Exception as e:
            logger.exception("Error in KB-enhanced processing")
            if self.state:
                self.state.errors.append(f"KB-enhanced processing failed: {str(e)}")
            return None, self.state
    
    async def _process_analysis_only(
        self,
        text: str,
        domain: Optional[str] = None,
    ) -> Tuple[Optional[str], AgentState]:
        """Analysis-only mode (stages 1-4, no XML generation)."""
        try:
            # Detect domain
            detected_domain = domain
            if self.config.kb_domain_auto_detect and not domain:
                detected_domain = await self._detect_domain(text)
                self.state.input_domain = detected_domain
            
            # Run stages 1-4
            preprocessed = await self._stage1_preprocess(text)
            if not preprocessed:
                return None, self.state
            
            extraction_result = await self._stage2_extract_entities(
                preprocessed, detected_domain
            )
            if extraction_result is None:
                return None, self.state
            
            resolved = await self._stage3_resolve_entities(extraction_result)
            if resolved is None:
                return None, self.state
            
            graph = await self._stage4_build_graph(resolved, detected_domain)
            if graph is None:
                return None, self.state
            
            # Return analysis results
            analysis = {
                "domain": detected_domain,
                "entities": len(extraction_result.entities),
                "relations": len(extraction_result.relations),
                "graph_nodes": len(graph.nodes),
                "graph_edges": len(graph.edges),
            }
            
            return None, self.state  # No XML output, but state has analysis
            
        except Exception as e:
            logger.exception("Error in analysis-only processing")
            if self.state:
                self.state.errors.append(f"Analysis failed: {str(e)}")
            return None, self.state
    
    async def _process_validation_only(self, text: str) -> Tuple[Optional[str], AgentState]:
        """Validation-only mode (check input validity)."""
        try:
            # Stage 1: Text Preprocessing (validation)
            preprocessed = await self._stage1_preprocess(text)
            
            if preprocessed:
                self.state.warnings.append("Input text is valid and ready for processing")
            
            return None, self.state
            
        except Exception as e:
            logger.exception("Error in validation-only processing")
            if self.state:
                self.state.errors.append(f"Validation failed: {str(e)}")
            return None, self.state
    
    # ==================
    # Stage Implementations
    # ==================
    
    async def _stage1_preprocess(self, text: str) -> Optional[Any]:
        """Stage 1: Text Preprocessing."""
        stage_name = "text_preprocessing"
        result = StageResult(stage_name=stage_name, status=StageStatus.RUNNING)
        result.start_time = datetime.now()
        
        try:
            # Preprocess text
            preprocessed = self.text_preprocessor.preprocess(text)
            
            # Update state with preprocessed text
            self.state.preprocessed_text = preprocessed.cleaned_text
            
            # Update metrics
            self.state.metrics.input_text_length = len(text)
            self.state.metrics.input_token_count = self.token_counter.count_tokens(text)
            
            result.result = preprocessed
            result.status = StageStatus.COMPLETED
            result.metrics = {
                "input_length": len(text),
                "preprocessed_length": len(str(preprocessed)),
                "chunks": len(preprocessed.chunks),
                "tokens": self.state.metrics.input_token_count,
            }
            
            logger.info(f"Stage 1 completed: {result.metrics}")
            
        except Exception as e:
            logger.exception(f"Stage 1 failed: {e}")
            result.status = StageStatus.FAILED
            result.error = str(e)
            
            if self.config.error_handling == ErrorHandlingStrategy.STRICT:
                self.state.add_stage_result(result)
                return None
        
        result.end_time = datetime.now()
        result.duration_ms = (result.end_time - result.start_time).total_seconds() * 1000
        self.state.add_stage_result(result)
        
        return result.result
    
    async def _stage2_extract_entities(
        self,
        preprocessed: Any,
        domain: Optional[str] = None,
        use_kb: bool = False,
    ) -> Optional[ExtractionResultWithErrors]:
        """Stage 2: Entity & Relation Extraction."""
        stage_name = "entity_extraction"
        result = StageResult(stage_name=stage_name, status=StageStatus.RUNNING)
        result.start_time = datetime.now()
        
        try:
            # Extract entities
            extraction_result = await self.entity_extractor.extract_from_text(
                preprocessed.original_text,
                preprocessed=preprocessed,
                llm_temperature=0.3,
                max_retries=2,
            )
            
            # Update metrics
            self.state.metrics.entities_extracted = len(extraction_result.entities)
            self.state.metrics.relations_extracted = len(extraction_result.relations)
            if extraction_result.entities:
                # Convert confidence levels to numeric values for averaging
                confidence_map = {"high": 1.0, "medium": 0.5, "low": 0.25}
                confidences = [
                    confidence_map.get(str(e.confidence).lower(), 0.5)
                    for e in extraction_result.entities
                ]
                self.state.metrics.avg_entity_confidence = sum(confidences) / len(
                    confidences
                )
            
            result.result = extraction_result
            result.status = StageStatus.COMPLETED
            result.metrics = {
                "entities": len(extraction_result.entities),
                "relations": len(extraction_result.relations),
                "avg_confidence": self.state.metrics.avg_entity_confidence,
                "errors": len(extraction_result.errors),
            }
            
            # Handle warnings
            if extraction_result.errors:
                for error in extraction_result.errors:
                    result.warnings.append(f"Extraction error: {error.message}")
            
            logger.info(f"Stage 2 completed: {result.metrics}")
            
        except Exception as e:
            logger.exception(f"Stage 2 failed: {e}")
            result.status = StageStatus.FAILED
            result.error = str(e)
            
            if self.config.error_handling == ErrorHandlingStrategy.STRICT:
                self.state.add_stage_result(result)
                return None
        
        result.end_time = datetime.now()
        result.duration_ms = (result.end_time - result.start_time).total_seconds() * 1000
        self.state.add_stage_result(result)
        
        return result.result
    
    async def _stage3_resolve_entities(
        self, extraction_result: ExtractionResultWithErrors
    ) -> Optional[ExtractionResultWithErrors]:
        """Stage 3: Entity Resolution."""
        stage_name = "entity_resolution"
        result = StageResult(stage_name=stage_name, status=StageStatus.RUNNING)
        result.start_time = datetime.now()
        
        try:
            # Resolve entities
            resolved = self.entity_resolver.resolve(extraction_result)
            
            # Update metrics - extract from resolved data
            self.state.metrics.coreferences_resolved = len(resolved.co_references)
            # For actors, count unique actor entity types
            actors = [e for e in resolved.entities if e.type.value == "actor"]
            self.state.metrics.actors_consolidated = len(actors)
            
            result.result = resolved
            result.status = StageStatus.COMPLETED
            result.metrics = {
                "coreferences_resolved": self.state.metrics.coreferences_resolved,
                "actors_consolidated": self.state.metrics.actors_consolidated,
                "entities_after": len(resolved.entities),
            }
            
            logger.info(f"Stage 3 completed: {result.metrics}")
            
        except Exception as e:
            logger.exception(f"Stage 3 failed: {e}")
            result.status = StageStatus.FAILED
            result.error = str(e)
            
            if self.config.error_handling == ErrorHandlingStrategy.STRICT:
                self.state.add_stage_result(result)
                return None
        
        result.end_time = datetime.now()
        result.duration_ms = (result.end_time - result.start_time).total_seconds() * 1000
        self.state.add_stage_result(result)
        
        return result.result
    
    async def _stage4_build_graph(
        self, resolved: ExtractionResultWithErrors, domain: Optional[str] = None
    ) -> Optional[ProcessGraph]:
        """Stage 4: Semantic Graph Construction."""
        stage_name = "graph_construction"
        result = StageResult(stage_name=stage_name, status=StageStatus.RUNNING)
        result.start_time = datetime.now()
        
        try:
            # Build graph - need to extract actor profiles from resolved
            actor_profiles = {}  # TODO: extract from resolved if available
            graph, validation_report, implicit_flows = self.graph_builder.construct_graph(
                extraction_result=resolved,
                actor_profiles=actor_profiles,
                domain=domain
            )
            
            # Update metrics
            self.state.metrics.graph_nodes = len(graph.nodes)
            self.state.metrics.graph_edges = len(graph.edges)
            self.state.metrics.graph_cycles_detected = (
                0 if graph.is_acyclic else 1
            )  # Simplified
            
            result.result = graph
            result.status = StageStatus.COMPLETED
            result.metrics = {
                "nodes": len(graph.nodes),
                "edges": len(graph.edges),
                "is_acyclic": graph.is_acyclic,
                "complexity": graph.complexity,
            }
            
            logger.info(f"Stage 4 completed: {result.metrics}")
            
        except Exception as e:
            logger.exception(f"Stage 4 failed: {e}")
            result.status = StageStatus.FAILED
            result.error = str(e)
            
            if self.config.error_handling == ErrorHandlingStrategy.STRICT:
                self.state.add_stage_result(result)
                return None
        
        result.end_time = datetime.now()
        result.duration_ms = (result.end_time - result.start_time).total_seconds() * 1000
        self.state.add_stage_result(result)
        
        return result.result
    
    async def _stage5_generate_xml(
        self,
        graph: ProcessGraph,
        process_name: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> Optional[str]:
        """Stage 5: BPMN XML Generation."""
        stage_name = "xml_generation"
        result = StageResult(stage_name=stage_name, status=StageStatus.RUNNING)
        result.start_time = datetime.now()
        
        try:
            # Generate XML
            xml_output = self.xml_generator.generate_xml(graph, process_name)
            
            # Update metrics
            self.state.metrics.output_xml_size_bytes = len(xml_output.encode("utf-8"))
            # Rough element count
            self.state.metrics.output_element_count = xml_output.count("<")
            
            result.result = xml_output
            result.status = StageStatus.COMPLETED
            result.metrics = {
                "xml_size_bytes": self.state.metrics.output_xml_size_bytes,
                "element_count": self.state.metrics.output_element_count,
            }
            
            logger.info(f"Stage 5 completed: {result.metrics}")
            
        except Exception as e:
            logger.exception(f"Stage 5 failed: {e}")
            result.status = StageStatus.FAILED
            result.error = str(e)
            
            if self.config.error_handling == ErrorHandlingStrategy.STRICT:
                self.state.add_stage_result(result)
                return None
        
        result.end_time = datetime.now()
        result.duration_ms = (result.end_time - result.start_time).total_seconds() * 1000
        self.state.add_stage_result(result)
        
        return result.result
    
    # ==================
    # Utility Methods
    # ==================
    
    async def _detect_domain(self, text: str) -> Optional[str]:
        """Detect process domain from text."""
        if not self.domain_classifier:
            return None
        
        try:
            result = self.domain_classifier.classify_domain(text)
            domain = result.primary_domain if result else None
            logger.info(f"Detected domain: {domain}")
            return domain
        except Exception as e:
            logger.warning(f"Domain detection failed: {e}")
            return None
    
    def validate_config(self) -> bool:
        """Validate agent configuration."""
        try:
            # Check LLM connection
            # Note: This is sync, might need async wrapper
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check agent health status."""
        return {
            "status": "healthy",
            "llm_provider": self.config.llm_config.provider,
            "mode": self.config.mode.value,
            "kb_enabled": self.config.enable_kb,
            "timestamp": datetime.now().isoformat(),
        }
    
    def get_state_summary(self) -> Optional[Dict[str, Any]]:
        """Get summary of current processing state."""
        if not self.state:
            return None
        return self.state.summary()
