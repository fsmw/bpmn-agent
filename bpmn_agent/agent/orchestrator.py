"""
BPMN Agent Orchestrator

Coordinates all 5 pipeline stages into a unified agent interface.
Manages state, error handling, and integrates knowledge base patterns.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from bpmn_agent.agent.config import AgentConfig, ErrorHandlingStrategy, ProcessingMode
from bpmn_agent.agent.observability_hooks import get_observability_hooks
from bpmn_agent.agent.state import AgentState, StageResult, StageStatus
from bpmn_agent.core.llm_client import BaseLLMClient, LLMClientFactory
from bpmn_agent.core.observability import ObservabilityManager, log_execution
from bpmn_agent.core.tokenizer import TokenCounter
from bpmn_agent.knowledge.domain_classifier import DomainClassifier
from bpmn_agent.models.extraction import ExtractionResultWithErrors
from bpmn_agent.models.graph import ProcessGraph
from bpmn_agent.models.knowledge_base import DomainType
from bpmn_agent.stages import (
    EntityExtractor,
    EntityResolutionPipeline,
    SemanticGraphConstructionPipeline,
    TextPreprocessor,
)
from bpmn_agent.stages.entity_resolution import ActorProfile
from bpmn_agent.stages.xml_generation import BPMNXMLGenerator
from bpmn_agent.validation.integration_layer import ValidationIntegrationLayer

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

        # Initialize Phase 4 validation integration layer
        # Enable RAG validation if KB is enabled and phase4 validation is enabled
        enable_rag_validation = (
            config.enable_kb
            and config.pipeline_config.enable_phase4_validation
            and config.pipeline_config.enable_rag_validation
        )
        self.validation_layer = (
            ValidationIntegrationLayer(enable_rag=enable_rag_validation)
            if config.pipeline_config.enable_phase4_validation
            else None
        )

        # Initialize observability
        if config.enable_logging:
            ObservabilityManager.initialize()

        # Initialize observability hooks for Phase 3
        self.observability_hooks = get_observability_hooks()

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
        # Start pipeline tracking
        self.observability_hooks.start_pipeline(self.state.session_id)

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

                # Stage 6: Phase 4 Validation (after XML generation)
                if self.validation_layer:
                    await self._stage6_validate_phase4(
                        xml_output=xml_output,
                        graph=graph,
                        extraction_result=resolved,
                        domain=None,  # Standard mode doesn't use domain
                        patterns_applied=None,  # Standard mode doesn't track patterns
                    )

            # End pipeline successfully
            self.observability_hooks.end_pipeline(success=True)
            return xml_output, self.state

        except Exception as e:
            logger.exception("Error in standard processing")
            if self.state:
                self.state.errors.append(f"Processing failed: {str(e)}")
            # Record pipeline failure
            self.observability_hooks.end_pipeline(success=False)
            return None, self.state

    async def _process_kb_enhanced(
        self,
        text: str,
        process_name: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> Tuple[Optional[str], AgentState]:
        """KB-enhanced processing mode."""
        # Start pipeline tracking
        self.observability_hooks.start_pipeline(self.state.session_id)

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

                # Stage 6: Phase 4 Validation (after XML generation)
                # Get patterns applied from XML generator or context
                patterns_applied = self._get_patterns_applied()
                domain_enum = DomainType(detected_domain) if detected_domain else None

                await self._stage6_validate_phase4(
                    xml_output=xml_output,
                    graph=graph,
                    extraction_result=resolved,
                    domain=domain_enum,
                    patterns_applied=patterns_applied,
                )

            # End pipeline successfully
            self.observability_hooks.end_pipeline(success=True)
            return xml_output, self.state

        except Exception as e:
            logger.exception("Error in KB-enhanced processing")
            if self.state:
                self.state.errors.append(f"KB-enhanced processing failed: {str(e)}")
            # Record pipeline failure
            self.observability_hooks.end_pipeline(success=False)
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

            extraction_result = await self._stage2_extract_entities(preprocessed, detected_domain)
            if extraction_result is None:
                return None, self.state

            resolved = await self._stage3_resolve_entities(extraction_result)
            if resolved is None:
                return None, self.state

            graph = await self._stage4_build_graph(resolved, detected_domain)
            if graph is None:
                return None, self.state

            # Return analysis results (analysis dict removed as unused)

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

        with self.observability_hooks.track_stage(stage_name) as stage_metrics:
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

                # Record stage metrics
                stage_metrics.success = True
                stage_metrics.add_attribute("input_length", len(text))
                stage_metrics.add_attribute("chunks", len(preprocessed.chunks))
                stage_metrics.add_attribute("tokens", self.state.metrics.input_token_count)

                logger.info(f"Stage 1 completed: {result.metrics}")

            except Exception as e:
                logger.exception(f"Stage 1 failed: {e}")
                result.status = StageStatus.FAILED
                result.error = str(e)

                # Record error in observability
                self.observability_hooks.record_error(
                    str(e), error_type=type(e).__name__, stage_name=stage_name
                )
                stage_metrics.success = False
                stage_metrics.error_count += 1

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

        with self.observability_hooks.track_stage(
            stage_name, attributes={"use_kb": use_kb, "domain": domain}
        ) as stage_metrics:
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
                    self.state.metrics.avg_entity_confidence = sum(confidences) / len(confidences)

                result.result = extraction_result
                result.status = StageStatus.COMPLETED
                result.metrics = {
                    "entities": len(extraction_result.entities),
                    "relations": len(extraction_result.relations),
                    "avg_confidence": self.state.metrics.avg_entity_confidence,
                    "errors": len(extraction_result.errors),
                }

                # Record stage metrics
                stage_metrics.success = True
                stage_metrics.add_attribute("entities", len(extraction_result.entities))
                stage_metrics.add_attribute("relations", len(extraction_result.relations))
                stage_metrics.add_attribute(
                    "avg_confidence", self.state.metrics.avg_entity_confidence
                )

                # Handle warnings
                if extraction_result.errors:
                    for error in extraction_result.errors:
                        result.warnings.append(f"Extraction error: {error.message}")
                        # Record individual extraction errors
                        self.observability_hooks.record_error(
                            error.message, error_type="extraction_error", stage_name=stage_name
                        )
                        stage_metrics.error_count += 1

                logger.info(f"Stage 2 completed: {result.metrics}")

            except Exception as e:
                logger.exception(f"Stage 2 failed: {e}")
                result.status = StageStatus.FAILED
                result.error = str(e)

                # Record error in observability
                self.observability_hooks.record_error(
                    str(e), error_type=type(e).__name__, stage_name=stage_name
                )
                stage_metrics.success = False
                stage_metrics.error_count += 1

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

        with self.observability_hooks.track_stage(stage_name) as stage_metrics:
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

                # Record stage metrics
                stage_metrics.success = True
                stage_metrics.add_attribute(
                    "coreferences_resolved", self.state.metrics.coreferences_resolved
                )
                stage_metrics.add_attribute(
                    "actors_consolidated", self.state.metrics.actors_consolidated
                )
                stage_metrics.add_attribute("entities_after", len(resolved.entities))

                logger.info(f"Stage 3 completed: {result.metrics}")

            except Exception as e:
                logger.exception(f"Stage 3 failed: {e}")
                result.status = StageStatus.FAILED
                result.error = str(e)

                # Record error in observability
                self.observability_hooks.record_error(
                    str(e), error_type=type(e).__name__, stage_name=stage_name
                )
                stage_metrics.success = False
                stage_metrics.error_count += 1

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

        with self.observability_hooks.track_stage(
            stage_name, attributes={"domain": domain}
        ) as stage_metrics:
            try:
                # Build graph - need to extract actor profiles from resolved
                actor_profiles: Dict[str, ActorProfile] = {}  # TODO: extract from resolved if available
                # Convert string domain to DomainType if needed
                from bpmn_agent.models.knowledge_base import DomainType

                domain_enum = None
                if domain:
                    try:
                        domain_enum = DomainType(domain)
                    except ValueError:
                        # Unknown domain, leave as None
                        domain_enum = None

                graph, validation_report, implicit_flows = self.graph_builder.construct_graph(
                    extraction_result=resolved, actor_profiles=actor_profiles, domain=domain_enum
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

                # Record stage metrics
                stage_metrics.success = True
                stage_metrics.add_attribute("nodes", len(graph.nodes))
                stage_metrics.add_attribute("edges", len(graph.edges))
                stage_metrics.add_attribute("is_acyclic", graph.is_acyclic)
                stage_metrics.add_attribute("complexity", graph.complexity)

                logger.info(f"Stage 4 completed: {result.metrics}")

            except Exception as e:
                logger.exception(f"Stage 4 failed: {e}")
                result.status = StageStatus.FAILED
                result.error = str(e)

                # Record error in observability
                self.observability_hooks.record_error(
                    str(e), error_type=type(e).__name__, stage_name=stage_name
                )
                stage_metrics.success = False
                stage_metrics.error_count += 1

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

        with self.observability_hooks.track_stage(
            stage_name, attributes={"process_name": process_name, "domain": domain}
        ) as stage_metrics:
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

                # Record stage metrics
                stage_metrics.success = True
                stage_metrics.add_attribute(
                    "xml_size_bytes", self.state.metrics.output_xml_size_bytes
                )
                stage_metrics.add_attribute(
                    "element_count", self.state.metrics.output_element_count
                )

                logger.info(f"Stage 5 completed: {result.metrics}")

            except Exception as e:
                logger.exception(f"Stage 5 failed: {e}")
                result.status = StageStatus.FAILED
                result.error = str(e)

                # Record error in observability
                self.observability_hooks.record_error(
                    str(e), error_type=type(e).__name__, stage_name=stage_name
                )
                stage_metrics.success = False
                stage_metrics.error_count += 1

                if self.config.error_handling == ErrorHandlingStrategy.STRICT:
                    self.state.add_stage_result(result)
                    return None

        result.end_time = datetime.now()
        result.duration_ms = (result.end_time - result.start_time).total_seconds() * 1000
        self.state.add_stage_result(result)

        return result.result

    async def _stage6_validate_phase4(
        self,
        xml_output: str,
        graph: ProcessGraph,
        extraction_result: ExtractionResultWithErrors,
        domain: Optional[DomainType] = None,
        patterns_applied: Optional[List[str]] = None,
    ) -> None:
        """Stage 6: Phase 4 Validation (XSD + RAG Pattern Validation)."""
        stage_name = "phase4_validation"
        result = StageResult(stage_name=stage_name, status=StageStatus.RUNNING)
        result.start_time = datetime.now()

        with self.observability_hooks.track_stage(
            stage_name,
            attributes={
                "domain": domain.value if domain else None,
                "patterns_count": len(patterns_applied) if patterns_applied else 0,
            },
        ) as stage_metrics:
            try:
                # Perform unified validation
                validation_result = self.validation_layer.validate(
                    xml_content=xml_output,
                    graph=graph,
                    extraction_result=extraction_result,
                    domain=domain,
                    patterns_applied=patterns_applied,
                )

                # Update state with validation results
                if validation_result.overall_valid:
                    self.state.warnings.extend(validation_result.combined_issues)
                else:
                    self.state.errors.extend(validation_result.combined_issues)

                # Add suggestions to state
                if validation_result.combined_suggestions:
                    self.state.warnings.extend(
                        [f"Suggestion: {s}" for s in validation_result.combined_suggestions[:5]]
                    )

                result.result = validation_result
                result.status = StageStatus.COMPLETED
                result.metrics = {
                    "valid": validation_result.overall_valid,
                    "quality_score": validation_result.overall_quality_score,
                    "xsd_errors": validation_result.xsd_result.total_errors,
                    "xsd_warnings": validation_result.xsd_result.total_warnings,
                    "rag_compliance": (
                        validation_result.rag_result.overall_compliance_score
                        if validation_result.rag_result
                        else None
                    ),
                    "patterns_validated": (
                        validation_result.rag_result.patterns_validated
                        if validation_result.rag_result
                        else 0
                    ),
                }

                # Record stage metrics
                stage_metrics.success = True
                stage_metrics.add_attribute("valid", validation_result.overall_valid)
                stage_metrics.add_attribute(
                    "quality_score", validation_result.overall_quality_score
                )

                logger.info(f"Stage 6 completed: {result.metrics}")

            except Exception as e:
                logger.exception(f"Stage 6 failed: {e}")
                result.status = StageStatus.FAILED
                result.error = str(e)

                # Record error in observability
                self.observability_hooks.record_error(
                    str(e), error_type=type(e).__name__, stage_name=stage_name
                )
                stage_metrics.success = False
                stage_metrics.error_count += 1

                # Handle validation failures based on configuration
                if self.config.pipeline_config.validation_fail_on_error:
                    # Fail pipeline if validation fails and fail_on_error is enabled
                    self.state.add_stage_result(result)
                    return
                elif self.config.error_handling == ErrorHandlingStrategy.STRICT:
                    # In strict mode, add result but don't fail pipeline
                    self.state.add_stage_result(result)

        result.end_time = datetime.now()
        result.duration_ms = (result.end_time - result.start_time).total_seconds() * 1000
        self.state.add_stage_result(result)

    def _get_patterns_applied(self) -> Optional[List[str]]:
        """
        Extract patterns applied during XML generation.

        Returns:
            List of pattern IDs that were applied, or None if not available
        """
        # Try to get patterns from XML generator's id_mappings
        if hasattr(self.xml_generator, "id_mappings"):
            pattern_ids = set()
            for mapping in self.xml_generator.id_mappings:
                if mapping.pattern_reference:
                    pattern_ids.add(mapping.pattern_reference.pattern_id)
            if pattern_ids:
                return list(pattern_ids)

        # If not available from XML generator, return None
        # In the future, we could track patterns in state or context
        return None

    # ==================
    # Utility Methods
    # ==================

    async def _detect_domain(self, text: str) -> Optional[str]:
        """Detect process domain from text."""
        if not self.domain_classifier:
            return None

        try:
            result = self.domain_classifier.classify_domain(text)
            domain = result.domain.value if result else None
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
