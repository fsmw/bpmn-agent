"""
Stage 2, Tasks 2.2.2-2.2.3: Entity & Relation Extraction

Implements:
- Task 2.2.2: Build entity extraction function with LLM
- Task 2.2.3: JSON parsing with error recovery
- KB Integration: Use domain patterns and examples to improve extraction

Uses LLM to extract entities and relationships, with robust error handling
and KB-augmented prompts for improved domain-specific accuracy.
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from bpmn_agent.core.llm_client import BaseLLMClient
from bpmn_agent.knowledge import ContextOptimizer, ContextSelector, PatternRecognizer
from bpmn_agent.models.extraction import (
    ConfidenceLevel,
    EntityAttribute,
    EntityType,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionError,
    ExtractionMetadata,
    ExtractionResultWithErrors,
    RelationType,
)
from bpmn_agent.models.knowledge_base import ContextPackage
from bpmn_agent.stages.extraction_prompts import (
    create_extraction_prompt,
    render_full_prompt,
)
from bpmn_agent.stages.text_preprocessing import PreprocessedText

logger = logging.getLogger(__name__)


# ===========================
# Task 2.2.3: JSON Parsing & Error Recovery
# ===========================


class JSONParser:
    """Robust JSON parser with error recovery."""

    @staticmethod
    def parse_extraction_response(response: str) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """
        Parse LLM response to JSON with error recovery.

        Args:
            response: Raw LLM response text

        Returns:
            (success: bool, parsed_dict: Dict, error_message: Optional[str])
        """
        # 1. Try primary parse
        try:
            data = json.loads(response)
            return True, data, None
        except json.JSONDecodeError:
            pass  # Try recovery

        # 2. Try removing markdown code blocks
        cleaned = response
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0]
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0]

        try:
            data = json.loads(cleaned)
            return True, data, None
        except json.JSONDecodeError:
            pass  # Try more recovery

        # 3. Try lenient parsing
        cleaned = JSONParser._fix_common_json_issues(cleaned)
        try:
            data = json.loads(cleaned)
            return True, data, None
        except json.JSONDecodeError as e:
            # Complete failure
            return False, {}, f"JSON parsing failed: {str(e)}"

    @staticmethod
    def _fix_common_json_issues(text: str) -> str:
        """
        Fix common JSON formatting issues.

        Args:
            text: Malformed JSON text

        Returns:
            Corrected JSON text
        """
        # Remove comments
        text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

        # Fix trailing commas
        text = re.sub(r",(\s*[}\]])", r"\1", text)

        # Fix single quotes to double quotes (carefully)
        # Only for key-value pairs
        text = re.sub(r"'([^']*)'(\s*:)", r'"\1"\2', text)
        text = re.sub(r":\s*'([^']*)'", r': "\1"', text)

        # Fix unquoted keys
        text = re.sub(r"([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)", r'\1"\2"\3', text)

        # Remove trailing whitespace
        text = text.strip()

        # Ensure proper structure
        if not text.startswith("{"):
            text = "{" + text
        if not text.endswith("}"):
            text = text + "}"

        return text


# ===========================
# KB-Augmented Prompt Builder
# ===========================


class KBAugmentedPromptBuilder:
    """Builds extraction prompts augmented with KB patterns and examples."""

    @staticmethod
    def build_kb_augmented_prompt(
        base_prompt: str,
        context_package: Optional[ContextPackage] = None,
        include_examples: bool = True,
        include_patterns: bool = True,
    ) -> str:
        """
        Augment extraction prompt with KB patterns and examples.

        Args:
            base_prompt: Original extraction prompt
            context_package: KB context with patterns/examples
            include_examples: Whether to include few-shot examples
            include_patterns: Whether to include pattern hints

        Returns:
            Augmented prompt string
        """
        if not context_package:
            return base_prompt

        augmentations = []

        # Add pattern hints if applicable
        if include_patterns and context_package.recognized_patterns:
            augmentations.append("\n## Relevant BPMN Patterns in This Domain:\n")
            for pattern_id in context_package.recognized_patterns[:3]:  # Limit to 3 patterns
                # Find pattern details
                for pattern in context_package.selected_patterns:
                    if pattern.id == pattern_id:
                        augmentations.append(
                            f"- **{pattern.name}**: {pattern.description[:100]}...\n"
                        )
                        augmentations.append(
                            f"  Structure: {', '.join(pattern.graph_structure.nodes[:4])}\n"
                        )
                        break

        # Add domain terminology if available
        if context_package.domain_terms:
            augmentations.append("\n## Domain-Specific Terminology:\n")
            for category, terms in context_package.domain_terms.items():
                augmentations.append(f"- {category}: {', '.join(terms[:5])}\n")

        # Add few-shot examples if available
        if include_examples and context_package.selected_examples:
            augmentations.append("\n## Example Process Descriptions (for reference):\n")
            for i, example in enumerate(context_package.selected_examples[:2], 1):
                augmentations.append(f"{i}. {example.text[:150]}...\n")

        # Add validation rules if available
        if context_package.validation_rules:
            augmentations.append("\n## Validation Rules for This Domain:\n")
            for rule in context_package.validation_rules[:3]:
                augmentations.append(f"- {rule}\n")

        return base_prompt + "\n".join(augmentations)


# ===========================
# Task 2.2.2: Entity Extraction Function
# ===========================


class EntityExtractor:
    """Extracts entities and relations from text using LLM with KB augmentation."""

    def __init__(self, llm_client: BaseLLMClient, enable_kb: bool = True):
        """
        Initialize entity extractor with optional KB augmentation.

        Args:
            llm_client: LLM client for API calls
            enable_kb: Whether to enable KB-augmented prompts
        """
        self.llm_client = llm_client
        self.prompt_template = create_extraction_prompt()
        self.enable_kb = enable_kb

        # Initialize KB components (lazy loading)
        self._context_optimizer: Optional[ContextOptimizer] = None
        self._context_selector: Optional[ContextSelector] = None
        self._pattern_recognizer: Optional[PatternRecognizer] = None

    def _ensure_kb_initialized(self) -> None:
        """Initialize KB components if enabled and not already done."""
        if not self.enable_kb:
            return

        if self._context_optimizer is None:
            try:
                self._context_selector = ContextSelector()
                self._context_optimizer = ContextOptimizer()
                self._pattern_recognizer = PatternRecognizer()
                logger.debug("KB components initialized for entity extraction")
            except Exception as e:
                logger.warning(f"Failed to initialize KB components: {e}")
                self.enable_kb = False

    async def extract_from_text(
        self,
        text: str,
        preprocessed: Optional[PreprocessedText] = None,
        llm_temperature: float = 0.3,
        max_retries: int = 2,
    ) -> ExtractionResultWithErrors:
        """
        Extract entities and relations from text using LLM with KB augmentation.

        Args:
            text: Original process description
            preprocessed: Optional preprocessed text (with chunks and KB metadata)
            llm_temperature: LLM temperature (0.0-1.0)
            max_retries: Number of retries on JSON parse failure

        Returns:
            ExtractionResultWithErrors with extracted entities and relations
        """
        start_time = datetime.now()
        errors: List[ExtractionError] = []
        entities: List[ExtractedEntity] = []
        relations: List[ExtractedRelation] = []
        context_package: Optional[ContextPackage] = None
        extraction_warnings: List[str] = []

        try:
            # 1. Validate input
            if not text or len(text.strip()) < 10:
                raise ValueError("Input text too short for extraction")

            # 2. Build KB context if available and enabled
            if self.enable_kb and preprocessed:
                try:
                    self._ensure_kb_initialized()
                    if self._context_selector and self._context_optimizer:
                        # Extract KB metadata from preprocessing
                        detected_domain = preprocessed.metadata.get("detected_domain")
                        detected_complexity = preprocessed.metadata.get("detected_complexity")

                        # First select context, then optimize
                        selected_context = self._context_selector.select_context(
                            text=text,
                            max_tokens=3000,
                            domain=detected_domain,
                            complexity=detected_complexity,
                        )

                        # Then optimize to fit token budget
                        context_package = self._context_optimizer.optimize_context(
                            context=selected_context,
                            max_tokens=2000,  # Reserve token budget
                        )
                        logger.debug(
                            f"KB context built: {len(context_package.selected_patterns)} patterns, "
                            f"{len(context_package.selected_examples)} examples"
                        )
                except Exception as e:
                    logger.warning(f"Failed to build KB context: {e}")
                    # Continue without KB context

            # 3. Render full prompt with KB augmentation if available
            full_prompt = render_full_prompt(self.prompt_template, text)

            if context_package:
                full_prompt = KBAugmentedPromptBuilder.build_kb_augmented_prompt(
                    full_prompt,
                    context_package=context_package,
                    include_examples=True,
                    include_patterns=True,
                )
                logger.debug("Extraction prompt augmented with KB context")

            # 4. Call LLM with retries
            parsed_data = None
            last_error = None

            for attempt in range(max_retries + 1):
                try:
                    # Call LLM
                    from bpmn_agent.core.llm_client import LLMMessage

                    response = await self.llm_client.call(
                        messages=[
                            LLMMessage(
                                role="system",
                                content=self.prompt_template.system_message,
                                name=None,
                            ),
                            LLMMessage(
                                role="user",
                                content=full_prompt,
                                name=None,
                            ),
                        ],
                        temperature=llm_temperature,
                        max_tokens=4096,
                    )

                    # Parse response
                    success, data, error_msg = JSONParser.parse_extraction_response(
                        response.content
                    )

                    if success:
                        parsed_data = data
                        break
                    else:
                        last_error = error_msg
                        if attempt < max_retries:
                            # Retry with lower temperature for more structured output
                            llm_temperature = max(0.0, llm_temperature - 0.1)

                except Exception as e:
                    last_error = str(e)
                    continue

            if not parsed_data:
                raise ValueError(
                    f"Failed to extract JSON after {max_retries + 1} attempts: {last_error}"
                )

            # 4. Extract entities from parsed data
            entities = self._parse_entities(parsed_data.get("entities", []))

            # 5. Extract relations from parsed data
            relations = self._parse_relations(parsed_data.get("relations", []))

            # 6. Validate extracted data
            extraction_warnings = parsed_data.get("warnings", [])

            # Warn if missing start/end events
            event_types = {e.type for e in entities}
            if EntityType.EVENT not in event_types:
                extraction_warnings.append("No events detected - missing start/end points?")

            # Warn if no actors
            actor_count = len([e for e in entities if e.type == EntityType.ACTOR])
            if actor_count == 0:
                extraction_warnings.append("No actors detected - unclear who performs activities")

        except Exception as e:
            errors.append(
                ExtractionError(
                    error_type="extraction_failed",
                    message=str(e),
                    severity="error",
                    recoverable=False,
                    context=None,
                )
            )

        # 7. Build metadata
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        confidence_counts = {
            ConfidenceLevel.HIGH: sum(1 for e in entities if e.confidence == ConfidenceLevel.HIGH),
            ConfidenceLevel.MEDIUM: sum(
                1 for e in entities if e.confidence == ConfidenceLevel.MEDIUM
            ),
            ConfidenceLevel.LOW: sum(1 for e in entities if e.confidence == ConfidenceLevel.LOW),
        }

        relation_confidence_counts = {
            ConfidenceLevel.HIGH: sum(1 for r in relations if r.confidence == ConfidenceLevel.HIGH),
            ConfidenceLevel.MEDIUM: sum(
                1 for r in relations if r.confidence == ConfidenceLevel.MEDIUM
            ),
            ConfidenceLevel.LOW: sum(1 for r in relations if r.confidence == ConfidenceLevel.LOW),
        }

        metadata = ExtractionMetadata(
            input_text=text,
            input_length=len(text),
            extraction_timestamp=start_time.isoformat(),
            extraction_duration_ms=duration_ms,
            llm_model=self.llm_client.config.model,
            llm_temperature=llm_temperature,
            stage="extraction",
            total_entities_extracted=len(entities),
            high_confidence_entities=confidence_counts[ConfidenceLevel.HIGH],
            medium_confidence_entities=confidence_counts[ConfidenceLevel.MEDIUM],
            low_confidence_entities=confidence_counts[ConfidenceLevel.LOW],
            total_relations_extracted=len(relations),
            high_confidence_relations=relation_confidence_counts[ConfidenceLevel.HIGH],
            warnings=extraction_warnings if extraction_warnings else [],
            notes=None,
        )

        return ExtractionResultWithErrors(
            entities=entities,
            relations=relations,
            metadata=metadata,
            errors=errors,
        )

    def _parse_entities(self, entities_data: List[Dict[str, Any]]) -> List[ExtractedEntity]:
        """
        Parse entities from LLM response.

        Args:
            entities_data: Raw entity data from JSON response

        Returns:
            List of ExtractedEntity objects
        """
        entities = []

        for entity_data in entities_data:
            try:
                entity_id = entity_data.get("id", f"entity_{len(entities)}")
                entity_type_str = entity_data.get("type", "activity").lower()

                # Map to EntityType
                entity_type = self._map_entity_type(entity_type_str, entity_data.get("name", ""))

                # Parse confidence
                confidence_str = entity_data.get("confidence", "medium").lower()
                confidence = (
                    ConfidenceLevel(confidence_str)
                    if confidence_str in ["high", "medium", "low"]
                    else ConfidenceLevel.MEDIUM
                )

                # Parse attributes
                attributes = {}
                if "attributes" in entity_data:
                    attrs = entity_data["attributes"]
                    if isinstance(attrs, dict):
                        for key, value in attrs.items():
                            attributes[key] = EntityAttribute(
                                key=key,
                                value=value,
                                value_type=type(value).__name__,
                                confidence=confidence,
                                source_text=None,
                            )

                entity = ExtractedEntity(
                    id=entity_id,
                    type=entity_type,
                    name=entity_data.get("name", "Unknown"),
                    description=entity_data.get("description"),
                    confidence=confidence,
                    source_text=entity_data.get("source_text"),
                    character_offsets=entity_data.get("character_offsets"),
                    attributes=attributes,
                    alternative_names=entity_data.get("alternative_names", []),
                    is_implicit=entity_data.get("is_implicit", False),
                    is_uncertain=entity_data.get("is_uncertain", False),
                )

                entities.append(entity)

            except Exception:
                # Log but don't fail - skip malformed entities
                continue

        return entities

    def _parse_relations(self, relations_data: List[Dict[str, Any]]) -> List[ExtractedRelation]:
        """
        Parse relations from LLM response.

        Args:
            relations_data: Raw relation data from JSON response

        Returns:
            List of ExtractedRelation objects
        """
        relations: List[ExtractedRelation] = []

        for relation_data in relations_data:
            try:
                relation_id = relation_data.get("id", f"relation_{len(relations)}")
                relation_type_str = relation_data.get("type", "precedes").lower()

                # Map to RelationType
                relation_type = self._map_relation_type(relation_type_str)

                # Parse confidence
                confidence_str = relation_data.get("confidence", "medium").lower()
                confidence = (
                    ConfidenceLevel(confidence_str)
                    if confidence_str in ["high", "medium", "low"]
                    else ConfidenceLevel.MEDIUM
                )

                relation = ExtractedRelation(
                    id=relation_id,
                    type=relation_type,
                    source_id=relation_data.get("source_id", "unknown"),
                    target_id=relation_data.get("target_id", "unknown"),
                    label=relation_data.get("label"),
                    confidence=confidence,
                    source_text=relation_data.get("source_text"),
                    attributes=relation_data.get("attributes", {}),
                    is_conditional=relation_data.get("condition") is not None,
                    condition_expression=relation_data.get("condition"),
                    is_implicit=relation_data.get("is_implicit", False),
                )

                relations.append(relation)

            except Exception:
                # Log but don't fail - skip malformed relations
                continue

        return relations

    @staticmethod
    def _map_entity_type(extracted_type: str, entity_name: str = "") -> EntityType:
        """
        Map extracted entity type to EntityType enum.

        Args:
            extracted_type: Type from LLM
            entity_name: Entity name for context

        Returns:
            EntityType enum value
        """
        extracted_lower = extracted_type.lower().strip()

        # Direct mappings
        type_map = {
            "activity": EntityType.ACTIVITY,
            "task": EntityType.ACTIVITY,
            "event": EntityType.EVENT,
            "start": EntityType.EVENT,
            "end": EntityType.EVENT,
            "gateway": EntityType.GATEWAY,
            "decision": EntityType.GATEWAY,
            "actor": EntityType.ACTOR,
            "participant": EntityType.ACTOR,
            "data": EntityType.DATA,
            "resource": EntityType.RESOURCE,
            "constraint": EntityType.CONSTRAINT,
            "organization": EntityType.ORGANIZATION,
        }

        if extracted_lower in type_map:
            return type_map[extracted_lower]

        # Default based on keywords
        if any(kw in extracted_lower for kw in ["activity", "task", "action"]):
            return EntityType.ACTIVITY
        elif any(kw in extracted_lower for kw in ["event", "start", "end", "message"]):
            return EntityType.EVENT
        elif any(kw in extracted_lower for kw in ["gateway", "decision", "choice"]):
            return EntityType.GATEWAY
        elif any(kw in extracted_lower for kw in ["actor", "role", "person", "participant"]):
            return EntityType.ACTOR

        return EntityType.ACTIVITY  # Default

    @staticmethod
    def _map_relation_type(extracted_type: str) -> RelationType:
        """
        Map extracted relation type to RelationType enum.

        Args:
            extracted_type: Type from LLM

        Returns:
            RelationType enum value
        """
        extracted_lower = extracted_type.lower().strip()

        # Direct mappings
        type_map = {
            "precedes": RelationType.PRECEDES,
            "triggers": RelationType.TRIGGERS,
            "conditional": RelationType.CONDITIONAL,
            "parallel": RelationType.PARALLEL_TO,
            "alternative": RelationType.ALTERNATIVE_TO,
            "involves": RelationType.INVOLVES,
            "uses": RelationType.USES,
            "produces": RelationType.PRODUCES,
            "consumes": RelationType.CONSUMES,
            "sends_to": RelationType.SENDS_TO,
            "sends": RelationType.SENDS_TO,
            "receives_from": RelationType.RECEIVES_FROM,
            "receives": RelationType.RECEIVES_FROM,
        }

        if extracted_lower in type_map:
            return type_map[extracted_lower]

        return RelationType.PRECEDES  # Default


__all__ = [
    "JSONParser",
    "EntityExtractor",
]
