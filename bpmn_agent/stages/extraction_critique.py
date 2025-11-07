"""
Self-Critique and Validation Stage

Implements validation and self-critique mechanism to improve extraction quality:
- Validates extracted entities and relationships
- Generates critique feedback based on validation rules
- Identifies potential improvements or issues
- Supports refinement loop for iterative extraction improvement
"""

import json
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, AsyncIterator

from pydantic import BaseModel, Field

from bpmn_agent.core.llm_client import BaseLLMClient, LLMMessage, LLMResponse
from bpmn_agent.models.extraction import (
    ConfidenceLevel,
    EntityType,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    RelationType,
)

logger = logging.getLogger(__name__)


class ValidationIssueSeverity(str, Enum):
    """Severity levels for validation issues."""

    CRITICAL = "critical"  # Blocks further processing
    ERROR = "error"  # Must be fixed
    WARNING = "warning"  # Should be fixed
    INFO = "info"  # Informational


class ValidationIssueType(str, Enum):
    """Types of validation issues."""

    MISSING_START_EVENT = "missing_start_event"
    MISSING_END_EVENT = "missing_end_event"
    DISCONNECTED_NODES = "disconnected_nodes"
    MISSING_LABELS = "missing_labels"
    DUPLICATE_ENTITIES = "duplicate_entities"
    INVALID_RELATIONSHIPS = "invalid_relationships"
    UNCLEAR_BRANCHING = "unclear_branching"
    AMBIGUOUS_ACTORS = "ambiguous_actors"
    LOW_CONFIDENCE_ENTITIES = "low_confidence_entities"
    INCOMPLETE_DATA_FLOW = "incomplete_data_flow"
    MISSING_BUSINESS_RULES = "missing_business_rules"


class ValidationIssue(BaseModel):
    """A single validation issue found during critique."""

    type: ValidationIssueType = Field(..., description="Issue type")
    severity: ValidationIssueSeverity = Field(..., description="Issue severity")
    message: str = Field(..., description="Human-readable issue description")
    affected_entity_ids: List[str] = Field(
        default_factory=list, description="Entity IDs affected by this issue"
    )
    suggestion: Optional[str] = Field(None, description="Suggested fix or improvement")
    confidence: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Confidence in the issue assessment"
    )


class CritiqueResult(BaseModel):
    """Result of the critique validation process."""

    is_valid: bool = Field(..., description="Whether extraction passes validation")
    issues: List[ValidationIssue] = Field(
        default_factory=list, description="List of validation issues found"
    )
    quality_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Overall quality score (0-1)"
    )
    suggestions_for_improvement: List[str] = Field(
        default_factory=list, description="Overall suggestions for improvement"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When critique was performed"
    )
    needs_refinement: bool = Field(
        default=False, description="Whether refinement loop should be triggered"
    )


class ExtractionValidator:
    """Validates extracted entities and relationships."""

    @staticmethod
    def validate_extraction(result: ExtractionResult) -> CritiqueResult:
        """
        Validate extraction result against business rules.

        Args:
            result: Extraction result to validate

        Returns:
            CritiqueResult with validation issues
        """
        issues = []

        # Check for start and end events
        has_start_event = any(
            entity.type == EntityType.EVENT and "start" in entity.name.lower()
            for entity in result.entities
        )
        has_end_event = any(
            entity.type == EntityType.EVENT and "end" in entity.name.lower()
            for entity in result.entities
        )

        if not has_start_event:
            issues.append(
                ValidationIssue(
                    type=ValidationIssueType.MISSING_START_EVENT,
                    severity=ValidationIssueSeverity.ERROR,
                    message="No start event detected in the extraction",
                    suggestion="Process should begin with a start event",
                )
            )

        if not has_end_event:
            issues.append(
                ValidationIssue(
                    type=ValidationIssueType.MISSING_END_EVENT,
                    severity=ValidationIssueSeverity.ERROR,
                    message="No end event detected in the extraction",
                    suggestion="Process should end with an end event",
                )
            )

        # Check for low confidence entities
        low_confidence_entities = [
            entity.id for entity in result.entities if entity.confidence == ConfidenceLevel.LOW
        ]
        if low_confidence_entities:
            issues.append(
                ValidationIssue(
                    type=ValidationIssueType.LOW_CONFIDENCE_ENTITIES,
                    severity=ValidationIssueSeverity.WARNING,
                    message=f"Found {len(low_confidence_entities)} low-confidence entities",
                    affected_entity_ids=low_confidence_entities,
                    suggestion="Review and validate low-confidence extractions",
                )
            )

        # Check for entities without descriptions
        entities_without_desc = [
            entity.id
            for entity in result.entities
            if not entity.description or entity.description.strip() == ""
        ]
        if len(entities_without_desc) > len(result.entities) * 0.3:
            issues.append(
                ValidationIssue(
                    type=ValidationIssueType.MISSING_LABELS,
                    severity=ValidationIssueSeverity.WARNING,
                    message=f"Many entities ({len(entities_without_desc)}) lack descriptions",
                    affected_entity_ids=entities_without_desc,
                    suggestion="Add descriptions to entities for clarity",
                )
            )

        # Check for isolated nodes
        connected_ids = set()
        for rel in result.relations:
            connected_ids.add(rel.source_id)
            connected_ids.add(rel.target_id)

        all_entity_ids = {entity.id for entity in result.entities}
        disconnected_ids = all_entity_ids - connected_ids

        if disconnected_ids:
            issues.append(
                ValidationIssue(
                    type=ValidationIssueType.DISCONNECTED_NODES,
                    severity=ValidationIssueSeverity.WARNING,
                    message=f"Found {len(disconnected_ids)} disconnected entities",
                    affected_entity_ids=list(disconnected_ids),
                    suggestion="Ensure all entities are connected by relationships",
                )
            )

        # Check for data flow completeness
        tasks = [e for e in result.entities if e.type == EntityType.ACTIVITY]
        data_entities = [e for e in result.entities if e.type == EntityType.DATA]

        if tasks and not data_entities:
            issues.append(
                ValidationIssue(
                    type=ValidationIssueType.INCOMPLETE_DATA_FLOW,
                    severity=ValidationIssueSeverity.INFO,
                    message="No data objects identified in process",
                    suggestion="Consider if process has data inputs/outputs",
                )
            )

        # Calculate quality score
        quality_score = ExtractionValidator._calculate_quality_score(result, issues)

        # Determine if needs refinement
        needs_refinement = any(
            issue.severity in [ValidationIssueSeverity.ERROR, ValidationIssueSeverity.CRITICAL]
            for issue in issues
        )

        # Generate overall suggestions
        suggestions = ExtractionValidator._generate_suggestions(result, issues)

        return CritiqueResult(
            is_valid=len([i for i in issues if i.severity == ValidationIssueSeverity.CRITICAL])
            == 0,
            issues=issues,
            quality_score=quality_score,
            suggestions_for_improvement=suggestions,
            needs_refinement=needs_refinement,
        )

    @staticmethod
    def _calculate_quality_score(result: ExtractionResult, issues: List[ValidationIssue]) -> float:
        """Calculate overall quality score."""
        base_score = 1.0

        # Deduct for each issue based on severity
        severity_weights = {
            ValidationIssueSeverity.CRITICAL: 0.3,
            ValidationIssueSeverity.ERROR: 0.15,
            ValidationIssueSeverity.WARNING: 0.05,
            ValidationIssueSeverity.INFO: 0.02,
        }

        for issue in issues:
            weight = severity_weights.get(issue.severity, 0.0)
            base_score -= weight * issue.confidence

        # Bonus for high-confidence entities
        high_confidence_count = sum(
            1 for entity in result.entities if entity.confidence == ConfidenceLevel.HIGH
        )
        if result.entities:
            confidence_bonus = (high_confidence_count / len(result.entities)) * 0.1
            base_score += confidence_bonus

        return max(0.0, min(1.0, base_score))

    @staticmethod
    def _generate_suggestions(result: ExtractionResult, issues: List[ValidationIssue]) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []

        if len(result.entities) < 3:
            suggestions.append(
                "Process seems simplified (< 3 entities). Verify if all steps were captured."
            )

        if len(result.relations) < len(result.entities) - 1:
            suggestions.append("Relations seem sparse. Ensure activities are properly sequenced.")

        actors = [e for e in result.entities if e.type == EntityType.ACTOR]
        if not actors:
            suggestions.append(
                "No actors identified. Consider if process involves roles or participants."
            )

        gateways = [e for e in result.entities if e.type == EntityType.GATEWAY]
        if not gateways:
            suggestions.append("No decision gateways found. Verify if process has branching logic.")

        return suggestions


class CritiqueAgent:
    """
    Generates critique feedback using LLM to identify extraction issues.
    """

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        """
        Initialize critique agent.

        Args:
            llm_client: LLM client for critique generation (optional)
        """
        self.llm_client = llm_client

    def _format_extraction_for_critique(self, result: ExtractionResult) -> str:
        """Format extraction result for LLM critique."""
        entities_str = "\n".join(
            [
                f"- {entity.name} ({entity.type.value}): {entity.description or 'No description'} [Confidence: {entity.confidence.value}]"
                for entity in result.entities
            ]
        )

        relations_str = "\n".join(
            [f"- {rel.type.value}: {rel.source_id} -> {rel.target_id}" for rel in result.relations]
        )

        return f"""Extracted Process:

Entities:
{entities_str}

Relationships:
{relations_str or "No relationships"}
"""

    def _create_critique_prompt(self, extraction_str: str, original_text: str) -> str:
        """Create LLM prompt for critique."""
        return f"""You are a business process expert. Review the following extraction from the process description and provide critique.

Original Process Description:
{original_text}

{extraction_str}

Please analyze this extraction and provide:
1. Any missing entities or relationships
2. Ambiguities or unclear connections
3. Low-confidence entities that should be reviewed
4. Suggestions for improvement
5. Overall quality assessment (0-1)

Provide your response as JSON with keys: issues, suggestions, confidence_assessment, quality_score
"""

    async def generate_critique_feedback(
        self,
        result: ExtractionResult,
        original_text: str,
        max_iterations: int = 3,
    ) -> Dict:
        """
        Generate critique feedback for extraction results.

        Args:
            result: Extraction result to critique
            original_text: Original input text
            max_iterations: Maximum refinement iterations

        Returns:
            Dictionary with critique feedback and recommendations
        """
        # First do rule-based validation
        validation = ExtractionValidator.validate_extraction(result)

        # Try LLM-based critique if client available
        llm_critique = None
        if self.llm_client:
            try:
                extraction_str = self._format_extraction_for_critique(result)
                prompt = self._create_critique_prompt(extraction_str, original_text)

                messages = [LLMMessage(role="user", content=prompt, name=None)]

                response = await self.llm_client.call(messages, temperature=0.3, max_tokens=1024)

                # Parse LLM response as JSON
                try:
                    # Handle union type: LLMResponse | AsyncIterator[str]
                    if isinstance(response, LLMResponse):
                        response_text = response.content
                    else:
                        # Collect async iterator
                        response_text = ""
                        async for chunk in response:
                            response_text += chunk
                    
                    # Try to extract JSON from response
                    if "```json" in response_text:
                        json_str = response_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in response_text:
                        json_str = response_text.split("```")[1].split("```")[0].strip()
                    else:
                        json_str = response_text

                    llm_critique = json.loads(json_str)
                except (json.JSONDecodeError, IndexError):
                    # If parsing fails, store raw response
                    llm_critique = {"raw_response": response_text if isinstance(response, LLMResponse) else response}
            except Exception as e:
                logger.warning(f"LLM critique failed: {e}")
                llm_critique = None

        return {
            "validation": validation,
            "llm_critique": llm_critique,
            "refinement_recommendations": self._generate_refinement_plan(
                validation, max_iterations
            ),
        }

    @staticmethod
    def _generate_refinement_plan(
        validation: CritiqueResult,
        max_iterations: int,
    ) -> Optional[Dict]:
        """Generate refinement plan if needed."""
        if not validation.needs_refinement:
            return None

        critical_issues = [
            issue
            for issue in validation.issues
            if issue.severity == ValidationIssueSeverity.CRITICAL
        ]
        error_issues = [
            issue for issue in validation.issues if issue.severity == ValidationIssueSeverity.ERROR
        ]

        return {
            "total_issues": len(validation.issues),
            "critical_count": len(critical_issues),
            "error_count": len(error_issues),
            "priority_fixes": [
                {
                    "issue_type": issue.type.value,
                    "description": issue.message,
                    "suggestion": issue.suggestion,
                }
                for issue in (critical_issues + error_issues)[:max_iterations]
            ],
            "estimated_iterations": min(len(critical_issues + error_issues), max_iterations),
        }


class ExtractionRefinementPipeline:
    """
    Pipeline for critique-based refinement of extraction results.

    Iteratively critiques extraction and attempts refinement.
    """

    def __init__(
        self,
        validator: Optional[ExtractionValidator] = None,
        critique_agent: Optional[CritiqueAgent] = None,
        max_iterations: int = 3,
    ):
        """
        Initialize refinement pipeline.

        Args:
            validator: Custom validator (uses default if not provided)
            critique_agent: Critique agent for LLM feedback
            max_iterations: Maximum refinement iterations
        """
        self.validator = validator or ExtractionValidator()
        self.critique_agent = critique_agent or CritiqueAgent()
        self.max_iterations = max_iterations
        self.iteration_history: List[Dict[str, Any]] = []

    def _create_refinement_prompt(
        self,
        original_text: str,
        current_extraction: str,
        issues: List[ValidationIssue],
    ) -> str:
        """Create LLM prompt for refinement based on issues."""
        issues_str = "\n".join(
            [
                f"- [{issue.severity.value}] {issue.message}: {issue.suggestion}"
                for issue in issues[:5]  # Top 5 issues
            ]
        )

        return f"""You are a business process extraction expert. The initial extraction had these issues:

{issues_str}

Original Process Description:
{original_text}

Current Extraction:
{current_extraction}

Please provide an improved extraction that addresses the issues above. 
Focus on:
1. Adding any missing entities (especially start/end events if missing)
2. Connecting disconnected nodes
3. Improving confidence in low-confidence entities
4. Adding descriptive labels

Return ONLY valid JSON with keys: entities, relations
Each entity should have: id, type, name, description, confidence
Each relation should have: id, type, source_id, target_id
"""

    async def refine_extraction(
        self,
        initial_result: ExtractionResult,
        original_text: str,
    ) -> Tuple[ExtractionResult, CritiqueResult, List[Dict]]:
        """
        Refine extraction result through critique loop.

        Args:
            initial_result: Initial extraction result
            original_text: Original input text

        Returns:
            (refined_result, final_critique, history)
        """
        current_result = initial_result
        best_result = initial_result
        best_quality = 0.0
        iteration = 0

        while iteration < self.max_iterations:
            # Validate current result
            validation = self.validator.validate_extraction(current_result)

            # Track best result
            if validation.quality_score > best_quality:
                best_quality = validation.quality_score
                best_result = current_result

            # Record iteration
            self.iteration_history.append(
                {
                    "iteration": iteration,
                    "validation": validation,
                    "issue_count": len(validation.issues),
                    "quality_score": validation.quality_score,
                }
            )

            # Check if refinement needed
            if not validation.needs_refinement:
                logger.info(f"Extraction validated at iteration {iteration}")
                return current_result, validation, self.iteration_history

            # Generate critique feedback
            critique_feedback = await self.critique_agent.generate_critique_feedback(
                current_result,
                original_text,
                self.max_iterations - iteration,
            )

            # Try to refine based on feedback if LLM is available
            if self.critique_agent.llm_client and critique_feedback.get("llm_critique"):
                try:
                    extraction_str = self.critique_agent._format_extraction_for_critique(
                        current_result
                    )
                    refinement_prompt = self._create_refinement_prompt(
                        original_text,
                        extraction_str,
                        validation.issues,
                    )

                    messages = [LLMMessage(role="user", content=refinement_prompt, name=None)]

                    response = await self.critique_agent.llm_client.call(
                        messages,
                        temperature=0.4,
                        max_tokens=2048,
                    )

                    # Handle union type: LLMResponse | AsyncIterator[str]
                    if isinstance(response, LLMResponse):
                        response_text = response.content
                    else:
                        # Collect async iterator
                        response_text = ""
                        async for chunk in response:
                            response_text += chunk

                    # Try to parse refined extraction
                    refined_data = self._parse_refined_extraction(response_text)
                    if refined_data:
                        current_result = self._apply_refinement(
                            current_result,
                            refined_data,
                        )
                        logger.info(f"Applied refinement at iteration {iteration}")
                    else:
                        logger.warning(f"Could not parse refinement at iteration {iteration}")
                        iteration += 1
                        continue

                except Exception as e:
                    logger.warning(f"LLM refinement failed at iteration {iteration}: {e}")

            iteration += 1

        # Return best result after max iterations
        final_validation = self.validator.validate_extraction(best_result)
        return best_result, final_validation, self.iteration_history

    @staticmethod
    def _parse_refined_extraction(response: str) -> Optional[Dict]:
        """Parse refined extraction from LLM response."""
        try:
            response_text = response
            # Try to extract JSON from response
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text

            return json.loads(json_str)  # type: ignore[no-any-return]
        except (json.JSONDecodeError, IndexError):
            return None

    @staticmethod
    def _apply_refinement(
        original: ExtractionResult,
        refined_data: Dict,
    ) -> ExtractionResult:
        """Apply refinement data to extraction result."""
        try:
            # Parse refined entities
            entities: List[ExtractedEntity] = []
            for entity_data in refined_data.get("entities", []):
                entity = ExtractedEntity(
                    id=entity_data.get("id", f"entity_{len(entities)}"),
                    type=EntityType(entity_data.get("type", "activity")),
                    name=entity_data.get("name", "Unknown"),
                    description=entity_data.get("description"),
                    confidence=ConfidenceLevel(entity_data.get("confidence", "medium")),
                    source_text=entity_data.get("source_text"),
                    character_offsets=entity_data.get("character_offsets"),
                    is_implicit=entity_data.get("is_implicit", False),
                    is_uncertain=entity_data.get("is_uncertain", False),
                )
                entities.append(entity)

            # Parse refined relations
            relations: List[ExtractedRelation] = []
            for rel_data in refined_data.get("relations", []):
                relation = ExtractedRelation(
                    id=rel_data.get("id", f"relation_{len(relations)}"),
                    type=RelationType(rel_data.get("type", "precedes")),
                    source_id=rel_data.get("source_id"),
                    target_id=rel_data.get("target_id"),
                    label=rel_data.get("label"),
                    source_text=rel_data.get("source_text"),
                    is_implicit=rel_data.get("is_implicit", False),
                    is_conditional=rel_data.get("is_conditional", False),
                    condition_expression=rel_data.get("condition_expression"),
                )
                relations.append(relation)

            # Create refined result - reconstruct metadata as dict for validation
            metadata_dict = (
                original.metadata.model_dump()
                if hasattr(original.metadata, "model_dump")
                else original.metadata.dict()
            )
            result_dict = {
                "entities": entities,
                "relations": relations,
                "co_references": original.co_references,
                "metadata": metadata_dict,
            }
            return ExtractionResult.model_validate(result_dict)
        except Exception as e:
            logger.warning(f"Failed to apply refinement: {e}")
            return original


__all__ = [
    "ValidationIssueSeverity",
    "ValidationIssueType",
    "ValidationIssue",
    "CritiqueResult",
    "ExtractionValidator",
    "CritiqueAgent",
    "ExtractionRefinementPipeline",
]
