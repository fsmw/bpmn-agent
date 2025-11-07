"""
Refinement Tools for BPMN Agent

Provides intelligent refinement and improvement capabilities:
- Clarification request generation
- Improvement suggestions
- Stage re-execution with enhanced context
- Interactive refinement workflows
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from bpmn_agent.knowledge.domain_classifier import DomainClassifier, DomainType
from bpmn_agent.models.extraction import (
    ExtractionResultWithErrors,
)
from bpmn_agent.models.graph import ProcessGraph
from bpmn_agent.tools.graph_analysis import AnomalyType, GraphAnalyzer
from bpmn_agent.tools.validation import ValidationLevel, ValidationResult

logger = logging.getLogger(__name__)


class ClarificationType(str, Enum):
    """Types of clarifications to request."""

    ENTITY_IDENTIFICATION = "entity_identification"
    RELATION_AMBIGUITY = "relation_ambiguity"
    ACTOR_ASSIGNMENT = "actor_assignment"
    PROCESS_SCOPE = "process_scope"
    DECISION_LOGIC = "decision_logic"
    EVENT_CONDITIONS = "event_conditions"
    SEQUENCE_ORDER = "sequence_order"


class SuggestionCategory(str, Enum):
    """Categories of improvement suggestions."""

    PROCESS_STRUCTURE = "process_structure"
    ENTITY_EXTRACTION = "entity_extraction"
    RELATION_EXTRACTION = "relation_extraction"
    SEMANTIC_VALIDATION = "semantic_validation"
    BPMN_COMPLIANCE = "bpmn_compliance"
    DOMAIN_SPECIFIC = "domain_specific"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"


@dataclass
class ClarificationQuestion:
    """A clarification question to ask the user."""

    question_type: ClarificationType
    question: str
    context: str
    possible_answers: List[str] = field(default_factory=list)
    element_id: Optional[str] = None
    element_type: Optional[str] = None
    priority: str = "medium"  # low, medium, high, critical
    explanation: str = ""


@dataclass
class ImprovementSuggestion:
    """An improvement suggestion for the process."""

    category: SuggestionCategory
    title: str
    description: str
    rationale: str
    impact: str = "medium"  # low, medium, high
    confidence: float = 0.0
    element_id: Optional[str] = None
    element_type: Optional[str] = None
    suggested_changes: List[str] = field(default_factory=list)
    effort: str = "moderate"  # low, moderate, high


@dataclass
class RefinementPlan:
    """A complete refinement plan with prioritized actions."""

    plan_id: str
    created_at: datetime
    clarifications_needed: List[ClarificationQuestion] = field(default_factory=list)
    improvements_suggested: List[ImprovementSuggestion] = field(default_factory=list)
    stage_reexecutions: List[str] = field(default_factory=list)
    estimated_effort: str = "moderate"
    expected_improvement: str = "medium"


class ClarificationRequester:
    """Generates clarification questions for ambiguous extractions."""

    def __init__(self):
        """Initialize clarification requester."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def request_clarification(
        self,
        extraction_result: ExtractionResultWithErrors,
        validation_result: Optional[ValidationResult] = None,
        graph: Optional[ProcessGraph] = None,
    ) -> List[ClarificationQuestion]:
        """Generate clarification questions from extraction ambiguities.

        Args:
            extraction_result: Extraction result with potential ambiguities
            validation_result: Optional validation issues
            graph: Optional process graph for context

        Returns:
            List of clarification questions
        """
        questions = []

        # Questions about ambiguous entities
        questions.extend(self._generate_entity_questions(extraction_result))

        # Questions about missing relations
        questions.extend(self._generate_relation_questions(extraction_result))

        # Questions about actor assignments
        questions.extend(self._generate_actor_questions(extraction_result))

        # Questions from validation issues
        if validation_result:
            questions.extend(
                self._generate_validation_questions(extraction_result, validation_result)
            )

        # Questions from graph analysis
        if graph:
            questions.extend(self._generate_graph_questions(graph, extraction_result))

        # Sort by priority
        questions.sort(
            key=lambda q: {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(q.priority, 1)
        )

        self.logger.info(f"Generated {len(questions)} clarification questions")
        return questions

    def _generate_entity_questions(
        self, extraction_result: ExtractionResultWithErrors
    ) -> List[ClarificationQuestion]:
        """Generate questions about ambiguous entities."""
        questions = []

        # Low confidence entities
        low_confidence = [e for e in extraction_result.entities if e.confidence == "low"]
        for entity in low_confidence[:3]:  # Limit to first 3
            questions.append(
                ClarificationQuestion(
                    question_type=ClarificationType.ENTITY_IDENTIFICATION,
                    question=f"Is '{entity.name}' actually a {entity.type} in this process?",
                    context=entity.source_context or "",
                    possible_answers=[
                        "Yes",
                        "No, it's a different activity",
                        "No, it's not part of the process",
                    ],
                    element_id=entity.identifier,
                    element_type=entity.type,
                    priority="medium",
                    explanation="Entity has low confidence in extraction",
                )
            )

        # Generic entity types
        generic_entities = [
            e
            for e in extraction_result.entities
            if e.type.lower() in ["activity", "task", "process", "system"]
        ]
        for entity in generic_entities[:2]:  # Limit to first 2
            questions.append(
                ClarificationQuestion(
                    question_type=ClarificationType.ENTITY_IDENTIFICATION,
                    question=f"What specific type of activity is '{entity.name}'?",
                    context=entity.source_context or "",
                    possible_answers=[
                        "Human task",
                        "Automated task",
                        "Decision point",
                        "External system action",
                    ],
                    element_id=entity.identifier,
                    element_type=entity.type,
                    priority="high",
                    explanation="Specific entity type improves process accuracy",
                )
            )

        return questions

    def _generate_relation_questions(
        self, extraction_result: ExtractionResultWithErrors
    ) -> List[ClarificationQuestion]:
        """Generate questions about relation ambiguities."""
        questions = []

        # Check for disconnected entities
        entities = set(f"{e.name}_{e.type}" for e in extraction_result.entities)
        connected = set()
        for rel in extraction_result.relations:
            connected.add(f"{rel.source_name}_{rel.source_type}")
            connected.add(f"{rel.target_name}_{rel.target_type}")

        disconnected = entities - connected
        for entity_key in list(disconnected)[:3]:  # Limit to first 3
            entity_name = entity_key.rsplit("_", 1)[0]
            entity_type = entity_key.rsplit("_", 1)[1]
            questions.append(
                ClarificationQuestion(
                    question_type=ClarificationType.RELATION_AMBIGUITY,
                    question=f"How does '{entity_name}' connect to other process elements?",
                    context="Entity appears isolated in current extraction",
                    possible_answers=[
                        "Start of process",
                        "End of process",
                        "Connected to specific step",
                        "Not part of main flow",
                    ],
                    element_type=entity_type,
                    priority="high",
                    explanation="Disconnected entities may break process flow",
                )
            )

        return questions

    def _generate_actor_questions(
        self, extraction_result: ExtractionResultWithErrors
    ) -> List[ClarificationQuestion]:
        """Generate questions about actor assignments."""
        questions = []

        # Tasks without actors
        tasks = [e for e in extraction_result.entities if e.type.lower() in ["task", "activity"]]
        actorless_tasks = [
            t
            for t in tasks
            if not any(
                "actor" in r.type.lower()
                for r in extraction_result.relations
                if r.source_name == t.name
            )
        ]

        for task in actorless_tasks[:3]:  # Limit to first 3
            questions.append(
                ClarificationQuestion(
                    question_type=ClarificationType.ACTOR_ASSIGNMENT,
                    question=f"Who performs the task '{task.name}'?",
                    context=task.source_context or "",
                    possible_answers=["User", "System", "Manager", "Team", "Unknown"],
                    element_id=task.identifier,
                    element_type=task.type,
                    priority="medium",
                    explanation="Actor assignment improves swimlane organization",
                )
            )

        return questions

    def _generate_validation_questions(
        self, extraction_result: ExtractionResultWithErrors, validation_result: ValidationResult
    ) -> List[ClarificationQuestion]:
        """Generate questions from validation issues."""
        questions = []

        for issue in validation_result.issues:
            if (
                issue.category.value == "extraction_quality"
                and issue.level == ValidationLevel.WARNING
            ):
                if "No entities extracted" in issue.message:
                    questions.append(
                        ClarificationQuestion(
                            question_type=ClarificationType.PROCESS_SCOPE,
                            question="Does the input text actually contain a process description?",
                            context="No clear process elements were extracted",
                            possible_answers=[
                                "Yes, it's a complete process",
                                "Yes, but it's very simple",
                                "No, it's just general text",
                            ],
                            priority="critical",
                            explanation="Process descriptions needed for extraction",
                        )
                    )

        return questions

    def _generate_graph_questions(
        self, graph: ProcessGraph, extraction_result: ExtractionResultWithErrors
    ) -> List[ClarificationQuestion]:
        """Generate questions from graph analysis issues."""
        questions = []

        analyzer = GraphAnalyzer()
        orphaned = analyzer.find_orphaned_nodes(graph)

        for orphan_id in orphaned[:2]:  # Limit to first 2
            node = next((n for n in graph.nodes if n.id == orphan_id), None)
            if node:
                questions.append(
                    ClarificationQuestion(
                        question_type=ClarificationType.SEQUENCE_ORDER,
                        question=f"Should '{node.label or node.id}' be included in the main process flow?",
                        context="Node is not reachable from start events",
                        possible_answers=[
                            "Yes, add connection from start",
                            "Yes, add connection from another step",
                            "No, remove it",
                        ],
                        element_id=orphan_id,
                        element_type=node.type,
                        priority="medium",
                        explanation="Orphaned nodes indicate incomplete process connections",
                    )
                )

        return questions


class ImprovementSuggester:
    """Generates improvement suggestions for process extraction."""

    def __init__(self):
        """Initialize improvement suggester."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.domain_classifier = DomainClassifier()

    def suggest_improvements(
        self,
        extraction_result: ExtractionResultWithErrors,
        validation_result: Optional[ValidationResult] = None,
        graph: Optional[ProcessGraph] = None,
        original_text: str = "",
    ) -> List[ImprovementSuggestion]:
        """Generate improvement suggestions from analysis.

        Args:
            extraction_result: Current extraction result
            validation_result: Optional validation result
            graph: Optional process graph
            original_text: Original input text

        Returns:
            List of improvement suggestions
        """
        suggestions = []

        # Process structure suggestions
        suggestions.extend(self._suggest_structure_improvements(extraction_result, graph))

        # Entity extraction suggestions
        suggestions.extend(self._suggest_entity_improvements(extraction_result))

        # Validation-based suggestions
        if validation_result:
            suggestions.extend(self._suggest_validation_improvements(validation_result))

        # Domain-specific suggestions
        suggestions.extend(self._suggest_domain_improvements(extraction_result, original_text))

        # Performance suggestions
        suggestions.extend(self._suggest_performance_improvements(extraction_result))

        # Sort by confidence and priority
        suggestions.sort(
            key=lambda s: (s.confidence, {"high": 3, "medium": 2, "low": 1}.get(s.impact, 2)),
            reverse=True,
        )

        self.logger.info(f"Generated {len(suggestions)} improvement suggestions")
        return suggestions

    def _suggest_structure_improvements(
        self, extraction_result: ExtractionResultWithErrors, graph: Optional[ProcessGraph]
    ) -> List[ImprovementSuggestion]:
        """Suggest process structure improvements."""
        suggestions = []

        if graph:
            # Use graph analysis to identify issues
            analyzer = GraphAnalyzer()
            analysis = analyzer.analyze_graph_structure(graph, extraction_result)

            # Suggest fixing isolated nodes
            for anomaly in analysis.anomalies:
                if anomaly.anomaly_type == AnomalyType.ISOLATED_NODE:
                    suggestions.append(
                        ImprovementSuggestion(
                            category=SuggestionCategory.PROCESS_STRUCTURE,
                            title="Connect Isolated Nodes",
                            description=f"Node {anomaly.node_id} is not connected to the main process flow",
                            rationale="Isolated nodes break process continuity and indicate extraction issues",
                            impact="medium",
                            confidence=0.8,
                            element_id=anomaly.node_id,
                            suggested_changes=[
                                "Check if entity was properly extracted",
                                "Add connecting sequence flows to main process",
                                "Review original text for missed relationships",
                            ],
                            effort="low",
                        )
                    )
                elif anomaly.anomaly_type == AnomalyType.UNCLOSED_GATEWAY:
                    suggestions.append(
                        ImprovementSuggestion(
                            category=SuggestionCategory.PROCESS_STRUCTURE,
                            title="Close Gateway Branches",
                            description=f"Gateway at {anomaly.node_id} has unclosed branches",
                            rationale="Unclosed gateways create incomplete process flows",
                            impact="high",
                            confidence=0.9,
                            element_id=anomaly.node_id,
                            suggested_changes=[
                                "Add converging gateway to close parallel flows",
                                "Ensure all decision branches reach proper endpoints",
                            ],
                            effort="moderate",
                        )
                    )

        # Suggest adding start/end events if missing
        entities_by_type = {}
        for entity in extraction_result.entities:
            entities_by_type.setdefault(entity.type.lower(), []).append(entity)

        if "start" not in entities_by_type and entities_by_type:
            suggestions.append(
                ImprovementSuggestion(
                    category=SuggestionCategory.PROCESS_STRUCTURE,
                    title="Add Start Event",
                    description="No clear start event identified in the process",
                    rationale="BPMN processes should begin with explicit start events",
                    impact="medium",
                    confidence=0.7,
                    suggested_changes=[
                        "Identify the initial trigger in the process",
                        "Add start event at process beginning",
                        "Connect start event to first activity",
                    ],
                    effort="low",
                )
            )

        return suggestions

    def _suggest_entity_improvements(
        self, extraction_result: ExtractionResultWithErrors
    ) -> List[ImprovementSuggestion]:
        """Suggest entity extraction improvements."""
        suggestions = []

        # Low confidence entities
        low_confidence = [e for e in extraction_result.entities if e.confidence == "low"]
        if len(low_confidence) > len(extraction_result.entities) * 0.3:
            suggestions.append(
                ImprovementSuggestion(
                    category=SuggestionCategory.ENTITY_EXTRACTION,
                    title="Improve Entity Confidence",
                    description=f"High number of low-confidence entities: {len(low_confidence)}/{len(extraction_result.entities)}",
                    rationale="Low confidence extraction reduces process quality",
                    impact="high",
                    confidence=0.8,
                    suggested_changes=[
                        "Refine extraction prompts with domain-specific examples",
                        "Add more context about entity types to extract",
                        "Consider providing clearer input text descriptions",
                    ],
                    effort="moderate",
                )
            )

        # Missing activity types
        generic_types = ["activity", "task", "process", "system"]
        generic_entities = [
            e for e in extraction_result.entities if e.type.lower() in generic_types
        ]
        if generic_entities:
            suggestions.append(
                ImprovementSuggestion(
                    category=SuggestionCategory.ENTITY_EXTRACTION,
                    title="Specify Entity Types",
                    description=f"Found {len(generic_entities)} entities with generic types",
                    rationale="Specific entity types improve BPMN element accuracy",
                    impact="medium",
                    confidence=0.7,
                    suggested_changes=[
                        "Use more descriptive business terminology",
                        "Clarify the function of each activity",
                        "Specify human vs automated task types",
                    ],
                    effort="low",
                )
            )

        return suggestions

    def _suggest_validation_improvements(
        self, validation_result: ValidationResult
    ) -> List[ImprovementSuggestion]:
        """Suggest improvements based on validation issues."""
        suggestions = []

        if not validation_result.is_valid:
            critical_issues = [
                i for i in validation_result.issues if i.level == ValidationLevel.CRITICAL
            ]
            if critical_issues:
                suggestions.append(
                    ImprovementSuggestion(
                        category=SuggestionCategory.BPMN_COMPLIANCE,
                        title="Fix Critical Validation Issues",
                        description=f"Found {len(critical_issues)} critical validation problems",
                        rationale="Critical issues prevent valid BPMN generation",
                        impact="high",
                        confidence=0.9,
                        suggested_changes=[
                            "Address XML structure problems",
                            "Fix connectivity issues",
                            "Ensure proper BPMN element usage",
                        ],
                        effort="high",
                    )
                )

        return suggestions

    def _suggest_domain_improvements(
        self, extraction_result: ExtractionResultWithErrors, original_text: str
    ) -> List[ImprovementSuggestion]:
        """Suggest domain-specific improvements."""
        suggestions = []

        # Detect domain
        try:
            domain_result = self.domain_classifier.classify_domain(original_text)
            if domain_result:
                domain = domain_result.domain
            else:
                domain = DomainType.GENERIC
        except Exception:
            domain = DomainType.GENERIC

        # Domain-specific suggestions
        if domain == DomainType.HR:
            suggestions.append(
                ImprovementSuggestion(
                    category=SuggestionCategory.DOMAIN_SPECIFIC,
                    title="Enhance HR Process Details",
                    description="Add HR-specific elements and terminology",
                    rationale="HR processes benefit from standard HR terminology",
                    impact="medium",
                    confidence=0.6,
                    suggested_changes=[
                        "Specify HR system names",
                        "Include approval hierarchies",
                        "Add compliance checkpoints",
                    ],
                    effort="low",
                )
            )
        elif domain == DomainType.FINANCE:
            suggestions.append(
                ImprovementSuggestion(
                    category=SuggestionCategory.DOMAIN_SPECIFIC,
                    title="Add Financial Controls",
                    description="Include financial validation and control points",
                    rationale="Finance processes require proper financial controls",
                    impact="high",
                    confidence=0.7,
                    suggested_changes=[
                        "Add approval monetary thresholds",
                        "Include validation stages",
                        "Specify audit checkpoints",
                    ],
                    effort="moderate",
                )
            )

        return suggestions

    def _suggest_performance_improvements(
        self, extraction_result: ExtractionResultWithErrors
    ) -> List[ImprovementSuggestion]:
        """Suggest performance optimization improvements."""
        suggestions = []

        if extraction_result.metadata and extraction_result.metadata.extraction_duration_ms:
            duration = extraction_result.metadata.extraction_duration_ms
            if duration > 30000:  # 30 seconds
                suggestions.append(
                    ImprovementSuggestion(
                        category=SuggestionCategory.PERFORMANCE_OPTIMIZATION,
                        title="Optimize Extraction Performance",
                        description=f"Extraction took {duration:.0f}ms, which is slower than optimal",
                        rationale="Faster extraction improves user experience",
                        impact="medium",
                        confidence=0.6,
                        suggested_changes=[
                            "Consider using smaller text chunks",
                            "Optimize LLM prompts for faster responses",
                            "Use more efficient model for basic extractions",
                        ],
                        effort="moderate",
                    )
                )

        return suggestions


class StageRerunner:
    """Handles intelligently re-executing pipeline stages."""

    def __init__(self):
        """Initialize stage rerunner."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def rerun_stage(
        self,
        stage_name: str,
        original_result: Any,
        improvements: List[ImprovementSuggestion],
        clarifications: List[ClarificationQuestion],
        original_text: str = "",
    ) -> Tuple[Any, bool]:
        """Re-run a specific stage with improved context.

        Args:
            stage_name: Name of stage to re-run
            original_result: Original result from the stage
            improvements: Improvements to apply
            clarifications: Clarifications that can help
            original_text: Original input text

        Returns:
            (new_result, improvement) tuple
        """
        self.logger.info(f"Re-running stage: {stage_name}")

        # Build enhanced context
        enhanced_context = self._build_enhanced_context(
            stage_name, improvements, clarifications, original_text
        )

        # For now, return the original result (implementation would depend on stage integration)
        # In a full implementation, this would:
        # 1. Update extraction prompts with clarifications
        # 2. Apply suggested improvements to parameters
        # 3. Re-execute the stage with better context
        # 4. Compare results and return improvement indicator

        improvement_indicated = len(improvements) > 0 or len(clarifications) > 0

        self.logger.info(
            f"Stage {stage_name} re-run completed, improvement indicated: {improvement_indicated}"
        )

        return original_result, improvement_indicated

    def _build_enhanced_context(
        self,
        stage_name: str,
        improvements: List[ImprovementSuggestion],
        clarifications: List[ClarificationQuestion],
        original_text: str,
    ) -> Dict[str, Any]:
        """Build enhanced context for stage re-execution."""
        context = {
            "stage_name": stage_name,
            "original_text": original_text,
            "improvements_to_apply": [s.description for s in improvements],
            "clarifications_available": [q.question for q in clarifications],
            "rerun_reason": "apply improvements and clarifications",
        }

        # Add stage-specific context
        if stage_name == "entity_extraction":
            context.update(
                {
                    "focus_on_entity_confidence": True,
                    "clarify_generic_types": True,
                    "include_actor_assignments": True,
                }
            )
        elif stage_name == "graph_construction":
            context.update(
                {
                    "ensure_connectivity": True,
                    "validate_gateways": True,
                    "check_start_end_events": True,
                }
            )

        return context


class RefinementOrchestrator:
    """Coordinates refinement activities."""

    def __init__(self):
        """Initialize refinement orchestrator."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.clarification_requester = ClarificationRequester()
        self.improvement_suggester = ImprovementSuggester()
        self.stage_rerunner = StageRerunner()

    async def create_refinement_plan(
        self,
        extraction_result: ExtractionResultWithErrors,
        validation_result: Optional[ValidationResult] = None,
        graph: Optional[ProcessGraph] = None,
        original_text: str = "",
    ) -> RefinementPlan:
        """Create a comprehensive refinement plan.

        Args:
            extraction_result: Current extraction result
            validation_result: Optional validation result
            graph: Optional process graph
            original_text: Original input text

        Returns:
            Complete refinement plan
        """
        import uuid

        plan_id = str(uuid.uuid4())

        plan = RefinementPlan(plan_id=plan_id, created_at=datetime.now())

        # Generate clarifications
        plan.clarifications_needed = self.clarification_requester.request_clarification(
            extraction_result, validation_result, graph
        )

        # Generate improvements
        plan.improvements_suggested = self.improvement_suggester.suggest_improvements(
            extraction_result, validation_result, graph, original_text
        )

        # Determine stage re-executions needed
        for improvement in plan.improvements_suggested:
            if improvement.category in [
                SuggestionCategory.ENTITY_EXTRACTION,
                SuggestionCategory.RELATION_EXTRACTION,
            ]:
                if "entity_extraction" not in plan.stage_reexecutions:
                    plan.stage_reexecutions.append("entity_extraction")

            elif improvement.category in [
                SuggestionCategory.PROCESS_STRUCTURE,
                SuggestionCategory.SEMANTIC_VALIDATION,
            ]:
                if "graph_construction" not in plan.stage_reexecutions:
                    plan.stage_reexecutions.append("graph_construction")

            elif improvement.category == SuggestionCategory.BPMN_COMPLIANCE:
                if "xml_generation" not in plan.stage_reexecutions:
                    plan.stage_reexecutions.append("xml_generation")

        # Estimate effort and impact
        plan.estimated_effort = self._estimate_effort(plan)
        plan.expected_improvement = self._estimate_improvement(plan)

        self.logger.info(
            f"Created refinement plan {plan_id}: {len(plan.clarifications_needed)} clarifications, {len(plan.improvements_suggested)} improvements"
        )

        return plan

    def _estimate_effort(self, plan: RefinementPlan) -> str:
        """Estimate total effort for refinement plan."""
        total_score = 0
        total_score += len(plan.clarifications_needed) * 1
        total_score += len(plan.improvements_suggested) * 2
        total_score += len(plan.stage_reexecutions) * 5

        if total_score <= 5:
            return "low"
        elif total_score <= 15:
            return "moderate"
        else:
            return "high"

    def _estimate_improvement(self, plan: RefinementPlan) -> str:
        """Estimate expected improvement impact."""
        high_impact_suggestions = [s for s in plan.improvements_suggested if s.impact == "high"]
        critical_questions = [q for q in plan.clarifications_needed if q.priority == "critical"]

        if high_impact_suggestions and critical_questions:
            return "high"
        elif len(plan.improvements_suggested) > 3:
            return "medium"
        else:
            return "low"


# Convenience functions for refinement
def request_clarification(
    extraction_result: ExtractionResultWithErrors,
    validation_result: Optional[ValidationResult] = None,
    graph: Optional[ProcessGraph] = None,
) -> List[ClarificationQuestion]:
    """Convenience function to request clarifications."""
    requester = ClarificationRequester()
    return requester.request_clarification(extraction_result, validation_result, graph)


def suggest_improvements(
    extraction_result: ExtractionResultWithErrors,
    validation_result: Optional[ValidationResult] = None,
    graph: Optional[ProcessGraph] = None,
    original_text: str = "",
) -> List[ImprovementSuggestion]:
    """Convenience function to suggest improvements."""
    suggester = ImprovementSuggester()
    return suggester.suggest_improvements(
        extraction_result, validation_result, graph, original_text
    )


async def rerun_stage(
    stage_name: str,
    original_result: Any,
    improvements: List[ImprovementSuggestion],
    clarifications: List[ClarificationQuestion],
    original_text: str = "",
) -> Tuple[Any, bool]:
    """Convenience function to re-run a stage."""
    rerunner = StageRerunner()
    return await rerunner.rerun_stage(
        stage_name, original_result, improvements, clarifications, original_text
    )


async def create_refinement_plan(
    extraction_result: ExtractionResultWithErrors,
    validation_result: Optional[ValidationResult] = None,
    graph: Optional[ProcessGraph] = None,
    original_text: str = "",
) -> RefinementPlan:
    """Convenience function to create refinement plan."""
    orchestrator = RefinementOrchestrator()
    return await orchestrator.create_refinement_plan(
        extraction_result, validation_result, graph, original_text
    )
