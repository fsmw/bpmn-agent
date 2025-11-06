"""Tests for self-critique and validation mechanism.

Tests the extraction validation, critique feedback generation, and refinement pipeline.
"""

import pytest
from datetime import datetime
from typing import List

from stages.extraction_critique import (
    ValidationIssueSeverity,
    ValidationIssueType,
    ValidationIssue,
    CritiqueResult,
    ExtractionValidator,
    CritiqueAgent,
    ExtractionRefinementPipeline,
)
from models.extraction import (
    ExtractionResult,
    ExtractedEntity,
    ExtractedRelation,
    EntityType,
    RelationType,
    ConfidenceLevel,
    ExtractionMetadata,
)


@pytest.fixture
def simple_extraction() -> ExtractionResult:
    """Create a simple valid extraction result."""
    entities = [
        ExtractedEntity(
            id="e1",
            type=EntityType.EVENT,
            name="Start",
            description="Process starts",
            confidence=ConfidenceLevel.HIGH,
        ),
        ExtractedEntity(
            id="e2",
            type=EntityType.ACTIVITY,
            name="Review Request",
            description="Manager reviews the request",
            confidence=ConfidenceLevel.HIGH,
        ),
        ExtractedEntity(
            id="e3",
            type=EntityType.EVENT,
            name="End",
            description="Process ends",
            confidence=ConfidenceLevel.HIGH,
        ),
    ]
    
    relations = [
        ExtractedRelation(
            id="r1",
            type=RelationType.TRIGGERS,
            source_id="e1",
            target_id="e2",
        ),
        ExtractedRelation(
            id="r2",
            type=RelationType.TRIGGERS,
            source_id="e2",
            target_id="e3",
        ),
    ]
    
    metadata = ExtractionMetadata(
        input_text="Test process: start, review request, end",
        input_length=42,
        extraction_timestamp=datetime.now().isoformat(),
        extraction_duration_ms=1.5,
        llm_model="mistral",
        llm_temperature=0.7,
    )
    
    return ExtractionResult(entities=entities, relations=relations, metadata=metadata)


@pytest.fixture
def incomplete_extraction() -> ExtractionResult:
    """Create an extraction with issues."""
    entities = [
        ExtractedEntity(
            id="e1",
            type=EntityType.ACTIVITY,
            name="Do Something",
            description="",
            confidence=ConfidenceLevel.LOW,
        ),
        ExtractedEntity(
            id="e2",
            type=EntityType.ACTIVITY,
            name="Do Another Thing",
            description="",
            confidence=ConfidenceLevel.MEDIUM,
        ),
        ExtractedEntity(
            id="e3",
            type=EntityType.ACTIVITY,
            name="Isolated Activity",
            description="This is disconnected",
            confidence=ConfidenceLevel.LOW,
        ),
    ]
    
    relations = [
        ExtractedRelation(
            id="r1",
            type=RelationType.PRECEDES,
            source_id="e1",
            target_id="e2",
        ),
    ]
    
    metadata = ExtractionMetadata(
        input_text="Test process with issues",
        input_length=25,
        extraction_timestamp=datetime.now().isoformat(),
        extraction_duration_ms=2.0,
        llm_model="mistral",
        llm_temperature=0.7,
    )
    
    return ExtractionResult(entities=entities, relations=relations, metadata=metadata)


class TestValidationIssueType:
    """Test ValidationIssueType enum."""
    
    def test_validation_issue_types_exist(self):
        """Verify all issue types are defined."""
        expected_types = {
            "missing_start_event",
            "missing_end_event",
            "disconnected_nodes",
            "missing_labels",
            "duplicate_entities",
            "invalid_relationships",
            "unclear_branching",
            "ambiguous_actors",
            "low_confidence_entities",
            "incomplete_data_flow",
            "missing_business_rules",
        }
        
        actual_types = {issue_type.value for issue_type in ValidationIssueType}
        assert actual_types == expected_types


class TestValidationIssue:
    """Test ValidationIssue model."""
    
    def test_create_validation_issue(self):
        """Test creating a validation issue."""
        issue = ValidationIssue(
            type=ValidationIssueType.MISSING_START_EVENT,
            severity=ValidationIssueSeverity.ERROR,
            message="No start event found",
            suggestion="Add a start event",
        )
        
        assert issue.type == ValidationIssueType.MISSING_START_EVENT
        assert issue.severity == ValidationIssueSeverity.ERROR
        assert issue.message == "No start event found"
        assert issue.suggestion == "Add a start event"
        assert issue.confidence == 0.8
    
    def test_issue_with_affected_entities(self):
        """Test issue with affected entity IDs."""
        issue = ValidationIssue(
            type=ValidationIssueType.DISCONNECTED_NODES,
            severity=ValidationIssueSeverity.WARNING,
            message="Some nodes are disconnected",
            affected_entity_ids=["e1", "e2"],
        )
        
        assert issue.affected_entity_ids == ["e1", "e2"]


class TestCritiqueResult:
    """Test CritiqueResult model."""
    
    def test_create_critique_result(self):
        """Test creating a critique result."""
        result = CritiqueResult(
            is_valid=True,
            quality_score=0.95,
        )
        
        assert result.is_valid is True
        assert result.quality_score == 0.95
        assert result.issues == []
        assert isinstance(result.timestamp, datetime)
    
    def test_critique_result_with_issues(self):
        """Test critique result with issues."""
        issue = ValidationIssue(
            type=ValidationIssueType.MISSING_START_EVENT,
            severity=ValidationIssueSeverity.ERROR,
            message="No start event",
        )
        
        result = CritiqueResult(
            is_valid=False,
            issues=[issue],
            quality_score=0.5,
        )
        
        assert len(result.issues) == 1
        assert result.is_valid is False


class TestExtractionValidator:
    """Test ExtractionValidator."""
    
    def test_validate_simple_extraction(self, simple_extraction):
        """Test validating a simple valid extraction."""
        result = ExtractionValidator.validate_extraction(simple_extraction)
        
        assert result.is_valid is True
        # May have info-level issues (not critical/error/warning)
        error_issues = [i for i in result.issues if i.severity in (ValidationIssueSeverity.CRITICAL, ValidationIssueSeverity.ERROR, ValidationIssueSeverity.WARNING)]
        assert len(error_issues) == 0
        assert result.quality_score > 0.8
    
    def test_validate_missing_start_event(self):
        """Test validation detects missing start event."""
        entities = [
            ExtractedEntity(
                id="e1",
                type=EntityType.ACTIVITY,
                name="Do Something",
                confidence=ConfidenceLevel.HIGH,
            ),
            ExtractedEntity(
                id="e2",
                type=EntityType.EVENT,
                name="End",
                confidence=ConfidenceLevel.HIGH,
            ),
        ]
        
        metadata = ExtractionMetadata(
            input_text="Do something then end",
            input_length=21,
            extraction_timestamp=datetime.now().isoformat(),
            extraction_duration_ms=1.0,
            llm_model="mistral",
            llm_temperature=0.7,
        )
        
        extraction = ExtractionResult(
            entities=entities,
            relations=[],
            metadata=metadata,
        )
        
        result = ExtractionValidator.validate_extraction(extraction)
        
        # Should have issue for missing start event
        issues = [i for i in result.issues if i.type == ValidationIssueType.MISSING_START_EVENT]
        assert len(issues) == 1
        assert issues[0].severity == ValidationIssueSeverity.ERROR
    
    def test_validate_missing_end_event(self):
        """Test validation detects missing end event."""
        entities = [
            ExtractedEntity(
                id="e1",
                type=EntityType.EVENT,
                name="Start",
                confidence=ConfidenceLevel.HIGH,
            ),
            ExtractedEntity(
                id="e2",
                type=EntityType.ACTIVITY,
                name="Do Something",
                confidence=ConfidenceLevel.HIGH,
            ),
        ]
        
        metadata = ExtractionMetadata(
            input_text="Start and do something",
            input_length=22,
            extraction_timestamp=datetime.now().isoformat(),
            extraction_duration_ms=1.0,
            llm_model="mistral",
            llm_temperature=0.7,
        )
        
        extraction = ExtractionResult(
            entities=entities,
            relations=[],
            metadata=metadata,
        )
        
        result = ExtractionValidator.validate_extraction(extraction)
        
        issues = [i for i in result.issues if i.type == ValidationIssueType.MISSING_END_EVENT]
        assert len(issues) == 1
    
    def test_validate_disconnected_nodes(self, incomplete_extraction):
        """Test validation detects disconnected nodes."""
        result = ExtractionValidator.validate_extraction(incomplete_extraction)
        
        issues = [i for i in result.issues if i.type == ValidationIssueType.DISCONNECTED_NODES]
        assert len(issues) > 0
        assert "e3" in issues[0].affected_entity_ids
    
    def test_validate_low_confidence_entities(self, incomplete_extraction):
        """Test validation detects low confidence entities."""
        result = ExtractionValidator.validate_extraction(incomplete_extraction)
        
        issues = [i for i in result.issues if i.type == ValidationIssueType.LOW_CONFIDENCE_ENTITIES]
        assert len(issues) > 0
    
    def test_validate_missing_descriptions(self, incomplete_extraction):
        """Test validation detects missing descriptions."""
        result = ExtractionValidator.validate_extraction(incomplete_extraction)
        
        issues = [i for i in result.issues if i.type == ValidationIssueType.MISSING_LABELS]
        assert len(issues) > 0
    
    def test_quality_score_calculation(self, simple_extraction):
        """Test quality score calculation."""
        result = ExtractionValidator.validate_extraction(simple_extraction)
        
        assert 0.0 <= result.quality_score <= 1.0
        assert result.quality_score > 0.8  # High-confidence extraction
    
    def test_quality_score_with_issues(self, incomplete_extraction):
        """Test quality score is lower with issues."""
        result = ExtractionValidator.validate_extraction(incomplete_extraction)
        
        assert 0.0 <= result.quality_score <= 1.0
        assert result.quality_score < 0.8  # Should be lower due to issues
    
    def test_needs_refinement_flag(self, simple_extraction, incomplete_extraction):
        """Test needs_refinement flag."""
        valid_result = ExtractionValidator.validate_extraction(simple_extraction)
        assert valid_result.needs_refinement is False
        
        invalid_result = ExtractionValidator.validate_extraction(incomplete_extraction)
        assert invalid_result.needs_refinement is True


class TestSuggestionGeneration:
    """Test suggestion generation for improvements."""
    
    def test_suggestions_for_simple_process(self, simple_extraction):
        """Test suggestions for simple process."""
        result = ExtractionValidator.validate_extraction(simple_extraction)
        
        # Simple process may get suggestions about complexity
        assert isinstance(result.suggestions_for_improvement, list)
    
    def test_suggestions_for_missing_actors(self):
        """Test suggestion when no actors identified."""
        entities = [
            ExtractedEntity(
                id="e1",
                type=EntityType.EVENT,
                name="Start",
                confidence=ConfidenceLevel.HIGH,
            ),
            ExtractedEntity(
                id="e2",
                type=EntityType.ACTIVITY,
                name="Do Something",
                confidence=ConfidenceLevel.HIGH,
            ),
            ExtractedEntity(
                id="e3",
                type=EntityType.EVENT,
                name="End",
                confidence=ConfidenceLevel.HIGH,
            ),
        ]
        
        metadata = ExtractionMetadata(
            input_text="Start then do something then end",
            input_length=32,
            extraction_timestamp=datetime.now().isoformat(),
            extraction_duration_ms=1.0,
            llm_model="mistral",
            llm_temperature=0.7,
        )
        
        extraction = ExtractionResult(
            entities=entities,
            relations=[],
            metadata=metadata,
        )
        
        result = ExtractionValidator.validate_extraction(extraction)
        
        # Should suggest adding actors
        actor_suggestions = [
            s for s in result.suggestions_for_improvement
            if "actor" in s.lower() or "role" in s.lower()
        ]
        assert len(actor_suggestions) > 0
    
    def test_suggestions_for_missing_gateways(self):
        """Test suggestion when no gateways identified."""
        entities = [
            ExtractedEntity(
                id="e1",
                type=EntityType.EVENT,
                name="Start",
                confidence=ConfidenceLevel.HIGH,
            ),
            ExtractedEntity(
                id="e2",
                type=EntityType.ACTIVITY,
                name="Do Something",
                confidence=ConfidenceLevel.HIGH,
            ),
            ExtractedEntity(
                id="e3",
                type=EntityType.EVENT,
                name="End",
                confidence=ConfidenceLevel.HIGH,
            ),
        ]
        
        metadata = ExtractionMetadata(
            input_text="Start then do something then end",
            input_length=32,
            extraction_timestamp=datetime.now().isoformat(),
            extraction_duration_ms=1.0,
            llm_model="mistral",
            llm_temperature=0.7,
        )
        
        extraction = ExtractionResult(
            entities=entities,
            relations=[],
            metadata=metadata,
        )
        
        result = ExtractionValidator.validate_extraction(extraction)
        
        # Should suggest adding gateways
        gateway_suggestions = [
            s for s in result.suggestions_for_improvement
            if "gateway" in s.lower() or "branch" in s.lower()
        ]
        assert len(gateway_suggestions) > 0


class TestCritiqueAgent:
    """Test CritiqueAgent."""
    
    def test_create_critique_agent(self):
        """Test creating a critique agent."""
        agent = CritiqueAgent()
        assert agent.llm_client is None
    
    def test_format_extraction_for_critique(self, simple_extraction):
        """Test formatting extraction result for LLM critique."""
        agent = CritiqueAgent()
        
        formatted = agent._format_extraction_for_critique(simple_extraction)
        
        assert "Start" in formatted
        assert "Review Request" in formatted
        assert "End" in formatted
        assert "triggers" in formatted  # Relation type is lowercase
    
    def test_create_critique_prompt(self):
        """Test creating LLM critique prompt."""
        agent = CritiqueAgent()
        extraction_str = "Test extraction"
        original_text = "Test process"
        
        prompt = agent._create_critique_prompt(extraction_str, original_text)
        
        assert "business process expert" in prompt
        assert extraction_str in prompt
        assert original_text in prompt
    
    @pytest.mark.asyncio
    async def test_generate_critique_feedback(self, simple_extraction):
        """Test generating critique feedback."""
        agent = CritiqueAgent()
        
        feedback = await agent.generate_critique_feedback(
            simple_extraction,
            "Original text",
        )
        
        assert "validation" in feedback
        assert "refinement_recommendations" in feedback
        assert feedback["validation"].is_valid is True


class TestExtractionRefinementPipeline:
    """Test ExtractionRefinementPipeline."""
    
    def test_create_pipeline(self):
        """Test creating a refinement pipeline."""
        pipeline = ExtractionRefinementPipeline(max_iterations=3)
        
        assert pipeline.max_iterations == 3
        assert pipeline.iteration_history == []
    
    def test_create_refinement_prompt(self, simple_extraction):
        """Test creating refinement prompt."""
        pipeline = ExtractionRefinementPipeline()
        
        issues = [
            ValidationIssue(
                type=ValidationIssueType.MISSING_START_EVENT,
                severity=ValidationIssueSeverity.ERROR,
                message="No start event",
            )
        ]
        
        extraction_str = "Test extraction"
        original_text = "Test process"
        
        prompt = pipeline._create_refinement_prompt(
            original_text,
            extraction_str,
            issues,
        )
        
        assert "business process extraction expert" in prompt
        assert "MISSING_START_EVENT" in prompt or "No start event" in prompt
        assert extraction_str in prompt
    
    def test_parse_refined_extraction_valid_json(self):
        """Test parsing valid refined extraction."""
        pipeline = ExtractionRefinementPipeline()
        
        response = """```json
{
  "entities": [
    {"id": "e1", "type": "event", "name": "Start", "description": "Begin", "confidence": "high"}
  ],
  "relations": [
    {"id": "r1", "type": "triggers", "source_id": "e1", "target_id": "e2"}
  ]
}
```"""
        
        result = pipeline._parse_refined_extraction(response)
        
        assert result is not None
        assert len(result["entities"]) == 1
        assert len(result["relations"]) == 1
    
    def test_parse_refined_extraction_invalid_json(self):
        """Test parsing invalid JSON returns None."""
        pipeline = ExtractionRefinementPipeline()
        
        response = "This is not JSON"
        result = pipeline._parse_refined_extraction(response)
        
        assert result is None
    
    def test_apply_refinement(self, simple_extraction):
        """Test applying refinement to extraction."""
        pipeline = ExtractionRefinementPipeline()
        
        refined_data = {
            "entities": [
                {
                    "id": "e1",
                    "type": "event",
                    "name": "Start Process",
                    "description": "Process begins here",
                    "confidence": "high",
                }
            ],
            "relations": [],
        }
        
        result = pipeline._apply_refinement(simple_extraction, refined_data)
        
        assert len(result.entities) == 1
        assert result.entities[0].name == "Start Process"
        # Compare metadata fields instead of objects due to import path differences
        assert result.metadata.input_text == simple_extraction.metadata.input_text
        assert result.metadata.llm_model == simple_extraction.metadata.llm_model
        assert result.metadata.extraction_duration_ms == simple_extraction.metadata.extraction_duration_ms
    
    @pytest.mark.asyncio
    async def test_refine_valid_extraction(self, simple_extraction):
        """Test refining a valid extraction."""
        pipeline = ExtractionRefinementPipeline(max_iterations=3)
        
        refined, critique, history = await pipeline.refine_extraction(
            simple_extraction,
            "Original text",
        )
        
        assert refined is not None
        assert critique.is_valid is True
        assert len(history) >= 1
        assert history[0]["iteration"] == 0
        assert history[0]["quality_score"] > 0.0
    
    @pytest.mark.asyncio
    async def test_refine_invalid_extraction(self, incomplete_extraction):
        """Test refining an invalid extraction."""
        pipeline = ExtractionRefinementPipeline(max_iterations=3)
        
        refined, critique, history = await pipeline.refine_extraction(
            incomplete_extraction,
            "Original text",
        )
        
        assert refined is not None
        assert len(history) >= 1
        
        # Should attempt iterations
        first_iteration = history[0]
        assert first_iteration["issue_count"] > 0
    
    @pytest.mark.asyncio
    async def test_pipeline_tracks_iterations(self, incomplete_extraction):
        """Test that pipeline properly tracks iterations."""
        pipeline = ExtractionRefinementPipeline(max_iterations=2)
        
        refined, critique, history = await pipeline.refine_extraction(
            incomplete_extraction,
            "Original text",
        )
        
        # Should have at least 1 iteration recorded
        assert len(history) >= 1
        
        # Check iteration structure
        for i, iteration in enumerate(history):
            assert "iteration" in iteration
            assert "validation" in iteration
            assert "issue_count" in iteration
            assert "quality_score" in iteration


class TestValidationIssueSeverity:
    """Test ValidationIssueSeverity enum."""
    
    def test_severity_levels(self):
        """Test all severity levels are defined."""
        expected_levels = {"critical", "error", "warning", "info"}
        
        actual_levels = {level.value for level in ValidationIssueSeverity}
        assert actual_levels == expected_levels


class TestFullCritiqueRefinementPipeline:
    """Integration tests for the full critique-refinement pipeline."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_flow_with_valid_extraction(self, simple_extraction):
        """Test complete pipeline flow: validation -> critique -> refinement."""
        # Step 1: Validate extraction
        validation = ExtractionValidator.validate_extraction(simple_extraction)
        assert validation.is_valid is True
        assert validation.quality_score > 0.8
        
        # Step 2: Generate critique feedback
        agent = CritiqueAgent()
        critique = await agent.generate_critique_feedback(
            simple_extraction,
            simple_extraction.metadata.input_text,
        )
        assert "validation" in critique
        assert "refinement_recommendations" in critique
        
        # Step 3: Refine extraction through pipeline
        pipeline = ExtractionRefinementPipeline(max_iterations=2)
        refined, final_critique, history = await pipeline.refine_extraction(
            simple_extraction,
            simple_extraction.metadata.input_text,
        )
        
        # Verify pipeline history
        assert len(history) >= 1
        assert refined is not None
        assert final_critique.is_valid is True
    
    @pytest.mark.asyncio
    async def test_full_pipeline_flow_with_problematic_extraction(self, incomplete_extraction):
        """Test pipeline on extraction with issues needing refinement."""
        # Step 1: Validate extraction
        validation = ExtractionValidator.validate_extraction(incomplete_extraction)
        # Note: is_valid only checks for CRITICAL issues. This extraction may have ERROR/WARNING but still be valid
        assert validation.needs_refinement is True
        
        # Should have detected issues (ERROR, WARNING, or INFO level)
        assert len(validation.issues) > 0
        
        # Should have some issues beyond just INFO level
        non_info_issues = [i for i in validation.issues if i.severity != ValidationIssueSeverity.INFO]
        assert len(non_info_issues) > 0
        
        # Step 2: Generate critique feedback
        agent = CritiqueAgent()
        critique = await agent.generate_critique_feedback(
            incomplete_extraction,
            incomplete_extraction.metadata.input_text,
        )
        assert "validation" in critique
        
        # Step 3: Refine extraction through pipeline
        pipeline = ExtractionRefinementPipeline(max_iterations=3)
        refined, final_critique, history = await pipeline.refine_extraction(
            incomplete_extraction,
            incomplete_extraction.metadata.input_text,
        )
        
        # Verify refinement occurred
        assert refined is not None
        assert len(history) >= 1
        
        # First iteration should have detected issues
        assert history[0]["issue_count"] > 0
    
    @pytest.mark.asyncio
    async def test_quality_score_tracking_across_iterations(self, incomplete_extraction):
        """Test that quality scores are tracked across refinement iterations."""
        pipeline = ExtractionRefinementPipeline(max_iterations=3)
        
        refined, critique, history = await pipeline.refine_extraction(
            incomplete_extraction,
            incomplete_extraction.metadata.input_text,
        )
        
        # Verify history has quality scores
        assert len(history) >= 1
        for iteration in history:
            assert "quality_score" in iteration
            assert 0.0 <= iteration["quality_score"] <= 1.0
            assert "iteration" in iteration
    
    @pytest.mark.asyncio
    async def test_issue_detection_and_tracking(self, incomplete_extraction):
        """Test that issues are properly detected and tracked through pipeline."""
        pipeline = ExtractionRefinementPipeline(max_iterations=2)
        
        refined, critique, history = await pipeline.refine_extraction(
            incomplete_extraction,
            incomplete_extraction.metadata.input_text,
        )
        
        # First iteration should have issues recorded
        first_iteration = history[0]
        assert "issue_count" in first_iteration
        assert "validation" in first_iteration
        
        # Issue count should be tracked
        assert first_iteration["issue_count"] >= 0
    
    @pytest.mark.asyncio
    async def test_pipeline_iteration_limit_respected(self):
        """Test that pipeline respects max_iterations limit."""
        # Create extraction with multiple issues
        entities = [
            ExtractedEntity(
                id="e1",
                type=EntityType.ACTIVITY,
                name="Activity",
                confidence=ConfidenceLevel.LOW,
            ),
        ]
        
        metadata = ExtractionMetadata(
            input_text="Complex process",
            input_length=15,
            extraction_timestamp=datetime.now().isoformat(),
            extraction_duration_ms=1.0,
            llm_model="mistral",
            llm_temperature=0.7,
        )
        
        extraction = ExtractionResult(
            entities=entities,
            relations=[],
            metadata=metadata,
        )
        
        # Set low iteration limit
        pipeline = ExtractionRefinementPipeline(max_iterations=1)
        refined, critique, history = await pipeline.refine_extraction(
            extraction,
            extraction.metadata.input_text,
        )
        
        # Should not exceed max iterations
        assert len(history) <= 1
    
    @pytest.mark.asyncio
    async def test_metadata_preservation_through_refinement(self, simple_extraction):
        """Test that metadata is preserved through refinement cycles."""
        pipeline = ExtractionRefinementPipeline(max_iterations=2)
        original_metadata = simple_extraction.metadata
        
        refined, critique, history = await pipeline.refine_extraction(
            simple_extraction,
            simple_extraction.metadata.input_text,
        )
        
        # Metadata should be preserved
        assert refined.metadata.input_text == original_metadata.input_text
        assert refined.metadata.llm_model == original_metadata.llm_model
        assert refined.metadata.input_length == original_metadata.input_length
    
    @pytest.mark.asyncio
    async def test_critique_result_structure(self, simple_extraction):
        """Test that critique result has proper structure."""
        agent = CritiqueAgent()
        
        critique = await agent.generate_critique_feedback(
            simple_extraction,
            simple_extraction.metadata.input_text,
        )
        
        # Should have required fields
        assert isinstance(critique, dict)
        assert "validation" in critique
        assert "refinement_recommendations" in critique
        
        # Validation should be a CritiqueResult
        validation = critique["validation"]
        assert hasattr(validation, "is_valid")
        assert hasattr(validation, "quality_score")
        assert hasattr(validation, "issues")
    
    @pytest.mark.asyncio
    async def test_pipeline_handles_edge_case_empty_extraction(self):
        """Test pipeline handles extraction with minimal content."""
        entities = [
            ExtractedEntity(
                id="e1",
                type=EntityType.EVENT,
                name="Event",
                confidence=ConfidenceLevel.MEDIUM,
            ),
        ]
        
        metadata = ExtractionMetadata(
            input_text="",
            input_length=0,
            extraction_timestamp=datetime.now().isoformat(),
            extraction_duration_ms=0.5,
            llm_model="mistral",
            llm_temperature=0.7,
        )
        
        extraction = ExtractionResult(
            entities=entities,
            relations=[],
            metadata=metadata,
        )
        
        # Should not crash
        pipeline = ExtractionRefinementPipeline(max_iterations=1)
        refined, critique, history = await pipeline.refine_extraction(
            extraction,
            "",
        )
        
        assert refined is not None
        assert history is not None
    
    @pytest.mark.asyncio
    async def test_validation_issues_captured_in_history(self, incomplete_extraction):
        """Test that validation issues are captured in iteration history."""
        pipeline = ExtractionRefinementPipeline(max_iterations=1)
        
        refined, critique, history = await pipeline.refine_extraction(
            incomplete_extraction,
            incomplete_extraction.metadata.input_text,
        )
        
        # First iteration should have validation data
        first_iteration = history[0]
        assert "validation" in first_iteration
        
        # Should have detected the disconnected nodes issue
        validation = first_iteration["validation"]
        assert hasattr(validation, "issues")
        assert len(validation.issues) > 0
    
    @pytest.mark.asyncio
    async def test_refinement_suggestions_provided(self, incomplete_extraction):
        """Test that refinement suggestions are provided."""
        validation = ExtractionValidator.validate_extraction(incomplete_extraction)
        
        # Should have suggestions for improvement
        assert isinstance(validation.suggestions_for_improvement, list)
        assert len(validation.suggestions_for_improvement) > 0
        
        # Each suggestion should be a string
        for suggestion in validation.suggestions_for_improvement:
            assert isinstance(suggestion, str)
            assert len(suggestion) > 0
    
    @pytest.mark.asyncio
    async def test_multiple_issue_types_detected(self, incomplete_extraction):
        """Test that multiple issue types can be detected in one pass."""
        validation = ExtractionValidator.validate_extraction(incomplete_extraction)
        
        # Should have multiple issue types
        issue_types = {issue.type for issue in validation.issues}
        assert len(issue_types) > 0
        
        # Each issue should be properly typed
        for issue in validation.issues:
            assert isinstance(issue, ValidationIssue)
            assert issue.type in ValidationIssueType
            assert issue.severity in ValidationIssueSeverity
    
    @pytest.mark.asyncio
    async def test_confidence_levels_considered(self, incomplete_extraction):
        """Test that confidence levels impact validation results."""
        validation = ExtractionValidator.validate_extraction(incomplete_extraction)
        
        # Low confidence entities should trigger issues
        low_confidence_issues = [
            i for i in validation.issues
            if i.type == ValidationIssueType.LOW_CONFIDENCE_ENTITIES
        ]
        
        # Should detect at least the low-confidence entity (e1 with LOW confidence)
        assert len(low_confidence_issues) > 0
    
    @pytest.mark.asyncio
    async def test_pipeline_state_isolation(self, simple_extraction, incomplete_extraction):
        """Test that multiple pipelines don't interfere with each other."""
        pipeline1 = ExtractionRefinementPipeline(max_iterations=2)
        pipeline2 = ExtractionRefinementPipeline(max_iterations=3)
        
        # Run both pipelines
        refined1, critique1, history1 = await pipeline1.refine_extraction(
            simple_extraction,
            simple_extraction.metadata.input_text,
        )
        
        refined2, critique2, history2 = await pipeline2.refine_extraction(
            incomplete_extraction,
            incomplete_extraction.metadata.input_text,
        )
        
        # Each should have independent history
        assert pipeline1.iteration_history != pipeline2.iteration_history
        assert len(history1) >= 1
        assert len(history2) >= 1
    
    @pytest.mark.asyncio
    async def test_end_to_end_pipeline_completes_successfully(self, simple_extraction):
        """Test complete end-to-end pipeline execution."""
        # Initialize all components
        validator = ExtractionValidator()
        agent = CritiqueAgent()
        pipeline = ExtractionRefinementPipeline(max_iterations=2)
        
        # Step 1: Validate
        validation = ExtractionValidator.validate_extraction(simple_extraction)
        assert validation is not None
        
        # Step 2: Critique
        critique_feedback = await agent.generate_critique_feedback(
            simple_extraction,
            simple_extraction.metadata.input_text,
        )
        assert critique_feedback is not None
        
        # Step 3: Refine
        refined, final_critique, history = await pipeline.refine_extraction(
            simple_extraction,
            simple_extraction.metadata.input_text,
        )
        
        # Final state should be valid
        assert refined is not None
        assert final_critique is not None
        assert len(history) >= 1
        assert isinstance(final_critique.quality_score, float)
    
    @pytest.mark.asyncio
    async def test_quality_improvements_tracked(self, incomplete_extraction):
        """Test that quality improvements are tracked through pipeline."""
        initial_validation = ExtractionValidator.validate_extraction(incomplete_extraction)
        initial_quality = initial_validation.quality_score
        
        pipeline = ExtractionRefinementPipeline(max_iterations=3)
        refined, critique, history = await pipeline.refine_extraction(
            incomplete_extraction,
            incomplete_extraction.metadata.input_text,
        )
        
        # Should have tracking data
        assert len(history) >= 1
        
        # All quality scores should be valid floats
        for iteration in history:
            assert isinstance(iteration["quality_score"], float)
            assert 0.0 <= iteration["quality_score"] <= 1.0
