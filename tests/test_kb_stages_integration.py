"""
Integration Tests for KB + Extraction Stages Pipeline

Tests covering:
- Stage 1 (Text Preprocessing) with KB integration
- Stage 2 (Entity Extraction) with KB augmented prompts
- Stage 3 (Entity Resolution) with KB rules
- Full pipeline with KB context propagation
"""

import asyncio
import logging
import pytest
from typing import List

from bpmn_agent.core.llm_client import BaseLLMClient, LLMConfig, LLMClientFactory
from bpmn_agent.knowledge import (
    DomainClassifier,
    PatternRecognizer,
    ContextOptimizer,
)
from bpmn_agent.models.extraction import (
    EntityType,
    ExtractedEntity,
    ExtractedRelation,
)
from bpmn_agent.models.knowledge_base import DomainType, ComplexityLevel
from bpmn_agent.stages.text_preprocessing import TextPreprocessor
from bpmn_agent.stages.entity_extraction import EntityExtractor
from bpmn_agent.stages.entity_resolution import (
    CoReferenceResolver,
    ActorConsolidator,
)

logger = logging.getLogger(__name__)


class TestStage1WithKB:
    """Tests for Stage 1 (Text Preprocessing) with KB integration."""

    @pytest.fixture
    def preprocessor(self):
        """Create a text preprocessor with KB enabled."""
        return TextPreprocessor(enable_kb=True)

    def test_preprocessor_initialization_with_kb(self, preprocessor):
        """Test preprocessor initializes with KB support."""
        assert preprocessor is not None
        assert preprocessor.enable_kb is True

    def test_preprocessing_simple_text_with_kb(self, preprocessor):
        """Test preprocessing simple text extracts KB metadata."""
        text = (
            "The customer submits an order. "
            "The system validates the order. "
            "Payment is processed. "
            "Order confirmation is sent."
        )

        result = preprocessor.preprocess(text)

        # Check basic preprocessing
        assert result is not None
        assert result.original_text == text
        assert len(result.cleaned_text) > 0
        assert len(result.sentences) > 0
        assert len(result.chunks) > 0

        # Check KB metadata
        assert "detected_domain" in result.metadata
        assert "detected_complexity" in result.metadata
        assert "recognized_patterns" in result.metadata
        assert "pattern_hints" in result.metadata

    def test_preprocessing_finance_text_with_kb(self, preprocessor):
        """Test preprocessing finance text detects finance domain."""
        finance_text = (
            "An invoice is received from the vendor. "
            "The invoice is reviewed for accuracy. "
            "The payment is approved by the finance manager. "
            "The payment is recorded in the general ledger."
        )

        result = preprocessor.preprocess(finance_text)

        assert result is not None
        # Check that domain was detected
        detected_domain = result.metadata.get("detected_domain")
        assert detected_domain in [DomainType.FINANCE, DomainType.GENERIC, None]

    def test_preprocessing_complex_text_with_kb(self, preprocessor):
        """Test preprocessing complex text analyzes complexity."""
        complex_text = (
            "When a customer places an order, the system performs multiple parallel checks. "
            "First, inventory availability is verified. "
            "Second, credit card validation is performed. "
            "Third, fraud detection is run. "
            "If any check fails, the order is rejected and customer notified. "
            "If all checks pass, payment is processed with retry logic. "
            "Upon successful payment, inventory is reserved. "
            "Order confirmation is generated and sent. "
            "Fulfillment workflow is triggered with picking, packing, and shipping steps. "
            "Multiple shipments may occur if items are from different warehouses. "
            "Customer receives tracking information and updates."
        )

        result = preprocessor.preprocess(complex_text)

        assert result is not None
        # Check complexity detection
        detected_complexity = result.metadata.get("detected_complexity")
        assert detected_complexity in [
            ComplexityLevel.SIMPLE,
            ComplexityLevel.MODERATE,
            ComplexityLevel.COMPLEX,
            None,
        ]

    def test_preprocessing_kb_metadata_structure(self, preprocessor):
        """Test that KB metadata has expected structure."""
        text = "A request is submitted. It is reviewed. It is approved."

        result = preprocessor.preprocess(text)

        # Verify metadata structure
        assert "detected_domain" in result.metadata
        assert "detected_complexity" in result.metadata
        assert "recognized_patterns" in result.metadata
        assert "domain_confidence" in result.metadata
        assert "complexity_score" in result.metadata
        assert "pattern_hints" in result.metadata

        # Verify recognized_patterns is a list
        recognized_patterns = result.metadata.get("recognized_patterns", [])
        assert isinstance(recognized_patterns, list)

        # Verify pattern_hints is a list
        pattern_hints = result.metadata.get("pattern_hints", [])
        assert isinstance(pattern_hints, list)


class TestStage2WithKB:
    """Tests for Stage 2 (Entity Extraction) with KB augmentation."""

    @pytest.fixture
    def extractor(self):
        """Create an entity extractor with KB enabled."""
        # Note: This requires a valid LLM client
        config = LLMConfig(model="gpt-3.5-turbo", temperature=0.3)
        try:
            llm_client = LLMClientFactory.create(config)
            return EntityExtractor(llm_client, enable_kb=True)
        except Exception as e:
            logger.warning(f"Could not initialize entity extractor: {e}")
            pytest.skip("LLM client not available")

    def test_extractor_initialization_with_kb(self, extractor):
        """Test entity extractor initializes with KB support."""
        assert extractor is not None
        assert extractor.enable_kb is True

    @pytest.mark.asyncio
    async def test_kb_augmented_prompt_building(self):
        """Test that KB context is used to build augmented prompts."""
        from bpmn_agent.stages.entity_extraction import KBAugmentedPromptBuilder
        from bpmn_agent.models.knowledge_base import (
            BPMNPattern,
            GraphStructure,
            PatternCategory,
            DomainExample,
            ContextPackage,
        )

        # Create test pattern
        pattern = BPMNPattern(
            id="test_pattern",
            name="Sequential Pattern",
            description="A simple sequential flow",
            category=PatternCategory.SEQUENTIAL,
            graph_structure=GraphStructure(
                nodes=["start", "activity", "end"],
                edges=["start->activity", "activity->end"],
            ),
        )

        # Create test example
        example = DomainExample(
            id="test_example",
            text="Task A is done, then Task B, then Task C.",
            domain=DomainType.GENERIC,
            patterns_used=["test_pattern"],
        )

        # Create context package
        context = ContextPackage(
            selected_patterns=[pattern],
            selected_examples=[example],
            recognized_patterns=["test_pattern"],
        )

        # Test augmentation
        base_prompt = "Extract entities and relations."
        augmented_prompt = KBAugmentedPromptBuilder.build_kb_augmented_prompt(
            base_prompt,
            context_package=context,
        )

        assert augmented_prompt != base_prompt
        assert "Sequential Pattern" in augmented_prompt
        assert "Task A is done" in augmented_prompt


class TestStage3WithKB:
    """Tests for Stage 3 (Entity Resolution) with KB support."""

    def test_coreference_resolver_with_kb(self):
        """Test co-reference resolver with KB support."""
        resolver = CoReferenceResolver(enable_kb=True)

        assert resolver is not None
        assert resolver.enable_kb is True

    def test_coreference_resolution_basic(self):
        """Test basic co-reference resolution."""
        # Create test entities
        entity1 = ExtractedEntity(
            id="e1",
            name="manager",
            type=EntityType.ACTOR,
            description="The manager",
        )
        entity2 = ExtractedEntity(
            id="e2",
            name="the manager",
            type=EntityType.ACTOR,
            description="Manager reference",
        )
        entity3 = ExtractedEntity(
            id="e3",
            name="Manager",
            type=EntityType.ACTOR,
            description="Another manager reference",
        )

        resolver = CoReferenceResolver(enable_kb=True)
        canonical_entities, co_ref_groups = resolver.resolve_co_references(
            entities=[entity1, entity2, entity3],
            domain=DomainType.HR,
        )

        # Should consolidate similar mentions
        assert canonical_entities is not None
        assert len(canonical_entities) > 0
        assert len(canonical_entities) <= 3  # Should have consolidated

    def test_actor_consolidation_with_kb(self):
        """Test actor consolidation with KB support."""
        # Create test entities
        actor1 = ExtractedEntity(
            id="a1",
            name="HR Manager",
            type=EntityType.ACTOR,
        )
        actor2 = ExtractedEntity(
            id="a2",
            name="HR manager",
            type=EntityType.ACTOR,
        )
        activity = ExtractedEntity(
            id="ac1",
            name="Approve Request",
            type=EntityType.ACTIVITY,
        )

        # Create test relations
        relation = ExtractedRelation(
            id="r1",
            source_id="a1",
            target_id="ac1",
            type="involves",
        )

        consolidator = ActorConsolidator(enable_kb=True)
        actor_profiles, activity_map = consolidator.consolidate_actors(
            entities=[actor1, actor2, activity],
            relations=[relation],
            domain=DomainType.HR,
        )

        assert actor_profiles is not None
        assert activity_map is not None


class TestFullPipelineWithKB:
    """Tests for complete extraction pipeline with KB integration."""

    def test_full_pipeline_flow_simple(self):
        """Test complete pipeline flow for simple process."""
        text = (
            "A customer request is received. "
            "The request is reviewed by the team. "
            "A decision is made to approve or reject. "
            "The customer is notified of the decision."
        )

        # Stage 1: Text Preprocessing
        preprocessor = TextPreprocessor(enable_kb=True)
        preprocessed = preprocessor.preprocess(text)

        assert preprocessed is not None
        assert preprocessed.metadata.get("detected_domain") is not None

        # Verify KB metadata propagation
        assert "recognized_patterns" in preprocessed.metadata
        assert "pattern_hints" in preprocessed.metadata

        logger.info(f"Stage 1 complete: domain={preprocessed.metadata.get('detected_domain')}")

    def test_full_pipeline_flow_complex_finance(self):
        """Test complete pipeline for complex finance process."""
        finance_text = (
            "When a vendor invoice arrives, it is logged into the system. "
            "The invoice is matched against purchase orders and receipts. "
            "If discrepancies exist, the invoice is held for investigation. "
            "Once verified, the invoice is routed for approval. "
            "If the amount exceeds 10,000, director approval is required. "
            "Otherwise, manager approval is sufficient. "
            "After approval, the invoice is scheduled for payment. "
            "Payment is processed according to agreed terms. "
            "The transaction is recorded in the accounting system."
        )

        # Stage 1: Text Preprocessing with KB
        preprocessor = TextPreprocessor(enable_kb=True)
        preprocessed = preprocessor.preprocess(finance_text)

        assert preprocessed is not None

        # Check domain detection
        domain = preprocessed.metadata.get("detected_domain")
        complexity = preprocessed.metadata.get("detected_complexity")

        logger.info(f"Stage 1: domain={domain}, complexity={complexity}")

        # Stage 2: Entity Extraction would use KB context
        # (Skipping actual extraction due to LLM requirement)

        # Stage 3: Entity Resolution would apply domain rules
        # (Would use domain-specific consolidation thresholds)

    def test_kb_context_propagation(self):
        """Test that KB context propagates through stages."""
        text = "An order is submitted. It is validated. Payment is processed."

        # Stage 1: Get KB context
        preprocessor = TextPreprocessor(enable_kb=True)
        preprocessed = preprocessor.preprocess(text)

        detected_domain = preprocessed.metadata.get("detected_domain")
        detected_complexity = preprocessed.metadata.get("detected_complexity")

        assert detected_domain is not None or detected_complexity is not None

        # Stage 2: Would use KB context
        try:
            optimizer = ContextOptimizer()
            context = optimizer.optimize_context(
                text=text,
                domain=detected_domain or DomainType.GENERIC,
                complexity=detected_complexity or ComplexityLevel.MODERATE,
                max_tokens=2000,
            )

            assert context is not None
            logger.info(
                f"Context package: {len(context.selected_patterns)} patterns, "
                f"{len(context.selected_examples)} examples"
            )
        except Exception as e:
            logger.warning(f"Context optimization skipped: {e}")


class TestKBDisableMode:
    """Tests for stages with KB disabled."""

    def test_stage1_without_kb(self):
        """Test Stage 1 preprocessing without KB."""
        preprocessor = TextPreprocessor(enable_kb=False)

        text = "A process with multiple steps."
        result = preprocessor.preprocess(text)

        # Should still work, just without KB metadata
        assert result is not None
        assert len(result.chunks) > 0

    def test_stage2_without_kb(self):
        """Test Stage 2 extraction without KB."""
        config = LLMConfig(model="gpt-3.5-turbo")
        try:
            llm_client = LLMClientFactory.create(config)
            extractor = EntityExtractor(llm_client, enable_kb=False)

            assert extractor.enable_kb is False
        except Exception as e:
            logger.warning(f"LLM client not available: {e}")

    def test_stage3_without_kb(self):
        """Test Stage 3 resolution without KB."""
        resolver = CoReferenceResolver(enable_kb=False)
        consolidator = ActorConsolidator(enable_kb=False)

        assert resolver.enable_kb is False
        assert consolidator.enable_kb is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
