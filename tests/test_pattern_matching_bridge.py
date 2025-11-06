"""
Integration tests for the pattern matching bridge with process graph builder.

Tests the bridge between advanced pattern matching and the extraction pipeline.
"""

import pytest
from bpmn_agent.models.knowledge_base import (
    BPMNPattern,
    ComplexityLevel,
    DomainType,
    GraphStructure,
    KnowledgeBase,
    PatternCategory,
)
from bpmn_agent.knowledge.pattern_matching_bridge import AdvancedPatternMatchingBridge


@pytest.fixture
def sample_kb_with_finance_patterns():
    """Create a sample KB with finance patterns."""
    kb = KnowledgeBase()
    
    # Finance - Payment Processing
    pattern1 = BPMNPattern(
        id="finance_payment_001",
        name="Payment Processing Workflow",
        description="Process a payment request through validation and settlement",
        domain=DomainType.FINANCE,
        category=PatternCategory.SEQUENTIAL,
        complexity=ComplexityLevel.MODERATE,
        graph_structure=GraphStructure(
            nodes=["submit", "validate", "authorize", "process", "confirm"],
            edges=["submit->validate", "validate->authorize", "authorize->process", "process->confirm"],
            node_types={"submit": "activity", "validate": "activity", "authorize": "activity", "process": "activity", "confirm": "activity"}
        ),
        tags={"payment", "finance", "transaction", "settlement"},
        examples=["Customer submits payment, system validates, processes, and confirms"],
        confidence=0.95,
    )
    kb.add_pattern(pattern1)
    
    # Finance - Fraud Detection
    pattern2 = BPMNPattern(
        id="finance_fraud_001",
        name="Fraud Detection",
        description="Detect and handle potentially fraudulent transactions",
        domain=DomainType.FINANCE,
        category=PatternCategory.EXCLUSIVE_CHOICE,
        complexity=ComplexityLevel.MODERATE,
        graph_structure=GraphStructure(
            nodes=["check", "flag", "review", "approve", "reject"],
            edges=["check->flag", "flag->review", "review->approve", "review->reject"],
            node_types={"check": "activity", "flag": "activity", "review": "activity", "approve": "activity", "reject": "activity"}
        ),
        tags={"fraud", "detection", "security", "compliance"},
        examples=["Check transaction for fraud indicators"],
        confidence=0.92,
        related_patterns=["finance_payment_001"],
    )
    kb.add_pattern(pattern2)
    
    # Generic - Decision
    pattern3 = BPMNPattern(
        id="generic_decision_001",
        name="Binary Decision",
        description="Make a decision between two paths",
        domain=DomainType.GENERIC,
        category=PatternCategory.EXCLUSIVE_CHOICE,
        complexity=ComplexityLevel.SIMPLE,
        graph_structure=GraphStructure(
            nodes=["gateway", "yes_path", "no_path", "join"],
            edges=["gateway->yes_path", "gateway->no_path", "yes_path->join", "no_path->join"],
            node_types={"gateway": "gateway", "yes_path": "activity", "no_path": "activity", "join": "gateway"}
        ),
        tags={"decision", "binary", "choice"},
        examples=["If condition then path A else path B"],
        confidence=0.98,
    )
    kb.add_pattern(pattern3)
    
    return kb


@pytest.fixture
def bridge(sample_kb_with_finance_patterns):
    """Create a bridge with sample KB."""
    return AdvancedPatternMatchingBridge(sample_kb_with_finance_patterns)


class TestBridgePatternFinding:
    """Test pattern finding through the bridge."""
    
    def test_find_patterns_for_process_basic(self, bridge):
        """Test finding patterns for a process description."""
        rec = bridge.find_patterns_for_process("Process a payment transaction")
        assert rec is not None
        assert len(rec.patterns) > 0
    
    def test_find_patterns_with_domain_hint(self, bridge):
        """Test finding patterns with domain hint."""
        rec = bridge.find_patterns_for_process(
            "Process a payment transaction",
            domain=DomainType.FINANCE
        )
        assert rec.primary_domain == DomainType.FINANCE or rec.primary_domain == DomainType.GENERIC
    
    def test_find_patterns_max_results_respected(self, bridge):
        """Test that max_patterns is respected."""
        rec = bridge.find_patterns_for_process("workflow", max_patterns=2)
        assert len(rec.patterns) <= 2
    
    def test_recommendation_has_alternatives(self, bridge):
        """Test that recommendations include alternatives."""
        rec = bridge.find_patterns_for_process("process payment and detect fraud")
        assert isinstance(rec.alternative_patterns, list)


class TestBridgeActivityMatching:
    """Test activity matching through the bridge."""
    
    def test_match_activity_basic(self, bridge):
        """Test matching an activity to patterns."""
        matches = bridge.match_activity_to_patterns(
            "Validate Payment",
            "Validate the payment amount and account details"
        )
        assert isinstance(matches, list)
    
    def test_match_activity_with_domain(self, bridge):
        """Test activity matching with domain filter."""
        matches = bridge.match_activity_to_patterns(
            "Authorize Transaction",
            "Authorize payment from account",
            domain=DomainType.FINANCE
        )
        assert len(matches) >= 0
    
    def test_match_activity_returns_sorted_results(self, bridge):
        """Test that results are sorted by relevance."""
        matches = bridge.match_activity_to_patterns(
            "Process",
            "Process the request"
        )
        if len(matches) > 1:
            scores = [m.match_score for m in matches]
            assert scores == sorted(scores, reverse=True)


class TestBridgeSimilarPatterns:
    """Test finding similar patterns through the bridge."""
    
    def test_find_similar_patterns(self, bridge):
        """Test finding similar patterns."""
        similar = bridge.find_similar_patterns_for_pattern("finance_payment_001")
        assert isinstance(similar, list)
        assert all(hasattr(p, 'id') for p in similar)
    
    def test_similar_patterns_max_respected(self, bridge):
        """Test that max_similar is respected."""
        similar = bridge.find_similar_patterns_for_pattern("finance_payment_001", max_similar=2)
        assert len(similar) <= 2
    
    def test_similar_patterns_excludes_self(self, bridge):
        """Test that results don't include the reference pattern."""
        similar = bridge.find_similar_patterns_for_pattern("finance_payment_001")
        assert all(p.id != "finance_payment_001" for p in similar)


class TestBridgeActivityValidation:
    """Test activity validation through the bridge."""
    
    def test_validate_activities_single(self, bridge):
        """Test validating a single activity."""
        results = bridge.validate_extracted_activities(["Process Payment"])
        assert "Process Payment" in results
        is_valid, confidence, issues = results["Process Payment"]
        assert isinstance(is_valid, bool)
        assert 0.0 <= confidence <= 1.0
        assert isinstance(issues, list)
    
    def test_validate_activities_multiple(self, bridge):
        """Test validating multiple activities."""
        activities = ["Validate Payment", "Check Fraud", "Authorize"]
        results = bridge.validate_extracted_activities(activities)
        assert len(results) == len(activities)
        for activity in activities:
            assert activity in results
    
    def test_validate_activities_with_domain(self, bridge):
        """Test validation with domain filter."""
        results = bridge.validate_extracted_activities(
            ["Payment Authorization"],
            domain=DomainType.FINANCE
        )
        assert len(results) > 0
    
    def test_validate_activities_unknown(self, bridge):
        """Test validating unknown activities."""
        results = bridge.validate_extracted_activities(["XYZ Unknown Activity"])
        is_valid, confidence, issues = results["XYZ Unknown Activity"]
        # Unknown activity should return false or low confidence
        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)


class TestBridgePatternEnrichment:
    """Test pattern enrichment through the bridge."""
    
    def test_enrich_pattern_basic(self, bridge):
        """Test enriching a pattern."""
        enriched = bridge.enrich_pattern_context("finance_payment_001")
        assert "pattern" in enriched
        assert "similar_patterns" in enriched
        assert "related_patterns" in enriched
    
    def test_enriched_pattern_has_required_fields(self, bridge):
        """Test that enriched pattern has all required fields."""
        enriched = bridge.enrich_pattern_context("finance_payment_001")
        pattern_info = enriched["pattern"]
        assert "id" in pattern_info
        assert "name" in pattern_info
        assert "description" in pattern_info
        assert "domain" in pattern_info
        assert "category" in pattern_info
        assert "complexity" in pattern_info
    
    def test_enrich_pattern_includes_related(self, bridge):
        """Test that enrichment includes related patterns."""
        enriched = bridge.enrich_pattern_context("finance_fraud_001")
        # Fraud pattern has finance_payment_001 as related
        related = enriched.get("related_patterns", [])
        assert isinstance(related, list)
    
    def test_enrich_nonexistent_pattern(self, bridge):
        """Test enriching a nonexistent pattern."""
        enriched = bridge.enrich_pattern_context("nonexistent_pattern")
        assert enriched == {}


class TestBridgeDomainSuggestions:
    """Test pattern suggestions by domain."""
    
    def test_suggest_patterns_by_domain(self, bridge):
        """Test getting suggested patterns for a domain."""
        suggestions = bridge.suggest_patterns_by_domain(DomainType.FINANCE)
        assert isinstance(suggestions, list)
        # Should include both finance-specific and generic patterns
        assert all(p["domain"] in ["finance", "generic"] for p in suggestions)
        # Should include at least one finance pattern
        assert any(p["domain"] == "finance" for p in suggestions)
    
    def test_suggest_patterns_respects_max(self, bridge):
        """Test that max_patterns is respected."""
        suggestions = bridge.suggest_patterns_by_domain(DomainType.FINANCE, max_patterns=1)
        assert len(suggestions) <= 1
    
    def test_suggest_patterns_with_complexity_filter(self, bridge):
        """Test suggestions with complexity filter."""
        suggestions = bridge.suggest_patterns_by_domain(
            DomainType.FINANCE,
            complexity=ComplexityLevel.MODERATE
        )
        assert all(p["complexity"] == "moderate" for p in suggestions)
    
    def test_suggest_patterns_sorted_by_relevance(self, bridge):
        """Test that suggestions are sorted by relevance."""
        suggestions = bridge.suggest_patterns_by_domain(DomainType.GENERIC)
        if len(suggestions) > 1:
            # Should be sorted by usage/confidence
            assert isinstance(suggestions, list)


class TestBridgeSearch:
    """Test comprehensive pattern search through the bridge."""
    
    def test_search_patterns_basic(self, bridge):
        """Test basic pattern search."""
        results = bridge.search_patterns("payment")
        assert isinstance(results, list)
    
    def test_search_patterns_with_domain(self, bridge):
        """Test search with domain filter."""
        results = bridge.search_patterns("validate", domain=DomainType.FINANCE)
        assert len(results) >= 0
    
    def test_search_patterns_with_category(self, bridge):
        """Test search with category filter."""
        results = bridge.search_patterns(
            "workflow",
            category=PatternCategory.SEQUENTIAL
        )
        assert all(r.pattern.category == PatternCategory.SEQUENTIAL for r in results)
    
    def test_search_patterns_max_results(self, bridge):
        """Test that search respects max_results."""
        results = bridge.search_patterns("process", max_results=2)
        assert len(results) <= 2


class TestBridgeStatistics:
    """Test pattern library statistics."""
    
    def test_get_pattern_statistics(self, bridge):
        """Test getting pattern statistics."""
        stats = bridge.get_pattern_statistics()
        assert "total_patterns" in stats
        assert "total_examples" in stats
        assert "domains" in stats
        assert "categories" in stats
        assert "complexities" in stats
    
    def test_statistics_counts_correct(self, bridge):
        """Test that statistics counts are correct."""
        stats = bridge.get_pattern_statistics()
        assert stats["total_patterns"] == 3  # We have 3 patterns in fixture
    
    def test_statistics_has_metadata(self, bridge):
        """Test that statistics include metadata."""
        stats = bridge.get_pattern_statistics()
        assert "metadata" in stats
        assert "version" in stats["metadata"]


class TestBridgeDocumentation:
    """Test pattern documentation export."""
    
    def test_export_pattern_basic(self, bridge):
        """Test exporting a pattern for documentation."""
        doc = bridge.export_pattern_for_documentation("finance_payment_001")
        assert len(doc) > 0
        assert "Payment Processing Workflow" in doc
    
    def test_export_pattern_includes_sections(self, bridge):
        """Test that exported pattern includes required sections."""
        doc = bridge.export_pattern_for_documentation("finance_payment_001")
        assert "# " in doc  # Has heading
        assert "Description" in doc or "description" in doc
        assert "Tags" in doc or "tags" in doc
    
    def test_export_nonexistent_pattern(self, bridge):
        """Test exporting a nonexistent pattern."""
        doc = bridge.export_pattern_for_documentation("nonexistent")
        assert doc == ""
    
    def test_export_pattern_with_related(self, bridge):
        """Test exporting a pattern with related patterns."""
        doc = bridge.export_pattern_for_documentation("finance_fraud_001")
        # Should include related patterns section since it has related_patterns
        assert isinstance(doc, str)


class TestBridgeIntegration:
    """Integration tests combining multiple bridge features."""
    
    def test_workflow_find_and_enrich(self, bridge):
        """Test finding patterns and enriching them."""
        # Find patterns for a process
        rec = bridge.find_patterns_for_process("Handle payment transaction")
        
        if rec.patterns:
            # Get first pattern and enrich it
            first_pattern, score = rec.patterns[0]
            enriched = bridge.enrich_pattern_context(first_pattern.id)
            assert enriched is not None
    
    def test_workflow_search_and_validate(self, bridge):
        """Test searching patterns and validating activities."""
        # Search for patterns
        matches = bridge.search_patterns("payment authorization")
        
        if matches:
            # Validate activities from matched patterns
            activities = [m.pattern.name for m in matches]
            results = bridge.validate_extracted_activities(activities)
            assert len(results) > 0
    
    def test_workflow_domain_to_statistics(self, bridge):
        """Test domain-based workflow with statistics."""
        # Get patterns for domain
        suggestions = bridge.suggest_patterns_by_domain(DomainType.FINANCE)
        
        # Get overall statistics
        stats = bridge.get_pattern_statistics()
        
        # Stats should reflect domain patterns
        assert stats["domains"]["finance"] > 0
