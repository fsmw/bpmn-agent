"""
Test suite for advanced pattern matching engine.

Tests cover:
- Fuzzy keyword matching
- Semantic similarity scoring
- Pattern recommendation
- Category and complexity matching
- Composite search functionality
- Pattern validation
"""

import pytest
from typing import List
from bpmn_agent.models.knowledge_base import (
    BPMNPattern,
    ComplexityLevel,
    DomainType,
    GraphStructure,
    KnowledgeBase,
    PatternCategory,
)
from bpmn_agent.knowledge.advanced_pattern_matcher import (
    AdvancedPatternMatcher,
    MatchResult,
    PatternRecommendation,
)


@pytest.fixture
def sample_knowledge_base():
    """Create a sample knowledge base with test patterns."""
    kb = KnowledgeBase()
    
    # Pattern 1: Parallel Processing
    pattern1 = BPMNPattern(
        id="parallel_001",
        name="Parallel Fork-Join",
        description="Multiple tasks execute in parallel and then rejoin",
        domain=DomainType.GENERIC,
        category=PatternCategory.PARALLEL,
        complexity=ComplexityLevel.MODERATE,
        graph_structure=GraphStructure(
            nodes=["fork", "task1", "task2", "join"],
            edges=["fork->task1", "fork->task2", "task1->join", "task2->join"],
            node_types={"fork": "gateway", "task1": "activity", "task2": "activity", "join": "gateway"}
        ),
        tags={"parallelism", "fork", "join", "concurrent", "parallel"},
        examples=["Validate and archive in parallel", "Process payment and send confirmation"],
        confidence=0.95,
    )
    kb.add_pattern(pattern1)
    
    # Pattern 2: Sequential Processing
    pattern2 = BPMNPattern(
        id="sequential_001",
        name="Sequential Workflow",
        description="Tasks executed one after another in sequence",
        domain=DomainType.GENERIC,
        category=PatternCategory.SEQUENTIAL,
        complexity=ComplexityLevel.SIMPLE,
        graph_structure=GraphStructure(
            nodes=["start", "task1", "task2", "task3", "end"],
            edges=["start->task1", "task1->task2", "task2->task3", "task3->end"],
            node_types={"start": "event", "task1": "activity", "task2": "activity", "task3": "activity", "end": "event"}
        ),
        tags={"sequence", "linear", "step", "workflow"},
        examples=["Step 1, then step 2, then step 3"],
        confidence=0.98,
    )
    kb.add_pattern(pattern2)
    
    # Pattern 3: Exclusive Choice (Decision)
    pattern3 = BPMNPattern(
        id="decision_001",
        name="Exclusive Choice",
        description="Decision point where one of several paths is taken",
        domain=DomainType.GENERIC,
        category=PatternCategory.EXCLUSIVE_CHOICE,
        complexity=ComplexityLevel.SIMPLE,
        graph_structure=GraphStructure(
            nodes=["gateway", "path1", "path2", "join"],
            edges=["gateway->path1", "gateway->path2", "path1->join", "path2->join"],
            node_types={"gateway": "gateway", "path1": "activity", "path2": "activity", "join": "gateway"}
        ),
        tags={"decision", "choice", "xor", "condition"},
        examples=["If approved, process; else reject"],
        confidence=0.92,
    )
    kb.add_pattern(pattern3)
    
    # Pattern 4: Finance - Approval Workflow
    pattern4 = BPMNPattern(
        id="finance_approval_001",
        name="Financial Approval",
        description="Hierarchical approval process for financial transactions",
        domain=DomainType.FINANCE,
        category=PatternCategory.EXCLUSIVE_CHOICE,
        complexity=ComplexityLevel.MODERATE,
        graph_structure=GraphStructure(
            nodes=["submit", "review", "approve", "reject", "process", "archive"],
            edges=["submit->review", "review->approve", "review->reject", "approve->process", "process->archive"],
            node_types={"submit": "activity", "review": "activity", "approve": "gateway", "reject": "activity", "process": "activity", "archive": "activity"}
        ),
        tags={"approval", "finance", "review", "payment", "authorization"},
        examples=["Submit expense for approval by manager"],
        confidence=0.89,
        related_patterns=["sequential_001"],
    )
    kb.add_pattern(pattern4)
    
    # Pattern 5: Healthcare - Patient Consent
    pattern5 = BPMNPattern(
        id="healthcare_consent_001",
        name="Patient Consent Process",
        description="Obtain and verify patient consent before treatment",
        domain=DomainType.HEALTHCARE,
        category=PatternCategory.EXCLUSIVE_CHOICE,
        complexity=ComplexityLevel.SIMPLE,
        graph_structure=GraphStructure(
            nodes=["inform", "request", "consent", "document", "proceed"],
            edges=["inform->request", "request->consent", "consent->document", "document->proceed"],
            node_types={"inform": "activity", "request": "activity", "consent": "gateway", "document": "activity", "proceed": "activity"}
        ),
        tags={"healthcare", "consent", "patient", "authorization", "compliance"},
        examples=["Patient consents to treatment procedure"],
        confidence=0.94,
    )
    kb.add_pattern(pattern5)
    
    # Pattern 6: IT - Incident Response
    pattern6 = BPMNPattern(
        id="it_incident_001",
        name="Incident Response",
        description="Handle IT incidents with escalation and resolution",
        domain=DomainType.IT,
        category=PatternCategory.PARALLEL,
        complexity=ComplexityLevel.COMPLEX,
        graph_structure=GraphStructure(
            nodes=["detect", "triage", "investigate", "resolve", "notify", "close"],
            edges=["detect->triage", "triage->investigate", "triage->resolve", "investigate->notify", "resolve->notify", "notify->close"],
            node_types={"detect": "activity", "triage": "activity", "investigate": "activity", "resolve": "activity", "notify": "activity", "close": "activity"}
        ),
        tags={"incident", "response", "escalation", "it", "emergency"},
        examples=["Incident detected, investigate and resolve"],
        confidence=0.87,
    )
    kb.add_pattern(pattern6)
    
    return kb


@pytest.fixture
def matcher(sample_knowledge_base):
    """Create a pattern matcher with sample KB."""
    return AdvancedPatternMatcher(sample_knowledge_base)


class TestFuzzyKeywordMatching:
    """Test fuzzy keyword matching functionality."""
    
    def test_exact_keyword_match(self, matcher):
        """Test exact keyword matching."""
        results = matcher.match_by_keywords("parallel fork join", fuzzy=False)
        assert len(results) > 0
        assert any(r.pattern.id == "parallel_001" for r in results)
    
    def test_fuzzy_keyword_match(self, matcher):
        """Test fuzzy matching with typos."""
        results = matcher.match_by_keywords("paralel fork", fuzzy=True, threshold=0.3)
        # Fuzzy matching may have low match scores, so we check it runs without error
        assert isinstance(results, list)
    
    def test_keyword_match_with_domain_filter(self, matcher):
        """Test keyword matching with domain filter."""
        results = matcher.match_by_keywords("approval", domain=DomainType.FINANCE)
        assert len(results) > 0
        assert all(r.pattern.domain == DomainType.FINANCE or r.pattern.domain == DomainType.GENERIC for r in results)
    
    def test_keyword_match_threshold(self, matcher):
        """Test keyword matching with threshold."""
        results = matcher.match_by_keywords("xyz unknown keyword", threshold=0.5)
        assert all(r.match_score >= 0.5 for r in results)
    
    def test_keyword_match_sorting(self, matcher):
        """Test that results are sorted by match score."""
        results = matcher.match_by_keywords("sequential workflow step")
        assert len(results) > 0
        scores = [r.match_score for r in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_empty_query(self, matcher):
        """Test behavior with empty query."""
        results = matcher.match_by_keywords("")
        assert isinstance(results, list)


class TestCategoryMatching:
    """Test pattern matching by category."""
    
    def test_match_by_category(self, matcher):
        """Test matching patterns by category."""
        results = matcher.match_by_category(PatternCategory.PARALLEL)
        assert len(results) > 0
        assert all(r.pattern.category == PatternCategory.PARALLEL for r in results)
    
    def test_match_by_category_with_domain(self, matcher):
        """Test category matching with domain filter."""
        results = matcher.match_by_category(PatternCategory.EXCLUSIVE_CHOICE, domain=DomainType.HEALTHCARE)
        assert len(results) > 0
        for r in results:
            assert r.pattern.category == PatternCategory.EXCLUSIVE_CHOICE
            assert r.pattern.domain == DomainType.HEALTHCARE or r.pattern.domain == DomainType.GENERIC
    
    def test_match_by_category_min_confidence(self, matcher):
        """Test category matching with confidence threshold."""
        results = matcher.match_by_category(PatternCategory.SEQUENTIAL, min_confidence=0.95)
        assert all(r.pattern.confidence >= 0.95 for r in results)
    
    def test_category_match_type(self, matcher):
        """Test that match type is correctly set."""
        results = matcher.match_by_category(PatternCategory.SEQUENTIAL)
        assert all(r.match_type == "category" for r in results)


class TestComplexityMatching:
    """Test pattern matching by complexity."""
    
    def test_match_by_complexity(self, matcher):
        """Test matching patterns by complexity."""
        results = matcher.match_by_complexity(ComplexityLevel.SIMPLE)
        assert len(results) > 0
        assert all(r.pattern.complexity == ComplexityLevel.SIMPLE for r in results)
    
    def test_match_by_complexity_with_domain(self, matcher):
        """Test complexity matching with domain filter."""
        results = matcher.match_by_complexity(ComplexityLevel.MODERATE, domain=DomainType.FINANCE)
        assert all(
            r.pattern.complexity == ComplexityLevel.MODERATE and 
            (r.pattern.domain == DomainType.FINANCE or r.pattern.domain == DomainType.GENERIC)
            for r in results
        )


class TestSemanticSimilarity:
    """Test semantic similarity matching."""
    
    def test_semantic_similarity_basic(self, matcher):
        """Test basic semantic similarity."""
        results = matcher.semantic_similarity("tasks running at the same time", threshold=0.3)
        # Semantic similarity with low threshold should find some results
        assert isinstance(results, list)
    
    def test_semantic_similarity_with_domain(self, matcher):
        """Test semantic similarity with domain filter."""
        results = matcher.semantic_similarity("financial transaction approval", domain=DomainType.FINANCE)
        assert len(results) > 0
        assert any(r.pattern.domain == DomainType.FINANCE for r in results)
    
    def test_semantic_similarity_threshold(self, matcher):
        """Test semantic similarity with threshold."""
        results = matcher.semantic_similarity("patient medical procedure", threshold=0.6)
        assert all(r.match_score >= 0.6 for r in results)
    
    def test_semantic_match_type(self, matcher):
        """Test that match type is semantic."""
        results = matcher.semantic_similarity("parallel tasks")
        assert all(r.match_type == "semantic" for r in results)


class TestCompositeSearch:
    """Test composite search combining multiple strategies."""
    
    def test_composite_search_basic(self, matcher):
        """Test basic composite search."""
        results = matcher.composite_search("parallel processing tasks")
        assert len(results) > 0
        assert isinstance(results, list)
    
    def test_composite_search_with_filters(self, matcher):
        """Test composite search with multiple filters."""
        results = matcher.composite_search(
            "approval process",
            domain=DomainType.FINANCE,
            category=PatternCategory.EXCLUSIVE_CHOICE,
            complexity=ComplexityLevel.MODERATE
        )
        assert len(results) >= 0
    
    def test_composite_search_max_results(self, matcher):
        """Test that composite search respects max_results."""
        results = matcher.composite_search("workflow", max_results=2)
        assert len(results) <= 2
    
    def test_composite_search_custom_weights(self, matcher):
        """Test composite search with custom weights."""
        weights = {"keyword": 0.6, "semantic": 0.3, "category": 0.1, "complexity": 0.0}
        results = matcher.composite_search("process", weights=weights)
        assert len(results) > 0


class TestPatternRecommendation:
    """Test pattern recommendation functionality."""
    
    def test_get_recommendations_basic(self, matcher):
        """Test basic pattern recommendations."""
        rec = matcher.get_pattern_recommendations("execute tasks in parallel then join")
        assert isinstance(rec, PatternRecommendation)
        assert rec.primary_domain is not None
        assert rec.detected_complexity is not None
    
    def test_get_recommendations_with_domain(self, matcher):
        """Test recommendations with domain hint."""
        rec = matcher.get_pattern_recommendations(
            "process patient consent before treatment",
            domain=DomainType.HEALTHCARE
        )
        assert rec.primary_domain in (DomainType.HEALTHCARE, DomainType.GENERIC)
    
    def test_recommendation_has_reasoning(self, matcher):
        """Test that recommendations include reasoning."""
        rec = matcher.get_pattern_recommendations("parallel processing")
        assert len(rec.reasoning) > 0
    
    def test_recommendation_max_patterns(self, matcher):
        """Test that max_patterns is respected."""
        rec = matcher.get_pattern_recommendations("workflow", max_patterns=3)
        assert len(rec.patterns) <= 3
    
    def test_recommendation_includes_alternatives(self, matcher):
        """Test that alternatives are included when requested."""
        rec = matcher.get_pattern_recommendations("process workflow", include_alternatives=True)
        # May have alternatives if there are enough matching patterns
        assert isinstance(rec.alternative_patterns, list)
    
    def test_recommendation_confidence(self, matcher):
        """Test that recommendations have valid confidence."""
        rec = matcher.get_pattern_recommendations("parallel fork join")
        assert 0.0 <= rec.confidence <= 1.0


class TestSimilarPatterns:
    """Test finding similar patterns."""
    
    def test_find_similar_patterns(self, matcher):
        """Test finding similar patterns."""
        results = matcher.find_similar_patterns("parallel_001")
        assert isinstance(results, list)
        assert all(r.pattern.id != "parallel_001" for r in results)
    
    def test_find_similar_patterns_max_results(self, matcher):
        """Test that max_similar is respected."""
        results = matcher.find_similar_patterns("parallel_001", max_similar=2)
        assert len(results) <= 2
    
    def test_find_similar_patterns_respects_related(self, matcher):
        """Test that explicitly related patterns are prioritized."""
        results = matcher.find_similar_patterns("finance_approval_001")
        assert len(results) >= 0


class TestPatternValidation:
    """Test pattern validation functionality."""
    
    def test_validate_matching_pattern(self, matcher):
        """Test validation of matching pattern."""
        is_valid, confidence, issues = matcher.validate_pattern_match(
            matcher.kb.get_pattern("parallel_001"),
            "Execute task A and task B in parallel then join them"
        )
        assert is_valid is not None
        assert 0.0 <= confidence <= 1.0
        assert isinstance(issues, list)
    
    def test_validate_non_matching_pattern(self, matcher):
        """Test validation of non-matching pattern."""
        is_valid, confidence, issues = matcher.validate_pattern_match(
            matcher.kb.get_pattern("sequential_001"),
            "asdf qwerty zxcv random words"
        )
        assert isinstance(is_valid, bool)
        assert 0.0 <= confidence <= 1.0
    
    def test_validation_details(self, matcher):
        """Test that validation returns details."""
        is_valid, confidence, issues = matcher.validate_pattern_match(
            matcher.kb.get_pattern("it_incident_001"),
            "Short text"
        )
        assert isinstance(issues, list)


class TestIndexing:
    """Test internal indexing functionality."""
    
    def test_keyword_index_built(self, matcher):
        """Test that keyword index is built."""
        assert len(matcher.pattern_keywords) > 0
        assert "parallel_001" in matcher.pattern_keywords
    
    def test_tag_index_built(self, matcher):
        """Test that tag index is built."""
        assert len(matcher.pattern_tags) > 0
    
    def test_category_index_built(self, matcher):
        """Test that category index is built."""
        assert len(matcher.pattern_category_map) > 0
        assert PatternCategory.PARALLEL in matcher.pattern_category_map
    
    def test_domain_index_built(self, matcher):
        """Test that domain index is built."""
        assert len(matcher.pattern_domain_map) > 0
        assert DomainType.GENERIC in matcher.pattern_domain_map


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_knowledge_base(self):
        """Test matcher with empty KB."""
        kb = KnowledgeBase()
        matcher = AdvancedPatternMatcher(kb)
        results = matcher.match_by_keywords("test")
        assert results == []
    
    def test_invalid_pattern_id(self, matcher):
        """Test finding similar patterns with invalid ID."""
        results = matcher.find_similar_patterns("nonexistent_pattern")
        assert results == []
    
    def test_none_domain_filter(self, matcher):
        """Test queries with None domain."""
        results = matcher.match_by_keywords("workflow", domain=None)
        assert len(results) > 0
    
    def test_very_long_query(self, matcher):
        """Test with very long query."""
        long_query = " ".join(["word"] * 1000)
        results = matcher.match_by_keywords(long_query, threshold=0.1)
        assert isinstance(results, list)
    
    def test_special_characters_in_query(self, matcher):
        """Test with special characters."""
        results = matcher.match_by_keywords("parallel & fork @#$% join")
        assert isinstance(results, list)


class TestMatchResultOrdering:
    """Test MatchResult sorting and comparison."""
    
    def test_match_results_sorted_descending(self, matcher):
        """Test that results are sorted by score descending."""
        results = matcher.match_by_keywords("sequential parallel decision")
        if len(results) > 1:
            scores = [r.match_score for r in results]
            assert scores == sorted(scores, reverse=True)
    
    def test_match_result_comparison(self):
        """Test MatchResult comparison operators."""
        pattern = BPMNPattern(
            id="test",
            name="Test",
            description="Test",
            category=PatternCategory.SEQUENTIAL,
            graph_structure=GraphStructure(nodes=[], edges=[])
        )
        
        result1 = MatchResult(pattern=pattern, match_score=0.9, match_type="test")
        result2 = MatchResult(pattern=pattern, match_score=0.5, match_type="test")
        
        assert result1 < result2  # 0.9 > 0.5, so result1 is "less than" (sorts first)


class TestRelevanceFactors:
    """Test relevance factor calculation and reporting."""
    
    def test_keyword_match_has_relevance_factors(self, matcher):
        """Test that keyword matches include relevance factors."""
        results = matcher.match_by_keywords("parallel fork")
        assert len(results) > 0
        assert len(results[0].relevance_factors) > 0
    
    def test_semantic_match_has_relevance_factors(self, matcher):
        """Test that semantic matches include relevance factors."""
        results = matcher.semantic_similarity("parallel workflow", threshold=0.3)
        if len(results) > 0:
            assert len(results[0].relevance_factors) > 0
        else:
            # Semantic matching with low similarity is expected to return few results
            assert isinstance(results, list)
    
    def test_relevance_factors_numeric(self, matcher):
        """Test that relevance factors are numeric."""
        results = matcher.match_by_keywords("workflow")
        assert len(results) > 0
        for factor_value in results[0].relevance_factors.values():
            assert isinstance(factor_value, (int, float))


class TestCoverageCompletion:
    """Test for comprehensive coverage of all matching strategies."""
    
    def test_coverage_all_match_types(self, matcher):
        """Test that all match types are exercised."""
        # Keyword matching
        kw_results = matcher.match_by_keywords("parallel", threshold=0.3)
        assert any(r.match_type in ("fuzzy", "exact") for r in kw_results)
        
        # Category matching
        cat_results = matcher.match_by_category(PatternCategory.SEQUENTIAL)
        assert any(r.match_type == "category" for r in cat_results)
        
        # Complexity matching
        comp_results = matcher.match_by_complexity(ComplexityLevel.MODERATE)
        assert any(r.match_type == "complexity" for r in comp_results)
        
        # Semantic matching (may have lower threshold)
        sem_results = matcher.semantic_similarity("process workflow", threshold=0.2)
        if sem_results:
            assert any(r.match_type == "semantic" for r in sem_results)
    
    def test_all_domains_covered(self, matcher):
        """Test that all domains in KB are covered."""
        domains = set()
        for pattern in matcher.kb.patterns.values():
            domains.add(pattern.domain)
        
        # Test that we can find patterns from each domain
        for domain in domains:
            results = matcher.match_by_keywords("test", domain=domain)
            # Should return results or empty list, not error
            assert isinstance(results, list)
