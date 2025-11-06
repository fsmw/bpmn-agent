"""
Unit Tests for RAG Feedback Loop (Phase 4)

Tests covering:
- Feedback recording
- Pattern effectiveness metrics
- Domain metrics tracking
- Pattern recommendations
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import List

from bpmn_agent.validation.rag_feedback_loop import (
    RAGFeedbackLoop,
    PatternEffectivenessMetrics
)
from bpmn_agent.validation.enhanced_xsd_validation import (
    XSDValidationResult,
    XSDValidationError,
    XSDValidationErrorLevel,
    ValidationErrorCategory
)
from bpmn_agent.models.knowledge_base import DomainType


@pytest.fixture
def sample_validation_result():
    """Sample XSD validation result for testing."""
    return XSDValidationResult(
        is_valid=True,
        quality_score=0.85,
        errors=[],
        warnings=[],
        suggestions=[]
    )


@pytest.fixture
def sample_validation_result_low_quality():
    """Sample XSD validation result with low quality score."""
    return XSDValidationResult(
        is_valid=True,
        quality_score=0.65,
        errors=[
            XSDValidationError(
                level=XSDValidationErrorLevel.WARNING,
                category=ValidationErrorCategory.SEMANTIC,
                message="Test warning",
                element_id="test_element",
                element_name="Test Element"
            )
        ],
        warnings=[],
        suggestions=["Fix this issue"]
    )


class TestRAGFeedbackLoop:
    """Tests for RAGFeedbackLoop."""
    
    def test_initialization(self):
        """Test feedback loop initialization."""
        feedback_loop = RAGFeedbackLoop()
        assert feedback_loop.enabled is True
        assert feedback_loop.pattern_metrics == {}
        assert isinstance(feedback_loop.domain_metrics, dict)
    
    def test_record_validation_findings_single_pattern(self, sample_validation_result):
        """Test recording feedback for single pattern."""
        feedback_loop = RAGFeedbackLoop()
        
        feedback_loop.record_validation_findings(
            validation_result=sample_validation_result,
            patterns_applied=["pattern_1"]
        )
        
        assert "pattern_1" in feedback_loop.pattern_metrics
        metrics = feedback_loop.pattern_metrics["pattern_1"]
        assert metrics.times_applied == 1
        assert metrics.times_successful == 1  # quality_score >= 0.8
        assert metrics.average_quality_score == 0.85
    
    def test_record_validation_findings_multiple_patterns(self, sample_validation_result):
        """Test recording feedback for multiple patterns."""
        feedback_loop = RAGFeedbackLoop()
        
        feedback_loop.record_validation_findings(
            validation_result=sample_validation_result,
            patterns_applied=["pattern_1", "pattern_2"]
        )
        
        assert "pattern_1" in feedback_loop.pattern_metrics
        assert "pattern_2" in feedback_loop.pattern_metrics
        assert feedback_loop.pattern_metrics["pattern_1"].times_applied == 1
        assert feedback_loop.pattern_metrics["pattern_2"].times_applied == 1
    
    def test_record_validation_findings_with_domain(self, sample_validation_result):
        """Test recording feedback with domain context."""
        feedback_loop = RAGFeedbackLoop()
        
        feedback_loop.record_validation_findings(
            validation_result=sample_validation_result,
            patterns_applied=["pattern_1"],
            domain=DomainType.HR
        )
        
        assert DomainType.HR in feedback_loop.domain_metrics
    
    def test_record_validation_findings_disabled(self, sample_validation_result):
        """Test that disabled feedback loop doesn't record."""
        feedback_loop = RAGFeedbackLoop()
        feedback_loop.enabled = False
        
        feedback_loop.record_validation_findings(
            validation_result=sample_validation_result,
            patterns_applied=["pattern_1"]
        )
        
        assert "pattern_1" not in feedback_loop.pattern_metrics
    
    def test_record_validation_findings_low_quality(self, sample_validation_result_low_quality):
        """Test recording feedback with low quality score."""
        feedback_loop = RAGFeedbackLoop()
        
        feedback_loop.record_validation_findings(
            validation_result=sample_validation_result_low_quality,
            patterns_applied=["pattern_1"]
        )
        
        metrics = feedback_loop.pattern_metrics["pattern_1"]
        assert metrics.times_applied == 1
        assert metrics.times_successful == 0  # quality_score < 0.8
        assert metrics.average_quality_score == 0.65
    
    def test_update_pattern_metrics_new_pattern(self, sample_validation_result):
        """Test updating metrics for new pattern."""
        feedback_loop = RAGFeedbackLoop()
        
        feedback_loop._update_pattern_metrics("pattern_1", sample_validation_result)
        
        assert "pattern_1" in feedback_loop.pattern_metrics
        metrics = feedback_loop.pattern_metrics["pattern_1"]
        assert metrics.pattern_id == "pattern_1"
        assert metrics.times_applied == 1
    
    def test_update_pattern_metrics_existing_pattern(self, sample_validation_result):
        """Test updating metrics for existing pattern."""
        feedback_loop = RAGFeedbackLoop()
        
        # Record twice
        feedback_loop._update_pattern_metrics("pattern_1", sample_validation_result)
        feedback_loop._update_pattern_metrics("pattern_1", sample_validation_result)
        
        metrics = feedback_loop.pattern_metrics["pattern_1"]
        assert metrics.times_applied == 2
        assert metrics.times_successful == 2
    
    def test_update_domain_metrics(self, sample_validation_result):
        """Test updating domain metrics."""
        feedback_loop = RAGFeedbackLoop()
        
        feedback_loop._update_domain_metrics(
            domain=DomainType.HR,
            validation_result=sample_validation_result,
            patterns_applied=["pattern_1"]
        )
        
        assert DomainType.HR in feedback_loop.domain_metrics
    
    def test_get_pattern_recommendations(self, sample_validation_result):
        """Test getting pattern recommendations."""
        feedback_loop = RAGFeedbackLoop()
        
        # Record some patterns
        feedback_loop.record_validation_findings(
            validation_result=sample_validation_result,
            patterns_applied=["pattern_1", "pattern_2"],
            domain=DomainType.HR
        )
        
        recommendations = feedback_loop.get_pattern_recommendations(
            domain=DomainType.HR,
            max_recommendations=5
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5
    
    def test_get_pattern_recommendations_no_data(self):
        """Test getting recommendations when no data exists."""
        feedback_loop = RAGFeedbackLoop()
        
        recommendations = feedback_loop.get_pattern_recommendations(
            domain=DomainType.HR,
            max_recommendations=5
        )
        
        assert isinstance(recommendations, list)
    
    def test_get_pattern_effectiveness_found(self, sample_validation_result):
        """Test getting pattern effectiveness when pattern exists."""
        feedback_loop = RAGFeedbackLoop()
        
        feedback_loop.record_validation_findings(
            validation_result=sample_validation_result,
            patterns_applied=["pattern_1"]
        )
        
        effectiveness = feedback_loop.get_pattern_effectiveness("pattern_1")
        
        assert effectiveness is not None
        assert "times_applied" in effectiveness
        assert "times_successful" in effectiveness
        assert "average_quality_score" in effectiveness
    
    def test_get_pattern_effectiveness_not_found(self):
        """Test getting pattern effectiveness when pattern doesn't exist."""
        feedback_loop = RAGFeedbackLoop()
        
        effectiveness = feedback_loop.get_pattern_effectiveness("nonexistent_pattern")
        
        assert effectiveness is None
    
    def test_get_feedback_summary(self, sample_validation_result):
        """Test getting feedback summary."""
        feedback_loop = RAGFeedbackLoop()
        
        # Record some feedback
        feedback_loop.record_validation_findings(
            validation_result=sample_validation_result,
            patterns_applied=["pattern_1", "pattern_2"],
            domain=DomainType.HR
        )
        
        summary = feedback_loop.get_feedback_summary()
        
        assert isinstance(summary, dict)
        assert "total_patterns_tracked" in summary
        assert "total_domains_tracked" in summary
    
    def test_reset_metrics(self, sample_validation_result):
        """Test resetting metrics."""
        feedback_loop = RAGFeedbackLoop()
        
        # Record some feedback
        feedback_loop.record_validation_findings(
            validation_result=sample_validation_result,
            patterns_applied=["pattern_1"]
        )
        
        assert len(feedback_loop.pattern_metrics) > 0
        
        feedback_loop.reset_metrics()
        
        assert len(feedback_loop.pattern_metrics) == 0
        assert len(feedback_loop.domain_metrics) == 0


@pytest.mark.unit
class TestRAGFeedbackLoopIntegration:
    """Integration tests for RAGFeedbackLoop."""
    
    def test_multiple_recordings_accumulate(self):
        """Test that multiple recordings accumulate correctly."""
        feedback_loop = RAGFeedbackLoop()
        
        high_quality = XSDValidationResult(
            is_valid=True,
            quality_score=0.9,
            errors=[],
            warnings=[],
            suggestions=[]
        )
        
        low_quality = XSDValidationResult(
            is_valid=True,
            quality_score=0.7,
            errors=[],
            warnings=[],
            suggestions=[]
        )
        
        # Record high quality
        feedback_loop.record_validation_findings(
            validation_result=high_quality,
            patterns_applied=["pattern_1"]
        )
        
        # Record low quality
        feedback_loop.record_validation_findings(
            validation_result=low_quality,
            patterns_applied=["pattern_1"]
        )
        
        metrics = feedback_loop.pattern_metrics["pattern_1"]
        assert metrics.times_applied == 2
        assert metrics.times_successful == 1  # Only high quality counts
        assert 0.7 <= metrics.average_quality_score <= 0.9
