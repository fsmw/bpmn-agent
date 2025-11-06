"""
Unit Tests for Validation Result Mapper (Phase 4)

Tests covering:
- XSD result mapping
- RAG result mapping
- Result merging
- Format conversion
"""

import pytest
from unittest.mock import Mock

from bpmn_agent.validation.result_mapper import (
    ValidationResultMapper,
    MappedValidationError,
    MergedValidationResult
)
from bpmn_agent.validation.enhanced_xsd_validation import (
    XSDValidationResult,
    XSDValidationError,
    XSDValidationErrorLevel,
    ValidationErrorCategory
)
from bpmn_agent.validation.rag_pattern_validator import (
    RAGValidationResult,
    PatternComplianceFinding
)
from bpmn_agent.validation.integration_layer import UnifiedValidationResult
from bpmn_agent.models.knowledge_base import DomainType


@pytest.fixture
def sample_xsd_result():
    """Sample XSD validation result."""
    return XSDValidationResult(
        is_valid=True,
        quality_score=0.85,
        errors=[
            XSDValidationError(
                level=XSDValidationErrorLevel.ERROR,
                category=ValidationErrorCategory.SYNTAX,
                message="Test error",
                element_id="test_element",
                element_name="Test Element",
                line_number=10,
                suggestion="Fix this"
            )
        ],
        warnings=[
            XSDValidationError(
                level=XSDValidationErrorLevel.WARNING,
                category=ValidationErrorCategory.SEMANTIC,
                message="Test warning",
                element_id="test_element_2",
                element_name="Test Element 2"
            )
        ],
        suggestions=["Suggestion 1", "Suggestion 2"]
    )


@pytest.fixture
def sample_rag_result():
    """Sample RAG validation result."""
    return RAGValidationResult(
        findings=[
            PatternComplianceFinding(
                pattern_id="pattern_1",
                pattern_name="Test Pattern",
                structure_compliance=0.9,
                element_compliance=0.8,
                relation_compliance=0.85,
                overall_score=0.85,
                issues=["Issue 1"],
                suggestions=["Suggestion 1"]
            )
        ],
        overall_compliance_score=0.85,
        patterns_validated=1,
        patterns_passed=1
    )


@pytest.fixture
def sample_unified_result(sample_xsd_result, sample_rag_result):
    """Sample unified validation result."""
    return UnifiedValidationResult(
        xsd_result=sample_xsd_result,
        rag_result=sample_rag_result,
        overall_valid=True,
        overall_quality_score=0.85,
        combined_issues=["Issue 1"],
        combined_suggestions=["Suggestion 1"],
        patterns_applied=["pattern_1"],
        domain=DomainType.GENERIC
    )


class TestValidationResultMapper:
    """Tests for ValidationResultMapper."""
    
    def test_initialization(self):
        """Test mapper initialization."""
        mapper = ValidationResultMapper()
        assert mapper is not None
        assert hasattr(mapper, 'map_xsd_result')
        assert hasattr(mapper, 'map_rag_result')
        assert hasattr(mapper, 'merge_results')
    
    def test_map_xsd_result(self, sample_xsd_result):
        """Test mapping XSD result to unified format."""
        mapper = ValidationResultMapper()
        
        mapped_errors = mapper.map_xsd_result(sample_xsd_result)
        
        assert isinstance(mapped_errors, list)
        assert len(mapped_errors) == 2  # 1 error + 1 warning
        
        error = mapped_errors[0]
        assert isinstance(error, MappedValidationError)
        assert error.source == 'xsd'
        assert error.level == 'error'
        assert error.category == 'syntax'
        assert error.message == "Test error"
        assert error.element_id == "test_element"
        assert error.suggestion == "Fix this"
    
    def test_map_xsd_result_empty(self):
        """Test mapping empty XSD result."""
        mapper = ValidationResultMapper()
        
        empty_result = XSDValidationResult(
            is_valid=True,
            quality_score=1.0,
            errors=[],
            warnings=[],
            suggestions=[]
        )
        
        mapped_errors = mapper.map_xsd_result(empty_result)
        assert isinstance(mapped_errors, list)
        assert len(mapped_errors) == 0
    
    def test_map_rag_result(self, sample_rag_result):
        """Test mapping RAG result to unified format."""
        mapper = ValidationResultMapper()
        
        mapped_errors = mapper.map_rag_result(sample_rag_result)
        
        assert isinstance(mapped_errors, list)
        assert len(mapped_errors) >= 0
        
        if len(mapped_errors) > 0:
            error = mapped_errors[0]
            assert isinstance(error, MappedValidationError)
            assert error.source == 'rag'
            assert error.pattern_id == "pattern_1"
    
    def test_map_rag_result_empty(self):
        """Test mapping empty RAG result."""
        mapper = ValidationResultMapper()
        
        empty_result = RAGValidationResult(
            findings=[],
            overall_compliance_score=1.0,
            patterns_validated=0,
            patterns_passed=0
        )
        
        mapped_errors = mapper.map_rag_result(empty_result)
        assert isinstance(mapped_errors, list)
    
    def test_merge_results(self, sample_unified_result):
        """Test merging unified result."""
        mapper = ValidationResultMapper()
        
        merged = mapper.merge_results(sample_unified_result)
        
        assert isinstance(merged, MergedValidationResult)
        assert merged.is_valid == True
        assert merged.quality_score == 0.85
        assert isinstance(merged.errors, list)
        assert isinstance(merged.warnings, list)
        assert isinstance(merged.suggestions, list)
        assert merged.patterns_applied == ["pattern_1"]
    
    def test_merge_results_no_rag(self, sample_xsd_result):
        """Test merging result without RAG."""
        mapper = ValidationResultMapper()
        
        unified = UnifiedValidationResult(
            xsd_result=sample_xsd_result,
            rag_result=None,
            overall_valid=True,
            overall_quality_score=0.85
        )
        
        merged = mapper.merge_results(unified)
        
        assert isinstance(merged, MergedValidationResult)
        assert merged.is_valid == True
    
    def test_to_dict(self, sample_unified_result):
        """Test converting merged result to dictionary."""
        mapper = ValidationResultMapper()
        
        merged = mapper.merge_results(sample_unified_result)
        result_dict = mapper.to_dict(merged)
        
        assert isinstance(result_dict, dict)
        assert "is_valid" in result_dict
        assert "quality_score" in result_dict
        assert "errors" in result_dict
        assert "warnings" in result_dict
        assert "suggestions" in result_dict
    
    def test_to_summary_string(self, sample_unified_result):
        """Test converting merged result to summary string."""
        mapper = ValidationResultMapper()
        
        merged = mapper.merge_results(sample_unified_result)
        summary = mapper.to_summary_string(merged)
        
        assert isinstance(summary, str)
        assert len(summary) > 0


@pytest.mark.unit
class TestValidationResultMapperIntegration:
    """Integration tests for ValidationResultMapper."""
    
    def test_full_mapping_workflow(self, sample_xsd_result, sample_rag_result):
        """Test complete mapping workflow."""
        mapper = ValidationResultMapper()
        
        # Map XSD result
        xsd_mapped = mapper.map_xsd_result(sample_xsd_result)
        assert isinstance(xsd_mapped, list)
        
        # Map RAG result
        rag_mapped = mapper.map_rag_result(sample_rag_result)
        assert isinstance(rag_mapped, list)
        
        # Create unified result
        unified = UnifiedValidationResult(
            xsd_result=sample_xsd_result,
            rag_result=sample_rag_result,
            overall_valid=True,
            overall_quality_score=0.85
        )
        
        # Merge results
        merged = mapper.merge_results(unified)
        assert isinstance(merged, MergedValidationResult)
        
        # Convert to dict
        result_dict = mapper.to_dict(merged)
        assert isinstance(result_dict, dict)
        
        # Convert to summary
        summary = mapper.to_summary_string(merged)
        assert isinstance(summary, str)
