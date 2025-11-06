"""
Validation Result Mapper for Phase 4

Maps and merges validation results between different formats.
Provides unified result conversion without information loss.
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

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

logger = logging.getLogger(__name__)


@dataclass
class MappedValidationError:
    """Mapped validation error with unified format."""
    level: str
    category: str
    message: str
    source: str  # 'xsd' or 'rag'
    element_id: Optional[str] = None
    element_name: Optional[str] = None
    line_number: Optional[int] = None
    suggestion: Optional[str] = None
    pattern_id: Optional[str] = None
    compliance_score: Optional[float] = None


@dataclass
class MergedValidationResult:
    """Merged validation result combining all sources."""
    is_valid: bool
    quality_score: float
    errors: List[MappedValidationError] = field(default_factory=list)
    warnings: List[MappedValidationError] = field(default_factory=list)
    info: List[MappedValidationError] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    patterns_applied: List[str] = field(default_factory=list)
    domain: Optional[str] = None


class ValidationResultMapper:
    """
    Maps and merges validation results between different formats.
    
    Provides conversion between:
    - XSDValidationResult
    - RAGValidationResult
    - UnifiedValidationResult
    - MergedValidationResult (unified format)
    
    Follows Adapter Pattern for format conversion.
    """
    
    def __init__(self):
        """Initialize result mapper."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def map_xsd_result(self, xsd_result: XSDValidationResult) -> List[MappedValidationError]:
        """
        Map XSD validation result to unified error format.
        
        Args:
            xsd_result: XSD validation result
            
        Returns:
            List of mapped validation errors
        """
        mapped_errors = []
        
        for error in xsd_result.errors:
            mapped_error = MappedValidationError(
                level=error.level.value,
                category=error.category.value,
                message=error.message,
                source='xsd',
                element_id=error.element_id,
                element_name=error.element_name,
                line_number=error.line_number,
                suggestion=error.suggestion
            )
            mapped_errors.append(mapped_error)
        
        return mapped_errors
    
    def map_rag_result(self, rag_result: RAGValidationResult) -> List[MappedValidationError]:
        """
        Map RAG validation result to unified error format.
        
        Args:
            rag_result: RAG validation result
            
        Returns:
            List of mapped validation errors
        """
        mapped_errors = []
        
        for finding in rag_result.findings:
            if finding.overall_score < 0.8:
                # Create error for low compliance
                level = 'error' if finding.overall_score < 0.6 else 'warning'
                
                mapped_error = MappedValidationError(
                    level=level,
                    category='pattern_compliance',
                    message=f"Pattern '{finding.pattern_name}' compliance: {finding.overall_score:.2f}",
                    source='rag',
                    pattern_id=finding.pattern_id,
                    suggestion='; '.join(finding.suggestions[:2]) if finding.suggestions else None,
                    compliance_score=finding.overall_score
                )
                mapped_errors.append(mapped_error)
                
                # Add individual issues
                for issue in finding.issues[:2]:
                    mapped_error = MappedValidationError(
                        level='info',
                        category='pattern_issue',
                        message=issue,
                        source='rag',
                        pattern_id=finding.pattern_id,
                        suggestion=finding.suggestions[0] if finding.suggestions else None
                    )
                    mapped_errors.append(mapped_error)
        
        return mapped_errors
    
    def merge_results(
        self,
        unified_result: UnifiedValidationResult
    ) -> MergedValidationResult:
        """
        Merge unified validation result into single merged result.
        
        Args:
            unified_result: Unified validation result
            
        Returns:
            Merged validation result
        """
        # Map XSD errors
        xsd_errors = self.map_xsd_result(unified_result.xsd_result)
        
        # Map RAG errors if available
        rag_errors = []
        if unified_result.rag_result:
            rag_errors = self.map_rag_result(unified_result.rag_result)
        
        # Combine all errors
        all_errors = xsd_errors + rag_errors
        
        # Separate by level
        errors = [e for e in all_errors if e.level in ['fatal', 'error']]
        warnings = [e for e in all_errors if e.level == 'warning']
        info = [e for e in all_errors if e.level == 'info']
        
        # Extract unique suggestions
        suggestions = list(set(
            [e.suggestion for e in all_errors if e.suggestion] +
            unified_result.combined_suggestions
        ))
        
        # Build metrics
        metrics = {
            'xsd_errors': unified_result.xsd_result.total_errors,
            'xsd_warnings': unified_result.xsd_result.total_warnings,
            'xsd_quality': unified_result.xsd_result.quality_score
        }
        
        if unified_result.rag_result:
            metrics['rag_compliance'] = unified_result.rag_result.overall_compliance_score
            metrics['patterns_validated'] = unified_result.rag_result.patterns_validated
            metrics['patterns_passed'] = unified_result.rag_result.patterns_passed
        
        return MergedValidationResult(
            is_valid=unified_result.overall_valid,
            quality_score=unified_result.overall_quality_score,
            errors=errors,
            warnings=warnings,
            info=info,
            suggestions=suggestions,
            metrics=metrics,
            patterns_applied=unified_result.patterns_applied,
            domain=unified_result.domain.value if unified_result.domain else None
        )
    
    def to_dict(self, merged_result: MergedValidationResult) -> Dict[str, Any]:
        """
        Convert merged result to dictionary format.
        
        Args:
            merged_result: Merged validation result
            
        Returns:
            Dictionary representation
        """
        return {
            'valid': merged_result.is_valid,
            'quality_score': merged_result.quality_score,
            'errors': [
                {
                    'level': e.level,
                    'category': e.category,
                    'message': e.message,
                    'source': e.source,
                    'element_id': e.element_id,
                    'suggestion': e.suggestion
                }
                for e in merged_result.errors
            ],
            'warnings': [
                {
                    'level': e.level,
                    'category': e.category,
                    'message': e.message,
                    'source': e.source,
                    'suggestion': e.suggestion
                }
                for e in merged_result.warnings
            ],
            'info': [
                {
                    'level': e.level,
                    'category': e.category,
                    'message': e.message,
                    'source': e.source
                }
                for e in merged_result.info
            ],
            'suggestions': merged_result.suggestions,
            'metrics': merged_result.metrics,
            'patterns_applied': merged_result.patterns_applied,
            'domain': merged_result.domain
        }
    
    def to_summary_string(self, merged_result: MergedValidationResult) -> str:
        """
        Convert merged result to human-readable summary string.
        
        Args:
            merged_result: Merged validation result
            
        Returns:
            Summary string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("VALIDATION SUMMARY")
        lines.append("=" * 60)
        lines.append(f"Valid: {'✅ YES' if merged_result.is_valid else '❌ NO'}")
        lines.append(f"Quality Score: {merged_result.quality_score:.2%}")
        lines.append("")
        
        if merged_result.errors:
            lines.append(f"Errors ({len(merged_result.errors)}):")
            for error in merged_result.errors[:5]:
                lines.append(f"  [{error.source.upper()}] {error.message}")
                if error.suggestion:
                    lines.append(f"    → {error.suggestion}")
            if len(merged_result.errors) > 5:
                lines.append(f"  ... and {len(merged_result.errors) - 5} more errors")
            lines.append("")
        
        if merged_result.warnings:
            lines.append(f"Warnings ({len(merged_result.warnings)}):")
            for warning in merged_result.warnings[:3]:
                lines.append(f"  [{warning.source.upper()}] {warning.message}")
            if len(merged_result.warnings) > 3:
                lines.append(f"  ... and {len(merged_result.warnings) - 3} more warnings")
            lines.append("")
        
        if merged_result.suggestions:
            lines.append("Suggestions:")
            for suggestion in merged_result.suggestions[:5]:
                lines.append(f"  • {suggestion}")
            lines.append("")
        
        if merged_result.patterns_applied:
            lines.append(f"Patterns Applied: {', '.join(merged_result.patterns_applied)}")
        
        if merged_result.domain:
            lines.append(f"Domain: {merged_result.domain}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
