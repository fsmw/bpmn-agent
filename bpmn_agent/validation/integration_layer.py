"""
Validation Integration Layer for Phase 4

Coordinates base validations + RAG validations and hides integration complexity.
Provides unified interface for validation orchestration.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from bpmn_agent.models.extraction import ExtractionResultWithErrors
from bpmn_agent.models.graph import ProcessGraph
from bpmn_agent.models.knowledge_base import DomainType
from bpmn_agent.validation.enhanced_xsd_validation import EnhancedXSDValidator, XSDValidationResult
from bpmn_agent.validation.rag_feedback_loop import RAGFeedbackLoop
from bpmn_agent.validation.rag_pattern_validator import RAGPatternValidator, RAGValidationResult

logger = logging.getLogger(__name__)


@dataclass
class UnifiedValidationResult:
    """Unified validation result combining XSD and RAG validations."""

    xsd_result: XSDValidationResult
    rag_result: Optional[RAGValidationResult] = None
    overall_valid: bool = True
    overall_quality_score: float = 0.0
    combined_issues: List[str] = field(default_factory=list)
    combined_suggestions: List[str] = field(default_factory=list)
    patterns_applied: List[str] = field(default_factory=list)
    domain: Optional[DomainType] = None


class ValidationIntegrationLayer:
    """
    Integration layer that coordinates base + RAG validations.

    Hides complexity of integrating multiple validators.
    Provides unified interface for validation orchestration.

    Follows Facade Pattern to simplify complex subsystem.
    Uses Dependency Injection for testability.
    """

    def __init__(
        self,
        xsd_validator: Optional[EnhancedXSDValidator] = None,
        rag_validator: Optional[RAGPatternValidator] = None,
        feedback_loop: Optional[RAGFeedbackLoop] = None,
        enable_rag: bool = True,
    ):
        """
        Initialize validation integration layer.

        Args:
            xsd_validator: XSD validator instance (optional, creates default if None)
            rag_validator: RAG pattern validator (optional, creates default if None)
            feedback_loop: Feedback loop for RAG improvement (optional)
            enable_rag: Whether to enable RAG validation
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.enable_rag = enable_rag

        # Initialize validators with graceful degradation
        if xsd_validator is None:
            self.xsd_validator = EnhancedXSDValidator(enable_kb_patterns=enable_rag)
        else:
            self.xsd_validator = xsd_validator

        if enable_rag:
            try:
                if rag_validator is None:
                    self.rag_validator = RAGPatternValidator()
                else:
                    self.rag_validator = rag_validator

                if feedback_loop is None:
                    self.feedback_loop = RAGFeedbackLoop()
                else:
                    self.feedback_loop = feedback_loop

                self.rag_enabled = True
                self.logger.info("ValidationIntegrationLayer initialized with RAG support")
            except Exception as e:
                self.logger.warning(f"RAG validation disabled due to error: {e}")
                self.rag_validator = None
                self.feedback_loop = None
                self.rag_enabled = False
        else:
            self.rag_validator = None
            self.feedback_loop = None
            self.rag_enabled = False
            self.logger.info("ValidationIntegrationLayer initialized without RAG support")

    def validate(
        self,
        xml_content: str,
        graph: Optional[ProcessGraph] = None,
        extraction_result: Optional[ExtractionResultWithErrors] = None,
        domain: Optional[DomainType] = None,
        patterns_applied: Optional[List[str]] = None,
    ) -> UnifiedValidationResult:
        """
        Perform unified validation (XSD + RAG).

        Args:
            xml_content: BPMN XML content to validate
            graph: Optional process graph for context
            extraction_result: Optional extraction result for semantic validation
            domain: Optional domain context
            patterns_applied: Optional list of pattern IDs that were applied

        Returns:
            UnifiedValidationResult combining all validation results
        """
        self.logger.info("Starting unified validation")

        # Step 1: XSD validation (always performed)
        xsd_result = self.xsd_validator.validate_xml_against_xsd(
            xml_content=xml_content,
            graph=graph,
            extraction_result=extraction_result,
            domain=domain.value if domain else None,
            patterns_applied=patterns_applied,
        )

        # Step 2: RAG pattern validation (if enabled)
        rag_result = None
        if self.rag_enabled and patterns_applied:
            try:
                rag_result = self.rag_validator.validate_pattern_compliance(
                    xml_content=xml_content,
                    patterns_applied=patterns_applied,
                    graph=graph,
                    domain=domain.value if domain else None,
                )
                self.logger.debug(
                    f"RAG validation completed: {rag_result.patterns_validated} patterns validated, "
                    f"compliance score: {rag_result.overall_compliance_score:.2f}"
                )
            except Exception as e:
                self.logger.error(f"RAG validation failed: {e}")
                rag_result = None

        # Step 3: Combine results
        unified_result = self._combine_results(
            xsd_result=xsd_result,
            rag_result=rag_result,
            patterns_applied=patterns_applied or [],
            domain=domain,
        )

        # Step 4: Record feedback (if RAG enabled)
        if self.rag_enabled and self.feedback_loop and patterns_applied:
            try:
                self.feedback_loop.record_validation_findings(
                    validation_result=xsd_result, patterns_applied=patterns_applied, domain=domain
                )
            except Exception as e:
                self.logger.warning(f"Failed to record feedback: {e}")

        self.logger.info(
            f"Unified validation completed: valid={unified_result.overall_valid}, "
            f"quality={unified_result.overall_quality_score:.2f}"
        )

        return unified_result

    def _combine_results(
        self,
        xsd_result: XSDValidationResult,
        rag_result: Optional[RAGValidationResult],
        patterns_applied: List[str],
        domain: Optional[DomainType],
    ) -> UnifiedValidationResult:
        """Combine XSD and RAG validation results."""
        # Start with XSD result as base
        overall_valid = xsd_result.is_valid
        overall_quality = xsd_result.quality_score

        # Combine issues
        combined_issues = []
        combined_suggestions = []

        # Add XSD issues
        for error in xsd_result.errors:
            combined_issues.append(f"[XSD] {error.message}")
            if error.suggestion:
                combined_suggestions.append(error.suggestion)

        # Add RAG issues if available
        if rag_result:
            # Adjust overall quality based on RAG compliance
            rag_weight = 0.2  # RAG contributes 20% to overall quality
            xsd_weight = 0.8
            overall_quality = (
                xsd_result.quality_score * xsd_weight
                + rag_result.overall_compliance_score * rag_weight
            )

            # Add RAG findings as issues if compliance is low
            for finding in rag_result.findings:
                if finding.overall_score < 0.8:
                    combined_issues.append(
                        f"[RAG Pattern] {finding.pattern_name}: {', '.join(finding.issues[:2])}"
                    )
                    combined_suggestions.extend(finding.suggestions[:2])

        # Determine overall validity
        # Invalid if XSD is invalid OR RAG compliance is very low
        if rag_result and rag_result.overall_compliance_score < 0.5:
            overall_valid = False

        return UnifiedValidationResult(
            xsd_result=xsd_result,
            rag_result=rag_result,
            overall_valid=overall_valid,
            overall_quality_score=overall_quality,
            combined_issues=combined_issues,
            combined_suggestions=combined_suggestions,
            patterns_applied=patterns_applied,
            domain=domain,
        )

    def get_validation_summary(self, result: UnifiedValidationResult) -> Dict[str, Any]:
        """
        Get human-readable validation summary.

        Args:
            result: Unified validation result

        Returns:
            Dictionary with summary information
        """
        summary = {
            "valid": result.overall_valid,
            "quality_score": result.overall_quality_score,
            "xsd_errors": result.xsd_result.total_errors,
            "xsd_warnings": result.xsd_result.total_warnings,
            "total_issues": len(result.combined_issues),
            "suggestions_count": len(result.combined_suggestions),
        }

        if result.rag_result:
            summary["rag_compliance"] = result.rag_result.overall_compliance_score
            summary["patterns_validated"] = result.rag_result.patterns_validated
            summary["patterns_passed"] = result.rag_result.patterns_passed

        if result.patterns_applied:
            summary["patterns_applied"] = result.patterns_applied

        if result.domain:
            summary["domain"] = result.domain.value

        return summary

    def get_feedback_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get feedback summary from RAG feedback loop.

        Returns:
            Feedback summary dictionary or None if RAG disabled
        """
        if not self.rag_enabled or not self.feedback_loop:
            return None

        return self.feedback_loop.get_feedback_summary()
