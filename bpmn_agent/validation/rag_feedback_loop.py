"""
RAG Feedback Loop for Phase 4

Records validation findings to improve future RAG context selection.
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from bpmn_agent.models.knowledge_base import DomainType
from bpmn_agent.validation.enhanced_xsd_validation import XSDValidationResult

logger = logging.getLogger(__name__)


@dataclass
class PatternEffectivenessMetrics:
    """Metrics tracking pattern effectiveness."""
    pattern_id: str
    times_applied: int = 0
    times_successful: int = 0
    average_quality_score: float = 0.0
    common_issues: List[str] = field(default_factory=list)


class RAGFeedbackLoop:
    """
    Feedback loop to improve RAG context selection based on validation results.
    
    Follows Observer Pattern for extensibility.
    Implements in-memory metrics tracking (can be persisted later).
    """
    
    def __init__(self):
        """Initialize feedback loop."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # In-memory metrics (could be persisted later)
        self.pattern_metrics: Dict[str, PatternEffectivenessMetrics] = {}
        self.domain_metrics: Dict[DomainType, Dict[str, float]] = defaultdict(dict)
        
        self.enabled = True
    
    def record_validation_findings(
        self,
        validation_result: XSDValidationResult,
        patterns_applied: List[str],
        domain: Optional[DomainType] = None
    ) -> None:
        """
        Record validation findings for RAG improvement.
        
        Args:
            validation_result: Result from Phase 4 validation
            patterns_applied: Patterns that were applied during generation
            domain: Domain context
        """
        if not self.enabled:
            return
        
        try:
            # Update pattern metrics
            for pattern_id in patterns_applied:
                self._update_pattern_metrics(pattern_id, validation_result)
            
            # Update domain metrics
            if domain:
                self._update_domain_metrics(domain, validation_result, patterns_applied)
            
            self.logger.debug(
                f"Recorded feedback for {len(patterns_applied)} patterns, "
                f"quality score: {validation_result.quality_score:.2f}"
            )
        except Exception as e:
            self.logger.error(f"Error recording feedback: {e}")
    
    def _update_pattern_metrics(
        self,
        pattern_id: str,
        validation_result: XSDValidationResult
    ) -> None:
        """Update effectiveness metrics for a pattern."""
        if pattern_id not in self.pattern_metrics:
            self.pattern_metrics[pattern_id] = PatternEffectivenessMetrics(
                pattern_id=pattern_id
            )
        
        metrics = self.pattern_metrics[pattern_id]
        metrics.times_applied += 1
        
        # Consider successful if quality score > 0.8
        if validation_result.quality_score >= 0.8:
            metrics.times_successful += 1
        
        # Update average quality score
        total_score = metrics.average_quality_score * (metrics.times_applied - 1)
        metrics.average_quality_score = (
            (total_score + validation_result.quality_score) / metrics.times_applied
        )
        
        # Track common issues
        if validation_result.total_errors > 0:
            for error in validation_result.errors[:3]:  # Top 3 errors
                error_msg = error.message[:100]  # Truncate long messages
                if error_msg not in metrics.common_issues:
                    metrics.common_issues.append(error_msg)
                    # Keep only last 10 issues
                    if len(metrics.common_issues) > 10:
                        metrics.common_issues.pop(0)
    
    def _update_domain_metrics(
        self,
        domain: DomainType,
        validation_result: XSDValidationResult,
        patterns_applied: List[str]
    ) -> None:
        """Update domain-specific metrics."""
        if 'total_validations' not in self.domain_metrics[domain]:
            self.domain_metrics[domain]['total_validations'] = 0
            self.domain_metrics[domain]['average_quality'] = 0.0
        
        self.domain_metrics[domain]['total_validations'] += 1
        
        # Update average quality
        current_avg = self.domain_metrics[domain]['average_quality']
        total_validations = self.domain_metrics[domain]['total_validations']
        
        self.domain_metrics[domain]['average_quality'] = (
            (current_avg * (total_validations - 1) + validation_result.quality_score) / 
            total_validations
        )
    
    def get_pattern_recommendations(
        self,
        domain: DomainType,
        max_recommendations: int = 5
    ) -> List[str]:
        """
        Get pattern recommendations based on historical effectiveness.
        
        Args:
            domain: Domain to get recommendations for
            max_recommendations: Maximum number of recommendations
            
        Returns:
            List of pattern IDs recommended for this domain
        """
        # Filter patterns by effectiveness (average quality > 0.7)
        effective_patterns = [
            (pid, metrics) for pid, metrics in self.pattern_metrics.items()
            if metrics.average_quality_score > 0.7 and metrics.times_applied > 0
        ]
        
        # Sort by success rate (times_successful / times_applied)
        effective_patterns.sort(
            key=lambda x: x[1].times_successful / max(x[1].times_applied, 1),
            reverse=True
        )
        
        # Also consider average quality score as tiebreaker
        effective_patterns.sort(
            key=lambda x: (
                x[1].times_successful / max(x[1].times_applied, 1),
                x[1].average_quality_score
            ),
            reverse=True
        )
        
        return [pid for pid, _ in effective_patterns[:max_recommendations]]
    
    def get_pattern_effectiveness(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        Get effectiveness metrics for a specific pattern.
        
        Args:
            pattern_id: Pattern ID to query
            
        Returns:
            Dictionary with effectiveness metrics or None if not found
        """
        if pattern_id not in self.pattern_metrics:
            return None
        
        metrics = self.pattern_metrics[pattern_id]
        success_rate = metrics.times_successful / max(metrics.times_applied, 1)
        
        return {
            'pattern_id': pattern_id,
            'times_applied': metrics.times_applied,
            'times_successful': metrics.times_successful,
            'success_rate': success_rate,
            'average_quality_score': metrics.average_quality_score,
            'common_issues': metrics.common_issues
        }
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get summary of feedback collected."""
        # Get top patterns by average quality
        top_patterns = sorted(
            self.pattern_metrics.items(),
            key=lambda x: x[1].average_quality_score,
            reverse=True
        )[:5]
        
        return {
            'patterns_tracked': len(self.pattern_metrics),
            'domains_tracked': len(self.domain_metrics),
            'top_patterns': [
                {
                    'pattern_id': pid,
                    'success_rate': m.times_successful / max(m.times_applied, 1),
                    'avg_quality': m.average_quality_score,
                    'times_applied': m.times_applied
                }
                for pid, m in top_patterns
            ],
            'domain_metrics': {
                domain.value: {
                    'total_validations': metrics.get('total_validations', 0),
                    'average_quality': metrics.get('average_quality', 0.0)
                }
                for domain, metrics in self.domain_metrics.items()
            }
        }
    
    def reset_metrics(self) -> None:
        """Reset all collected metrics (useful for testing)."""
        self.pattern_metrics.clear()
        self.domain_metrics.clear()
        self.logger.info("Feedback metrics reset")
