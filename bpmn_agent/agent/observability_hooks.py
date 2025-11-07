"""
Observability Hooks for Agent Orchestration

Provides metrics collection, tracing, and performance monitoring for:
- Pipeline stage execution
- Error recovery and fallback strategies
- Checkpoint operations
- End-to-end processing metrics
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger

from bpmn_agent.core.observability import (
    record_metric,
    span,
)


class StageMetricType(str, Enum):
    """Types of stage-level metrics."""

    EXECUTION_TIME = "stage_execution_time_ms"
    SUCCESS_RATE = "stage_success_rate"
    ERROR_COUNT = "stage_error_count"
    RETRY_COUNT = "stage_retry_count"
    FALLBACK_TRIGGERED = "stage_fallback_triggered"
    DEGRADATION_LEVEL = "stage_degradation_level"


class RecoveryMetricType(str, Enum):
    """Types of recovery-related metrics."""

    RECOVERY_ATTEMPT = "recovery_attempt_total"
    RECOVERY_SUCCESS = "recovery_success_total"
    RECOVERY_FAILURE = "recovery_failure_total"
    RETRY_BACKOFF_MS = "retry_backoff_ms"
    FALLBACK_USED = "fallback_used_total"


class CheckpointMetricType(str, Enum):
    """Types of checkpoint-related metrics."""

    CHECKPOINT_SAVE = "checkpoint_save_time_ms"
    CHECKPOINT_LOAD = "checkpoint_load_time_ms"
    CHECKPOINT_SIZE_BYTES = "checkpoint_size_bytes"
    CHECKPOINT_CLEANUP = "checkpoint_cleanup_time_ms"
    CHECKPOINT_SUCCESS = "checkpoint_success_total"


@dataclass
class StageMetrics:
    """Metrics collected for a single stage execution."""

    stage_name: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: float = 0.0
    success: bool = False
    error_count: int = 0
    retry_count: int = 0
    fallback_triggered: bool = False
    degradation_level: int = 0  # 0=normal, 1=degraded, 2=minimal
    input_size_bytes: int = 0
    output_size_bytes: int = 0
    token_count: int = 0
    error_messages: List[str] = field(default_factory=list)
    recovery_attempts: List[str] = field(default_factory=list)

    def finalize(self) -> None:
        """Finalize metrics collection."""
        if self.end_time is None:
            self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        base_dict = {
            "stage": self.stage_name,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error_count": self.error_count,
            "retry_count": self.retry_count,
            "fallback_triggered": self.fallback_triggered,
            "degradation_level": self.degradation_level,
            "input_size_bytes": self.input_size_bytes,
            "output_size_bytes": self.output_size_bytes,
            "token_count": self.token_count,
            "error_messages": self.error_messages,
            "recovery_attempts": self.recovery_attempts,
        }
        # Add custom attributes if they exist
        if hasattr(self, "_custom_attributes"):
            base_dict.update(self._custom_attributes)
        return base_dict

    def add_attribute(self, key: str, value: Any) -> None:
        """Add a custom metric attribute."""
        # Store custom attributes for later reporting
        if not hasattr(self, "_custom_attributes"):
            self._custom_attributes = {}
        self._custom_attributes[key] = value


@dataclass
class PipelineMetrics:
    """Aggregated metrics for entire pipeline execution."""

    session_id: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_duration_ms: float = 0.0
    stages: Dict[str, StageMetrics] = field(default_factory=dict)
    overall_success: bool = False
    recovery_count: int = 0
    fallback_count: int = 0
    checkpoint_saves: int = 0
    checkpoint_loads: int = 0
    total_tokens_used: int = 0
    total_input_size_bytes: int = 0
    total_output_size_bytes: int = 0

    def finalize(self) -> None:
        """Finalize all metrics."""
        if self.end_time is None:
            self.end_time = time.time()
        self.total_duration_ms = (self.end_time - self.start_time) * 1000

        # Calculate aggregates
        for stage_metrics in self.stages.values():
            stage_metrics.finalize()
            self.total_tokens_used += stage_metrics.token_count
            self.total_input_size_bytes += stage_metrics.input_size_bytes
            self.total_output_size_bytes += stage_metrics.output_size_bytes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "total_duration_ms": self.total_duration_ms,
            "overall_success": self.overall_success,
            "recovery_count": self.recovery_count,
            "fallback_count": self.fallback_count,
            "checkpoint_saves": self.checkpoint_saves,
            "checkpoint_loads": self.checkpoint_loads,
            "total_tokens_used": self.total_tokens_used,
            "total_input_size_bytes": self.total_input_size_bytes,
            "total_output_size_bytes": self.total_output_size_bytes,
            "stages": {
                stage_name: metrics.to_dict() for stage_name, metrics in self.stages.items()
            },
        }


class ObservabilityHooks:
    """Manages observability hooks for orchestration."""

    def __init__(self, enable_tracing: bool = True, enable_metrics: bool = True):
        """Initialize observability hooks.

        Args:
            enable_tracing: Enable OpenTelemetry tracing
            enable_metrics: Enable metrics collection
        """
        self.enable_tracing = enable_tracing
        self.enable_metrics = enable_metrics
        self._pipeline_metrics: Optional[PipelineMetrics] = None
        self._stage_stack: List[str] = []

    def start_pipeline(self, session_id: str) -> PipelineMetrics:
        """Start tracking a pipeline execution.

        Args:
            session_id: Unique session identifier

        Returns:
            PipelineMetrics tracker
        """
        self._pipeline_metrics = PipelineMetrics(session_id=session_id)
        logger.info("Pipeline started", extra={"session_id": session_id})
        return self._pipeline_metrics

    def end_pipeline(self, success: bool = True) -> PipelineMetrics:
        """End tracking a pipeline execution.

        Args:
            success: Whether pipeline succeeded

        Returns:
            Finalized PipelineMetrics
        """
        if self._pipeline_metrics is None:
            raise RuntimeError("No active pipeline to end")

        self._pipeline_metrics.overall_success = success
        self._pipeline_metrics.finalize()

        # Record metrics
        if self.enable_metrics:
            metrics_dict = self._pipeline_metrics.to_dict()
            record_metric(
                "pipeline_total_duration_ms",
                self._pipeline_metrics.total_duration_ms,
                attributes={"status": "success" if success else "failure"},
            )
            record_metric(
                "pipeline_recovery_count",
                self._pipeline_metrics.recovery_count,
            )
            record_metric(
                "pipeline_fallback_count",
                self._pipeline_metrics.fallback_count,
            )
            logger.info(
                "Pipeline completed",
                extra={"metrics": metrics_dict, "success": success},
            )

        return self._pipeline_metrics

    @contextmanager
    def track_stage(
        self,
        stage_name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for tracking stage execution.

        Args:
            stage_name: Name of the stage
            attributes: Optional attributes for tracing

        Yields:
            StageMetrics object for this stage
        """
        if self._pipeline_metrics is None:
            raise RuntimeError("No active pipeline")

        # Create stage metrics
        stage_metrics = StageMetrics(stage_name=stage_name)
        self._pipeline_metrics.stages[stage_name] = stage_metrics
        self._stage_stack.append(stage_name)

        # Create span if tracing enabled
        span_attrs = {"stage": stage_name, "session_id": self._pipeline_metrics.session_id}
        if attributes:
            span_attrs.update(attributes)

        try:
            if self.enable_tracing:
                with span(f"stage_{stage_name}", attributes=span_attrs) as span_obj:
                    logger.debug(f"Stage '{stage_name}' started")
                    yield stage_metrics
                    stage_metrics.finalize()
                    if span_obj:
                        span_obj.set_attribute("duration_ms", stage_metrics.duration_ms)
                        span_obj.set_attribute("success", stage_metrics.success)
            else:
                logger.debug(f"Stage '{stage_name}' started")
                yield stage_metrics
                stage_metrics.finalize()

            # Record stage metrics
            if self.enable_metrics:
                record_metric(
                    StageMetricType.EXECUTION_TIME.value,
                    stage_metrics.duration_ms,
                    attributes={"stage": stage_name},
                )
                if stage_metrics.error_count > 0:
                    record_metric(
                        StageMetricType.ERROR_COUNT.value,
                        stage_metrics.error_count,
                        attributes={"stage": stage_name},
                    )
                if stage_metrics.retry_count > 0:
                    record_metric(
                        StageMetricType.RETRY_COUNT.value,
                        stage_metrics.retry_count,
                        attributes={"stage": stage_name},
                    )
                if stage_metrics.fallback_triggered:
                    record_metric(
                        StageMetricType.FALLBACK_TRIGGERED.value,
                        1,
                        attributes={"stage": stage_name},
                    )

            logger.debug(
                f"Stage '{stage_name}' completed",
                extra=stage_metrics.to_dict(),
            )

        finally:
            self._stage_stack.pop()

    def record_error(
        self,
        error_message: str,
        error_type: str = "unknown",
        stage_name: Optional[str] = None,
    ) -> None:
        """Record an error during execution.

        Args:
            error_message: Description of the error
            error_type: Type/category of error
            stage_name: Optional stage name where error occurred
        """
        if self._pipeline_metrics is None:
            return

        # Get current stage if not specified
        if stage_name is None:
            stage_name = self._stage_stack[-1] if self._stage_stack else "unknown"

        # Record in stage metrics if available
        if stage_name in self._pipeline_metrics.stages:
            metrics = self._pipeline_metrics.stages[stage_name]
            metrics.error_count += 1
            metrics.error_messages.append(error_message)

        # Record metric
        if self.enable_metrics:
            record_metric(
                "error_total",
                1,
                attributes={"error_type": error_type, "stage": stage_name},
            )

        logger.warning(
            "Error recorded",
            extra={
                "error_message": error_message,
                "error_type": error_type,
                "stage": stage_name,
            },
        )

    def record_recovery_attempt(
        self,
        strategy: str,
        stage_name: Optional[str] = None,
        success: bool = False,
    ) -> None:
        """Record a recovery attempt.

        Args:
            strategy: Name of recovery strategy used
            stage_name: Optional stage where recovery occurred
            success: Whether recovery was successful
        """
        if self._pipeline_metrics is None:
            return

        self._pipeline_metrics.recovery_count += 1

        # Get current stage if not specified
        if stage_name is None:
            stage_name = self._stage_stack[-1] if self._stage_stack else "unknown"

        # Record in stage metrics
        if stage_name in self._pipeline_metrics.stages:
            metrics = self._pipeline_metrics.stages[stage_name]
            metrics.retry_count += 1
            metrics.recovery_attempts.append(f"{strategy}:{success}")

        # Record metric
        if self.enable_metrics:
            metric_type = (
                RecoveryMetricType.RECOVERY_SUCCESS.value
                if success
                else RecoveryMetricType.RECOVERY_FAILURE.value
            )
            record_metric(
                metric_type,
                1,
                attributes={"strategy": strategy, "stage": stage_name},
            )

        logger.info(
            f"Recovery attempt: {strategy}",
            extra={
                "strategy": strategy,
                "stage": stage_name,
                "success": success,
            },
        )

    def record_fallback(
        self,
        fallback_type: str,
        stage_name: Optional[str] = None,
        output_quality: int = 0,
    ) -> None:
        """Record a fallback to degraded mode.

        Args:
            fallback_type: Type of fallback applied
            stage_name: Optional stage where fallback occurred
            output_quality: Quality level of fallback output (0-100)
        """
        if self._pipeline_metrics is None:
            return

        self._pipeline_metrics.fallback_count += 1

        # Get current stage if not specified
        if stage_name is None:
            stage_name = self._stage_stack[-1] if self._stage_stack else "unknown"

        # Record in stage metrics
        if stage_name in self._pipeline_metrics.stages:
            metrics = self._pipeline_metrics.stages[stage_name]
            metrics.fallback_triggered = True
            metrics.degradation_level = max(1, 100 - output_quality) // 50  # 0-2

        # Record metric
        if self.enable_metrics:
            record_metric(
                RecoveryMetricType.FALLBACK_USED.value,
                1,
                attributes={"type": fallback_type, "stage": stage_name},
            )

        logger.warning(
            f"Fallback triggered: {fallback_type}",
            extra={
                "fallback_type": fallback_type,
                "stage": stage_name,
                "output_quality": output_quality,
            },
        )

    def record_checkpoint_operation(
        self,
        operation: str,
        duration_ms: float,
        size_bytes: int = 0,
        success: bool = True,
    ) -> None:
        """Record a checkpoint operation.

        Args:
            operation: Type of operation (save/load/cleanup)
            duration_ms: Duration of operation in milliseconds
            size_bytes: Size of checkpoint data
            success: Whether operation succeeded
        """
        if self._pipeline_metrics is None:
            return

        # Track save/load operations
        if operation == "save":
            self._pipeline_metrics.checkpoint_saves += 1
        elif operation == "load":
            self._pipeline_metrics.checkpoint_loads += 1

        # Record metric
        if self.enable_metrics:
            metric_type = f"checkpoint_{operation}_time_ms"
            record_metric(
                metric_type,
                duration_ms,
                attributes={"success": str(success)},
            )
            if size_bytes > 0:
                record_metric(
                    CheckpointMetricType.CHECKPOINT_SIZE_BYTES.value,
                    size_bytes,
                )

        logger.debug(
            f"Checkpoint {operation} recorded",
            extra={
                "operation": operation,
                "duration_ms": duration_ms,
                "size_bytes": size_bytes,
                "success": success,
            },
        )

    def record_stage_metrics(
        self,
        stage_name: str,
        input_size_bytes: int = 0,
        output_size_bytes: int = 0,
        token_count: int = 0,
    ) -> None:
        """Record detailed stage metrics.

        Args:
            stage_name: Name of the stage
            input_size_bytes: Size of stage input
            output_size_bytes: Size of stage output
            token_count: Number of tokens used
        """
        if self._pipeline_metrics is None:
            return

        if stage_name in self._pipeline_metrics.stages:
            metrics = self._pipeline_metrics.stages[stage_name]
            metrics.input_size_bytes = input_size_bytes
            metrics.output_size_bytes = output_size_bytes
            metrics.token_count = token_count

            if self.enable_metrics and token_count > 0:
                record_metric(
                    "stage_tokens_used",
                    token_count,
                    attributes={"stage": stage_name},
                )

    def get_current_metrics(self) -> Optional[PipelineMetrics]:
        """Get current pipeline metrics."""
        return self._pipeline_metrics


# Global observability hooks instance
_hooks: Optional[ObservabilityHooks] = None


def get_observability_hooks() -> ObservabilityHooks:
    """Get or create global observability hooks."""
    global _hooks
    if _hooks is None:
        _hooks = ObservabilityHooks()
    return _hooks


def initialize_observability_hooks(
    enable_tracing: bool = True,
    enable_metrics: bool = True,
) -> ObservabilityHooks:
    """Initialize global observability hooks.

    Args:
        enable_tracing: Enable OpenTelemetry tracing
        enable_metrics: Enable metrics collection

    Returns:
        ObservabilityHooks instance
    """
    global _hooks
    _hooks = ObservabilityHooks(
        enable_tracing=enable_tracing,
        enable_metrics=enable_metrics,
    )
    return _hooks


__all__ = [
    "StageMetricType",
    "RecoveryMetricType",
    "CheckpointMetricType",
    "StageMetrics",
    "PipelineMetrics",
    "ObservabilityHooks",
    "get_observability_hooks",
    "initialize_observability_hooks",
]
