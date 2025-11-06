"""
Error Handling and Recovery Strategies

Provides sophisticated error handling with fallback mechanisms,
retry logic, and graceful degradation for the BPMN Agent pipeline.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Optional, Type
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RecoveryStrategy(str, Enum):
    """Strategies for recovering from stage failures."""
    
    RETRY_SAME = "retry_same"  # Retry the same stage with same params
    RETRY_DEGRADED = "retry_degraded"  # Retry with simplified params
    FALLBACK_STAGE = "fallback_stage"  # Use fallback implementation
    SKIP_STAGE = "skip_stage"  # Skip stage and use empty result
    ABORT = "abort"  # Stop processing entirely


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    
    max_retries: int = 3
    initial_delay_ms: int = 100
    max_delay_ms: int = 5000
    backoff_multiplier: float = 2.0
    jitter: bool = True
    
    def get_delay_ms(self, attempt: int) -> float:
        """Calculate delay for retry attempt (exponential backoff)."""
        delay = self.initial_delay_ms * (self.backoff_multiplier ** attempt)
        delay = min(delay, self.max_delay_ms)
        
        if self.jitter:
            import random
            jitter = random.random() * (delay * 0.1)  # ±10% jitter
            delay += jitter
        
        return delay


@dataclass
class RecoveryPlan:
    """Recovery plan for a specific stage failure."""
    
    stage_name: str
    error_type: Type[Exception]
    strategy: RecoveryStrategy
    max_attempts: int = 3
    fallback_impl: Optional[Callable[..., Any]] = None
    fallback_result: Optional[Any] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ErrorContext:
    """Context information about an error."""
    
    stage_name: str
    error: Exception
    attempt: int
    timestamp: datetime
    input_summary: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class ErrorRecoveryEngine:
    """
    Manages error recovery strategies and execution.
    
    Supports:
    - Exponential backoff retry with jitter
    - Fallback to simplified parameters
    - Stage skipping with degraded output
    - Graceful degradation
    """
    
    def __init__(self, retry_config: Optional[RetryConfig] = None):
        """Initialize error recovery engine.
        
        Args:
            retry_config: Configuration for retry behavior
        """
        self.retry_config = retry_config or RetryConfig()
        self.recovery_plans: dict[str, RecoveryPlan] = {}
        self.error_history: List[ErrorContext] = []
    
    def register_recovery_plan(self, plan: RecoveryPlan) -> None:
        """Register a recovery plan for a specific stage/error.
        
        Args:
            plan: Recovery plan to register
        """
        key = f"{plan.stage_name}:{plan.error_type.__name__}"
        self.recovery_plans[key] = plan
        logger.debug(f"Registered recovery plan for {key}")
    
    async def execute_with_recovery(
        self,
        stage_name: str,
        func: Callable[..., Any],
        *args,
        **kwargs,
    ) -> tuple[Optional[Any], Optional[ErrorContext]]:
        """Execute a function with automatic error recovery.
        
        Args:
            stage_name: Name of the stage being executed
            func: Async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            (result, error_context) - result is None if recovery failed
        """
        input_summary = self._create_input_summary(args, kwargs)
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                logger.debug(f"{stage_name}: Attempt {attempt + 1}/{self.retry_config.max_retries + 1}")
                result = await func(*args, **kwargs)
                
                # Success
                if attempt > 0:
                    logger.info(f"{stage_name}: Recovered after {attempt} retries")
                
                return result, None
                
            except Exception as e:
                error_ctx = ErrorContext(
                    stage_name=stage_name,
                    error=e,
                    attempt=attempt + 1,
                    timestamp=datetime.now(),
                    input_summary=input_summary,
                )
                self.error_history.append(error_ctx)
                
                # Last attempt - apply recovery strategy
                if attempt >= self.retry_config.max_retries:
                    logger.error(f"{stage_name}: Failed after {attempt + 1} attempts")
                    return None, error_ctx
                
                # Wait before retry (exponential backoff)
                delay_ms = self.retry_config.get_delay_ms(attempt)
                await asyncio.sleep(delay_ms / 1000.0)
                logger.warning(
                    f"{stage_name}: Attempt {attempt + 1} failed, retrying in {delay_ms:.0f}ms",
                    extra={"error": str(e)[:100]}
                )
    
    async def execute_with_fallback(
        self,
        stage_name: str,
        primary_func: Callable[..., Any],
        fallback_func: Optional[Callable[..., Any]] = None,
        *args,
        **kwargs,
    ) -> tuple[Optional[Any], Optional[ErrorContext], bool]:
        """Execute with primary function and optional fallback.
        
        Args:
            stage_name: Name of the stage
            primary_func: Primary async function to execute
            fallback_func: Fallback function to use if primary fails
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            (result, error_context, used_fallback)
        """
        # Try primary function
        result, error_ctx = await self.execute_with_recovery(
            stage_name, primary_func, *args, **kwargs
        )
        
        if result is not None:
            return result, None, False
        
        # Primary failed, try fallback
        if fallback_func is None:
            logger.error(f"{stage_name}: No fallback available")
            return None, error_ctx, False
        
        logger.warning(f"{stage_name}: Using fallback implementation")
        
        try:
            fallback_result = await fallback_func(*args, **kwargs)
            logger.info(f"{stage_name}: Fallback succeeded")
            return fallback_result, error_ctx, True
            
        except Exception as e:
            logger.error(f"{stage_name}: Fallback also failed - {e}")
            fallback_error_ctx = ErrorContext(
                stage_name=stage_name,
                error=e,
                attempt=self.retry_config.max_retries + 2,
                timestamp=datetime.now(),
                input_summary=self._create_input_summary(args, kwargs),
                metadata={"fallback_attempted": True},
            )
            self.error_history.append(fallback_error_ctx)
            return None, fallback_error_ctx, True
    
    def get_error_summary(self) -> dict:
        """Get summary of all errors encountered.
        
        Returns:
            Dictionary with error statistics and history
        """
        if not self.error_history:
            return {
                "total_errors": 0,
                "stages_affected": [],
                "error_types": [],
                "recovery_rate": 0.0,
            }
        
        stages = set(ctx.stage_name for ctx in self.error_history)
        error_types = set(type(ctx.error).__name__ for ctx in self.error_history)
        
        # Count successful recoveries (errors followed by success)
        # Simplified: if only one attempt, it's a failure
        successful_recoveries = sum(1 for ctx in self.error_history if ctx.attempt > 1)
        
        return {
            "total_errors": len(self.error_history),
            "stages_affected": list(stages),
            "error_types": list(error_types),
            "unique_errors": len(error_types),
            "successful_recoveries": successful_recoveries,
            "recovery_rate": successful_recoveries / len(self.error_history) if self.error_history else 0.0,
        }
    
    def _create_input_summary(self, args: tuple, kwargs: dict) -> str:
        """Create a summary of input for logging.
        
        Args:
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Summary string
        """
        summary_parts = []
        
        for arg in args:
            if isinstance(arg, str):
                summary_parts.append(f"str[{len(arg)}]")
            elif isinstance(arg, (list, dict)):
                summary_parts.append(f"{type(arg).__name__}[{len(arg)}]")
            else:
                summary_parts.append(type(arg).__name__)
        
        for key, value in kwargs.items():
            if isinstance(value, str):
                summary_parts.append(f"{key}=str[{len(value)}]")
            elif isinstance(value, (list, dict)):
                summary_parts.append(f"{key}={type(value).__name__}[{len(value)}]")
            else:
                summary_parts.append(f"{key}={type(value).__name__}")
        
        return "(" + ", ".join(summary_parts[:5]) + ")"


class GracefulDegradationHandler:
    """Handles graceful degradation when stages fail."""
    
    @staticmethod
    def create_empty_extraction_result():
        """Create an empty extraction result for failed extraction stage."""
        from bpmn_agent.models.extraction import ExtractionResultWithErrors, ExtractionMetadata
        from datetime import datetime
        
        metadata = ExtractionMetadata(
            input_text="",
            input_length=0,
            extraction_timestamp=datetime.now().isoformat(),
            extraction_duration_ms=0.0,
            llm_model="fallback",
            llm_temperature=0.3,
            stage="fallback",
            total_entities_extracted=0,
            high_confidence_entities=0,
            medium_confidence_entities=0,
            low_confidence_entities=0,
            total_relations_extracted=0,
            high_confidence_relations=0,
            co_reference_groups=0,
            warnings=["Extraction failed, returning empty result"],
        )
        
        return ExtractionResultWithErrors(
            entities=[],
            relations=[],
            co_references=[],
            metadata=metadata,
            errors=[],
        )
    
    @staticmethod
    def create_empty_graph_result():
        """Create an empty graph result for failed graph construction."""
        from bpmn_agent.models.graph import ProcessGraph, GraphNode
        from datetime import datetime
        
        # Create start and end nodes for minimal valid graph
        start_node = GraphNode(
            id="start",
            type="start",
            label="Start",
            bpmn_type="bpmn:StartEvent",
        )
        end_node = GraphNode(
            id="end",
            type="end",
            label="End",
            bpmn_type="bpmn:EndEvent",
        )
        
        return ProcessGraph(
            id="fallback-graph",
            name="Fallback Process",
            description="Minimal fallback process graph",
            nodes=[start_node, end_node],
            edges=[],
            is_acyclic=True,
            is_connected=False,
            has_implicit_parallelism=False,
            complexity=1.0,
            version="1.0",
            created_timestamp=datetime.now().isoformat(),
            metadata={"reason": "fallback"},
        )
    
    @staticmethod
    def create_minimal_xml_output():
        """Create minimal valid BPMN XML for failed generation."""
        return """<?xml version="1.0" encoding="UTF-8"?>
<definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL"
             xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL"
             targetNamespace="http://opencode.ai/bpmn/minimal">
    <process id="minimal_process" name="Minimal Process">
        <startEvent id="start_event"/>
        <endEvent id="end_event"/>
        <sequenceFlow sourceRef="start_event" targetRef="end_event"/>
    </process>
</definitions>"""


class CascadingFailureDetector:
    """Detects and prevents cascading failures."""
    
    def __init__(self, max_consecutive_failures: int = 2):
        """Initialize detector.
        
        Args:
            max_consecutive_failures: Maximum consecutive failures before stopping
        """
        self.max_consecutive_failures = max_consecutive_failures
        self.consecutive_failures = 0
        self.last_failed_stage: Optional[str] = None
    
    def record_failure(self, stage_name: str) -> bool:
        """Record a stage failure.
        
        Args:
            stage_name: Name of the failed stage
            
        Returns:
            True if cascading failure detected, False otherwise
        """
        if stage_name == self.last_failed_stage:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 1
            self.last_failed_stage = stage_name
        
        if self.consecutive_failures >= self.max_consecutive_failures:
            logger.error(
                f"Cascading failure detected: {stage_name} failed "
                f"{self.consecutive_failures} times"
            )
            return True
        
        return False
    
    def record_success(self) -> None:
        """Record a successful stage execution."""
        self.consecutive_failures = 0
        self.last_failed_stage = None
    
    def reset(self) -> None:
        """Reset detector state."""
        self.consecutive_failures = 0
        self.last_failed_stage = None
