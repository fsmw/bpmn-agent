"""
Agent State Management

Tracks the state of processing through the pipeline stages,
including intermediate results and metrics.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime


class StageStatus(str, Enum):
    """Status of a pipeline stage."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult:
    """Result from a single pipeline stage."""
    
    stage_name: str
    status: StageStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    result: Any = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_success(self) -> bool:
        """Check if stage succeeded."""
        return self.status == StageStatus.COMPLETED
    
    @property
    def has_errors(self) -> bool:
        """Check if stage has errors."""
        return self.error is not None


@dataclass
class ProcessingMetrics:
    """Overall processing metrics."""
    
    total_stages: int = 5
    completed_stages: int = 0
    failed_stages: int = 0
    total_duration_ms: float = 0.0
    
    # Input metrics
    input_text_length: int = 0
    input_token_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    # Extraction metrics
    entities_extracted: int = 0
    relations_extracted: int = 0
    avg_entity_confidence: float = 0.0
    
    # Resolution metrics
    coreferences_resolved: int = 0
    actors_consolidated: int = 0
    
    # Graph metrics
    graph_nodes: int = 0
    graph_edges: int = 0
    graph_cycles_detected: int = 0
    
    # Output metrics
    output_xml_size_bytes: int = 0
    output_element_count: int = 0
    
    @property
    def completion_rate(self) -> float:
        """Get completion rate (0-1)."""
        if self.total_stages == 0:
            return 0.0
        return self.completed_stages / self.total_stages
    
    @property
    def success_rate(self) -> float:
        """Get success rate (0-1)."""
        failed_or_completed = self.failed_stages + self.completed_stages
        if failed_or_completed == 0:
            return 0.0
        return self.completed_stages / failed_or_completed


@dataclass
class AgentState:
    """Complete state of agent processing."""
    
    # Session info
    session_id: str
    start_time: datetime
    
    # Input
    input_text: str = ""
    input_domain: Optional[str] = None
    preprocessed_text: Optional[str] = None
    
    # Stage results
    stage_results: List[StageResult] = field(default_factory=list)
    
    # Overall metrics
    metrics: ProcessingMetrics = field(default_factory=ProcessingMetrics)
    
    # Final result
    output_xml: Optional[str] = None
    output_path: Optional[str] = None
    
    # Errors and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def is_complete(self) -> bool:
        """Check if processing is complete."""
        return self.output_xml is not None
    
    @property
    def completion_rate(self) -> float:
        """Get completion rate (0-1)."""
        return self.metrics.completion_rate
    
    @property
    def is_failed(self) -> bool:
        """Check if processing failed."""
        return len(self.errors) > 0 or any(
            r.status == StageStatus.FAILED for r in self.stage_results
        )
    
    @property
    def current_stage(self) -> Optional[str]:
        """Get name of currently running stage."""
        for result in self.stage_results:
            if result.status == StageStatus.RUNNING:
                return result.stage_name
        return None
    
    def add_stage_result(self, result: StageResult) -> None:
        """Add stage result to state.
        
        Args:
            result: Stage result to add
        """
        self.stage_results.append(result)
        
        if result.status == StageStatus.COMPLETED:
            self.metrics.completed_stages += 1
        elif result.status == StageStatus.FAILED:
            self.metrics.failed_stages += 1
        
        if result.error:
            self.errors.append(f"{result.stage_name}: {result.error}")
        
        self.warnings.extend(result.warnings)
    
    def get_stage_result(self, stage_name: str) -> Optional[StageResult]:
        """Get result for a specific stage.
        
        Args:
            stage_name: Name of the stage
            
        Returns:
            StageResult if found, None otherwise
        """
        for result in self.stage_results:
            if result.stage_name == stage_name:
                return result
        return None
    
    def summary(self) -> Dict[str, Any]:
        """Get summary of current state.
        
        Returns:
            Dictionary with state summary
        """
        return {
            "session_id": self.session_id,
            "input_domain": self.input_domain,
            "is_complete": self.is_complete,
            "is_failed": self.is_failed,
            "current_stage": self.current_stage,
            "stages_completed": self.metrics.completed_stages,
            "stages_total": self.metrics.total_stages,
            "completion_rate": self.metrics.completion_rate,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "metrics": self.metrics.__dict__,
        }
