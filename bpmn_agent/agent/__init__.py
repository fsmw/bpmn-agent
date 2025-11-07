"""
BPMN Agent Framework

Main package for the BPMN Agent, providing a unified orchestrator
that coordinates all pipeline stages with optional knowledge base integration.
"""

from bpmn_agent.agent.checkpoint import (
    CheckpointManager,
    CheckpointType,
    ResumableAgent,
)
from bpmn_agent.agent.config import (
    AgentConfig,
    ErrorHandlingStrategy,
    PipelineConfig,
    ProcessingMode,
)
from bpmn_agent.agent.error_handler import (
    CascadingFailureDetector,
    ErrorRecoveryEngine,
    GracefulDegradationHandler,
)
from bpmn_agent.agent.observability_hooks import (
    ObservabilityHooks,
    PipelineMetrics,
    StageMetrics,
    get_observability_hooks,
    initialize_observability_hooks,
)
from bpmn_agent.agent.orchestrator import BPMNAgent
from bpmn_agent.agent.state import (
    AgentState,
    ProcessingMetrics,
    StageResult,
    StageStatus,
)
from bpmn_agent.agent.tools import (
    GraphAnalyzer,
    ProcessRefinementTools,
    XMLValidator,
)

__all__ = [
    # Configuration
    "AgentConfig",
    "PipelineConfig",
    "ProcessingMode",
    "ErrorHandlingStrategy",
    # Orchestrator
    "BPMNAgent",
    # State Management
    "AgentState",
    "StageResult",
    "StageStatus",
    "ProcessingMetrics",
    # Error Handling
    "ErrorRecoveryEngine",
    "CascadingFailureDetector",
    "GracefulDegradationHandler",
    # Checkpointing
    "CheckpointManager",
    "CheckpointType",
    "ResumableAgent",
    # Tools
    "GraphAnalyzer",
    "XMLValidator",
    "ProcessRefinementTools",
    # Observability
    "ObservabilityHooks",
    "PipelineMetrics",
    "StageMetrics",
    "get_observability_hooks",
    "initialize_observability_hooks",
]
