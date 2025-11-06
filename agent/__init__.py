"""
BPMN Agent Framework

Main package for the BPMN Agent, providing a unified orchestrator
that coordinates all pipeline stages with optional knowledge base integration.
"""

from bpmn_agent.agent.config import (
    AgentConfig,
    ErrorHandlingStrategy,
    PipelineConfig,
    ProcessingMode,
)
from bpmn_agent.agent.orchestrator import BPMNAgent
from bpmn_agent.agent.state import (
    AgentState,
    ProcessingMetrics,
    StageResult,
    StageStatus,
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
]
