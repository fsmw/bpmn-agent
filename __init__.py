"""
BPMN Agent: Transform Natural Language to BPMN 2.0 Diagrams

An AI-powered agent that transforms natural language process descriptions
into valid BPMN 2.0 XML diagrams using LLMs (Ollama, OpenAI) and
optional knowledge base pattern matching.
"""

# Core components
from bpmn_agent.core.llm_client import LLMClientFactory, LLMConfig, LLMMessage
from bpmn_agent.core.observability import ObservabilityManager, ObservabilityConfig

# Agent framework
from bpmn_agent.agent import (
    AgentConfig,
    AgentState,
    BPMNAgent,
    ErrorHandlingStrategy,
    PipelineConfig,
    ProcessingMode,
    ProcessingMetrics,
    StageResult,
    StageStatus,
)

# Pipeline stages
from bpmn_agent.stages import (
    CoReferenceResolver,
    EntityExtractor,
    EntityResolutionPipeline,
    ExtractionPrompt,
    JSONParser,
    PreprocessedText,
    ProcessGraphBuilder,
    SemanticGraphConstructionPipeline,
    TextChunk,
    TextPreprocessor,
)

# Models
from bpmn_agent.models import (
    EndEvent,
    ExclusiveGateway,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    ExtractionResultWithErrors,
    ParallelGateway,
    Process,
    ProcessGraph,
    SequenceFlow,
    StartEvent,
    Task,
    UserTask,
)

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Core
    "LLMClientFactory",
    "LLMConfig",
    "LLMMessage",
    "ObservabilityManager",
    "ObservabilityConfig",
    # Agent
    "BPMNAgent",
    "AgentConfig",
    "PipelineConfig",
    "ProcessingMode",
    "ErrorHandlingStrategy",
    "AgentState",
    "StageResult",
    "StageStatus",
    "ProcessingMetrics",
    # Stages
    "TextPreprocessor",
    "TextChunk",
    "PreprocessedText",
    "EntityExtractor",
    "JSONParser",
    "ExtractionPrompt",
    "EntityResolutionPipeline",
    "CoReferenceResolver",
    "ProcessGraphBuilder",
    "SemanticGraphConstructionPipeline",
    # Models
    "Task",
    "UserTask",
    "StartEvent",
    "EndEvent",
    "ExclusiveGateway",
    "ParallelGateway",
    "SequenceFlow",
    "Process",
    "ProcessGraph",
    "ExtractedEntity",
    "ExtractedRelation",
    "ExtractionResult",
    "ExtractionResultWithErrors",
]
