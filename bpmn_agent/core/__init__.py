"""
Core infrastructure module for BPMN Agent.

Provides LLM abstraction, logging, observability, and configuration management.
"""

from .llm_client import (
    BaseLLMClient,
    LLMClientFactory,
    LLMConfig,
    LLMMessage,
    LLMProviderType,
    LLMResponse,
    OllamaClient,
    OpenAICompatibleClient,
    call_llm,
    stream_llm,
    validate_llm_config,
)
from .observability import (
    LogLevel,
    ObservabilityConfig,
    ObservabilityManager,
    Timer,
    log_execution,
    record_metric,
    span,
)
from .tokenizer import (
    ModelTokenizer,
    TokenCounter,
    count_tokens,
    estimate_cost,
)

__all__ = [
    # LLM
    "BaseLLMClient",
    "LLMClientFactory",
    "LLMConfig",
    "LLMMessage",
    "LLMProviderType",
    "LLMResponse",
    "OllamaClient",
    "OpenAICompatibleClient",
    "call_llm",
    "stream_llm",
    "validate_llm_config",
    # Observability
    "LogLevel",
    "ObservabilityConfig",
    "ObservabilityManager",
    "Timer",
    "log_execution",
    "record_metric",
    "span",
    # Tokenization
    "ModelTokenizer",
    "TokenCounter",
    "count_tokens",
    "estimate_cost",
]
