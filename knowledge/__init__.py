"""
Knowledge Base Module

Provides domain-aware context, pattern recognition, and vector search
for BPMN process extraction and generation.

Components:
- vector_store: Semantic similarity search for patterns and examples
- domain_classifier: Domain detection and complexity analysis
- context_manager: Intelligent context selection for LLM augmentation
- loader: Pattern library loading and management
"""

from .context_manager import ContextOptimizer, ContextSelector, TokenCounter
from .domain_classifier import ComplexityAnalyzer, DomainClassifier, PatternRecognizer
from .loader import PatternLibraryLoader
from .vector_store import EmbeddingService, VectorSearchResult, VectorStore

__all__ = [
    # Vector Store
    "EmbeddingService",
    "VectorSearchResult",
    "VectorStore",
    # Domain Classification
    "DomainClassifier",
    "ComplexityAnalyzer",
    "PatternRecognizer",
    # Context Management
    "TokenCounter",
    "ContextSelector",
    "ContextOptimizer",
    # Pattern Loading
    "PatternLibraryLoader",
]
