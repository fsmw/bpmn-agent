"""
Agent Configuration Schema

Defines configuration for the BPMNAgent including LLM settings,
pipeline options, and feature flags.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from bpmn_agent.core.llm_client import LLMConfig


class ProcessingMode(str, Enum):
    """Processing mode for the agent."""
    
    STANDARD = "standard"  # Normal end-to-end processing
    KB_ENHANCED = "kb_enhanced"  # Enhanced with knowledge base patterns
    VALIDATION_ONLY = "validation_only"  # Only validate, no generation
    ANALYSIS_ONLY = "analysis_only"  # Only analyze, return intermediate results


class ErrorHandlingStrategy(str, Enum):
    """Error handling strategy for pipeline stages."""
    
    STRICT = "strict"  # Stop on first error
    LENIENT = "lenient"  # Continue with warnings
    RECOVERY = "recovery"  # Attempt recovery strategies


@dataclass
class PipelineConfig:
    """Configuration for pipeline stages."""
    
    # Stage 1: Text Preprocessing
    chunk_size: int = 512
    chunk_overlap: int = 50
    normalize_whitespace: bool = True
    remove_special_chars: bool = False
    lowercase: bool = False
    
    # Stage 2: Entity Extraction
    extraction_retries: int = 2
    extraction_timeout: int = 120
    kb_augmented_prompts: bool = True
    
    # Stage 3: Entity Resolution
    coreference_threshold: float = 0.75
    actor_consolidation_enabled: bool = True
    
    # Stage 4: Graph Construction
    implicit_flow_inference: bool = True
    lane_structure_detection: bool = True
    
    # Stage 5: XML Generation
    include_diagram_interchange: bool = True
    validate_xsd: bool = False
    preserve_kb_metadata: bool = True
    
    # Stage 6: Phase 4 Validation
    enable_phase4_validation: bool = True
    enable_rag_validation: bool = True  # Requires enable_kb=True
    validation_fail_on_error: bool = False  # If True, fail pipeline on validation errors


@dataclass
class AgentConfig:
    """Complete agent configuration."""
    
    # LLM Configuration
    llm_config: LLMConfig
    
    # Processing Mode
    mode: ProcessingMode = ProcessingMode.STANDARD
    
    # Pipeline Configuration
    pipeline_config: PipelineConfig = field(default_factory=PipelineConfig)
    
    # Error Handling
    error_handling: ErrorHandlingStrategy = ErrorHandlingStrategy.RECOVERY
    
    # Knowledge Base
    enable_kb: bool = True
    kb_domain_auto_detect: bool = True
    
    # Observability
    enable_logging: bool = True
    enable_metrics: bool = True
    enable_tracing: bool = False
    
    # Behavior
    verbose: bool = False
    
    @classmethod
    def from_env(cls, mode: ProcessingMode = ProcessingMode.STANDARD) -> "AgentConfig":
        """Create agent config from environment variables.
        
        Args:
            mode: Processing mode for the agent
            
        Returns:
            AgentConfig instance
        """
        llm_config = LLMConfig.from_env()
        return cls(llm_config=llm_config, mode=mode)
    
    @classmethod
    def from_env_with_mode(cls, mode: str) -> "AgentConfig":
        """Create agent config from environment with mode string.
        
        Args:
            mode: Processing mode as string
            
        Returns:
            AgentConfig instance
        """
        try:
            processing_mode = ProcessingMode(mode)
        except ValueError:
            processing_mode = ProcessingMode.STANDARD
        
        return cls.from_env(processing_mode)
