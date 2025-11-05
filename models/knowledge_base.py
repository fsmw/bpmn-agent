"""
Knowledge Base Models and Data Structures

Defines the core data models for the BPMN knowledge base system, including:
- BPMN patterns (reusable process structures)
- Domain examples (annotated training data)
- Knowledge base container (pattern/example management)
- Context packages (curated knowledge for LLM augmentation)
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, ConfigDict, Field


class DomainType(str, Enum):
    """Supported business domains for pattern classification."""
    HR = "hr"
    FINANCE = "finance"
    IT = "it"
    HEALTHCARE = "healthcare"
    MANUFACTURING = "manufacturing"
    GENERIC = "generic"


class ComplexityLevel(str, Enum):
    """Process complexity classification."""
    SIMPLE = "simple"  # 1-5 elements
    MODERATE = "moderate"  # 5-15 elements
    COMPLEX = "complex"  # 15+ elements


class PatternCategory(str, Enum):
    """BPMN pattern types."""
    SEQUENTIAL = "sequential"  # Linear flow A -> B -> C
    PARALLEL = "parallel"  # Fork-join parallelism
    EXCLUSIVE_CHOICE = "exclusive_choice"  # XOR gateway
    INCLUSIVE_CHOICE = "inclusive_choice"  # OR gateway
    MULTI_INSTANCE = "multi_instance"  # Loop/iteration
    EVENT_DRIVEN = "event_driven"  # Event-based flow
    EXCEPTION_HANDLING = "exception_handling"  # Error paths
    SYNCHRONIZATION = "synchronization"  # Join patterns
    MESSAGE_PASSING = "message_passing"  # Inter-process communication
    DATA_FLOW = "data_flow"  # Data dependencies


class GraphStructure(BaseModel):
    """Represents the graph structure of a BPMN pattern."""
    
    nodes: List[str] = Field(
        ...,
        description="Node identifiers (gateways, activities, events)"
    )
    edges: List[str] = Field(
        ...,
        description="Edge descriptions (e.g., 'fork->task1', 'task1->join')"
    )
    node_types: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of node IDs to types (gateway, activity, event, etc.)"
    )
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "nodes": ["fork_gateway", "task1", "task2", "join_gateway"],
            "edges": ["fork->task1", "fork->task2", "task1->join", "task2->join"],
            "node_types": {
                "fork_gateway": "parallel_gateway",
                "task1": "activity",
                "task2": "activity",
                "join_gateway": "parallel_gateway"
            }
        }
    })


class BPMNPattern(BaseModel):
    """Represents a reusable BPMN process pattern."""
    
    id: str = Field(..., description="Unique pattern identifier")
    name: str = Field(..., description="Human-readable pattern name")
    description: str = Field(..., description="Detailed pattern description")
    
    # Classification
    domain: DomainType = Field(
        default=DomainType.GENERIC,
        description="Primary domain for this pattern"
    )
    category: PatternCategory = Field(
        ...,
        description="Pattern type/category"
    )
    complexity: ComplexityLevel = Field(
        default=ComplexityLevel.MODERATE,
        description="Complexity level of the pattern"
    )
    
    # Structure
    graph_structure: GraphStructure = Field(
        ...,
        description="Reference graph structure for the pattern"
    )
    
    # Knowledge Content
    examples: List[str] = Field(
        default_factory=list,
        description="Text examples where this pattern appears"
    )
    validation_rules: List[str] = Field(
        default_factory=list,
        description="Rules to validate pattern compliance"
    )
    anti_patterns: List[str] = Field(
        default_factory=list,
        description="Common mistakes or anti-patterns to avoid"
    )
    
    # Searchability and Metadata
    tags: Set[str] = Field(
        default_factory=set,
        description="Searchable keywords"
    )
    related_patterns: List[str] = Field(
        default_factory=list,
        description="IDs of related patterns"
    )
    
    # Quality and Tracking
    confidence: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Quality/confidence score (0-1)"
    )
    usage_count: int = Field(
        default=0,
        description="Number of times this pattern has been used"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Pattern creation timestamp"
    )
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp"
    )
    version: str = Field(default="1.0", description="Pattern version")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "id": "pattern_parallel_001",
            "name": "Parallel Fork-Join",
            "description": "Multiple independent tasks execute in parallel, then rejoin",
            "domain": "generic",
            "category": "parallel",
            "complexity": "moderate",
            "graph_structure": {
                "nodes": ["fork", "task1", "task2", "join"],
                "edges": ["fork->task1", "fork->task2", "task1->join", "task2->join"],
                "node_types": {
                    "fork": "parallel_gateway",
                    "task1": "activity",
                    "task2": "activity",
                    "join": "parallel_gateway"
                }
            },
            "examples": [
                "Process payment and send confirmation in parallel",
                "Validate and archive documents concurrently"
            ],
            "tags": ["parallelism", "fork", "join", "gateway"]
        }
    })


class DomainExample(BaseModel):
    """Real-world process example for few-shot learning."""
    
    id: str = Field(..., description="Unique example identifier")
    text: str = Field(..., description="Natural language process description")
    
    # Classification
    domain: DomainType = Field(
        default=DomainType.GENERIC,
        description="Domain category"
    )
    complexity: ComplexityLevel = Field(
        default=ComplexityLevel.MODERATE,
        description="Complexity level"
    )
    
    # Pattern Information
    patterns_used: List[str] = Field(
        default_factory=list,
        description="IDs of patterns present in this example"
    )
    
    # Expected Extraction
    entities_expected: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Expected entities by type (activities, events, gateways, actors, etc.)"
    )
    relations_expected: Dict[str, Any] = Field(
        default_factory=dict,
        description="Expected relations in the process"
    )
    
    # Quality Metrics
    difficulty: str = Field(
        default="medium",
        description="Example difficulty: easy, medium, hard"
    )
    validation_score: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Quality score for this example (0-1)"
    )
    
    # Metadata
    bpmn_structure: Optional[Dict[str, Any]] = Field(
        None,
        description="Expected BPMN XML structure (optional)"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )
    usage_count: int = Field(
        default=0,
        description="Number of times used in training/inference"
    )
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "id": "example_finance_001",
            "text": "A customer submits an order. The system validates the order and processes payment in parallel. Once both complete, an invoice is generated and sent.",
            "domain": "finance",
            "complexity": "moderate",
            "patterns_used": ["parallel_fork_join", "sequential"],
            "difficulty": "medium",
            "entities_expected": {
                "activities": [
                    "Submit Order",
                    "Validate Order",
                    "Process Payment",
                    "Generate Invoice",
                    "Send Invoice"
                ],
                "events": ["Order Submitted", "Both Tasks Complete"],
                "gateways": ["Parallel Fork", "Parallel Join"]
            }
        }
    })


class ContextPackage(BaseModel):
    """Curated knowledge selection for LLM augmentation."""
    
    # Selected Knowledge Elements
    selected_patterns: List[BPMNPattern] = Field(
        default_factory=list,
        description="Top N relevant patterns"
    )
    selected_examples: List[DomainExample] = Field(
        default_factory=list,
        description="3-5 few-shot examples"
    )
    domain_terms: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Domain-specific terminology by category"
    )
    validation_rules: List[str] = Field(
        default_factory=list,
        description="Relevant validation rules for the domain"
    )
    
    # Metadata
    detected_domain: DomainType = Field(
        default=DomainType.GENERIC,
        description="Detected primary domain"
    )
    detected_complexity: ComplexityLevel = Field(
        default=ComplexityLevel.MODERATE,
        description="Detected complexity level"
    )
    recognized_patterns: List[str] = Field(
        default_factory=list,
        description="IDs of patterns recognized in input text"
    )
    
    # Quality and Optimization Metrics
    token_count: int = Field(
        default=0,
        description="Estimated token count for this context"
    )
    max_tokens: int = Field(
        default=4000,
        description="Maximum tokens available for context"
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Overall confidence (0-1)"
    )
    
    # Optimization Details
    reasoning: str = Field(
        default="",
        description="Explanation of why these elements were selected"
    )
    optimization_level: str = Field(
        default="balanced",
        description="Optimization strategy: minimal, balanced, comprehensive"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Context package creation time"
    )
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class KnowledgeBaseMetadata(BaseModel):
    """Metadata about the knowledge base."""
    
    version: str = Field(default="1.0", description="KB version")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="KB creation date"
    )
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update date"
    )
    total_patterns: int = Field(default=0, description="Total patterns in KB")
    total_examples: int = Field(default=0, description="Total examples in KB")
    domains_covered: Set[DomainType] = Field(
        default_factory=set,
        description="Domains with patterns"
    )
    description: str = Field(
        default="",
        description="KB description"
    )


class KnowledgeBase(BaseModel):
    """Container for all knowledge base content."""
    
    # Core Content
    patterns: Dict[str, BPMNPattern] = Field(
        default_factory=dict,
        description="All patterns indexed by ID"
    )
    examples: Dict[str, DomainExample] = Field(
        default_factory=dict,
        description="All examples indexed by ID"
    )
    
    # Domain Vocabularies
    domain_vocabularies: Dict[DomainType, List[str]] = Field(
        default_factory=dict,
        description="Domain-specific terminology"
    )
    domain_keywords: Dict[DomainType, Dict[str, float]] = Field(
        default_factory=dict,
        description="Domain keywords with relevance scores"
    )
    
    # Validation and Quality
    anti_patterns: List[str] = Field(
        default_factory=list,
        description="Known anti-patterns to flag"
    )
    validation_templates: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Validation rule templates by pattern type"
    )
    
    # Metadata
    metadata: KnowledgeBaseMetadata = Field(
        default_factory=KnowledgeBaseMetadata,
        description="KB metadata"
    )
    
    # Vector Store Reference
    vector_store_ready: bool = Field(
        default=False,
        description="Whether vector embeddings have been built"
    )
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def get_patterns_by_domain(self, domain: DomainType) -> Dict[str, BPMNPattern]:
        """Get all patterns for a specific domain."""
        return {
            pid: pattern
            for pid, pattern in self.patterns.items()
            if pattern.domain == domain or pattern.domain == DomainType.GENERIC
        }
    
    def get_patterns_by_category(self, category: PatternCategory) -> Dict[str, BPMNPattern]:
        """Get all patterns of a specific category."""
        return {
            pid: pattern
            for pid, pattern in self.patterns.items()
            if pattern.category == category
        }
    
    def get_examples_by_domain(self, domain: DomainType) -> Dict[str, DomainExample]:
        """Get all examples for a specific domain."""
        return {
            eid: example
            for eid, example in self.examples.items()
            if example.domain == domain or example.domain == DomainType.GENERIC
        }
    
    def add_pattern(self, pattern: BPMNPattern) -> None:
        """Add a pattern to the knowledge base."""
        self.patterns[pattern.id] = pattern
        self.metadata.total_patterns = len(self.patterns)
        if pattern.domain not in self.metadata.domains_covered:
            self.metadata.domains_covered.add(pattern.domain)
        self.metadata.last_updated = datetime.now(timezone.utc)
    
    def add_example(self, example: DomainExample) -> None:
        """Add an example to the knowledge base."""
        self.examples[example.id] = example
        self.metadata.total_examples = len(self.examples)
        self.metadata.last_updated = datetime.now(timezone.utc)
    
    def get_pattern(self, pattern_id: str) -> Optional[BPMNPattern]:
        """Retrieve a specific pattern by ID."""
        return self.patterns.get(pattern_id)
    
    def get_example(self, example_id: str) -> Optional[DomainExample]:
        """Retrieve a specific example by ID."""
        return self.examples.get(example_id)
    
    def get_related_patterns(self, pattern_id: str) -> List[BPMNPattern]:
        """Get patterns related to a given pattern."""
        pattern = self.get_pattern(pattern_id)
        if not pattern:
            return []
        
        related = []
        for rel_id in pattern.related_patterns:
            rel_pattern = self.get_pattern(rel_id)
            if rel_pattern:
                related.append(rel_pattern)
        return related
    
    def update_pattern_usage(self, pattern_id: str, count: int = 1) -> None:
        """Update usage count for a pattern."""
        if pattern_id in self.patterns:
            self.patterns[pattern_id].usage_count += count
            self.patterns[pattern_id].last_updated = datetime.now(timezone.utc)
    
    def update_example_usage(self, example_id: str, count: int = 1) -> None:
        """Update usage count for an example."""
        if example_id in self.examples:
            self.examples[example_id].usage_count += count
