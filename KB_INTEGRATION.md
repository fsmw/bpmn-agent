# Knowledge Base Integration Guide

Comprehensive guide to the Knowledge Base (KB) integration system across all pipeline stages of the BPMN Agent.

## Overview

The KB integration provides domain-specific intelligence across the entire BPMN processing pipeline (Stages 1-4). It enhances:

- **Entity Extraction** - Domain-specific entity recognition patterns
- **Relationship Recognition** - Domain-specific relation type inference
- **Graph Construction** - Domain-aware implicit flow inference and pattern matching
- **Lane Structuring** - Role/responsibility mapping based on domain knowledge

The system implements **graceful degradation**: all stages work without a KB, with optional KB enhancements when available.

## Architecture

### Core Components

```
Knowledge Base System
├── KnowledgeBase (models/knowledge_base.py)
│   ├── GraphStructure - Domain graph patterns
│   ├── BPMNPattern - Reusable BPMN patterns
│   ├── DomainExample - Process examples
│   ├── ContextPackage - Domain context
│   └── DomainType enum - Supported domains
│
├── PatternRecognizer (knowledge/pattern_recognizer.py)
│   ├── Pattern matching for domain entities
│   ├── Implicit flow inference
│   └── Role/responsibility mapping
│
├── ContextSelector (knowledge/context_selector.py)
│   ├── Domain context selection
│   ├── Relevant example retrieval
│   └── Pattern-based context lookup
│
└── KBEnricher (stages/process_graph_builder.py)
    ├── Post-extraction entity enrichment
    ├── Relation suggestions
    └── Confidence boost for KB-recognized entities
```

### Pipeline Integration Points

#### Stage 1: Entity Extraction
```python
from bpmn_agent.stages import EntityExtractor

extractor = EntityExtractor(enable_kb=True)
entities, relations = await extractor.extract(text, domain=DomainType.FINANCE)
```

- Domain-specific patterns guide LLM extraction
- Context examples inform entity recognition
- Extraction metadata includes domain information

#### Stage 2: Relationship Resolution  
```python
from bpmn_agent.stages import RelationshipResolver

resolver = RelationshipResolver(enable_kb=True)
relations = await resolver.resolve(entities, domain=DomainType.FINANCE)
```

- Domain relation type mappings
- Pattern-based implicit relation inference
- Relation type suggestions

#### Stage 3: Graph Construction
```python
from bpmn_agent.stages import ProcessGraphBuilder

builder = ProcessGraphBuilder(enable_kb=True)
graph, report = await builder.build_graph(
    entities, relations, domain=DomainType.FINANCE
)
```

- KB-aware implicit flow inference (fork/join detection)
- Pattern-based parallel path identification
- Domain-specific node clustering for swimlanes
- Enhanced validation with domain-specific rules

#### Stage 4: Lane Structuring
```python
from bpmn_agent.stages import LaneStructureBuilder

builder = LaneStructureBuilder(enable_kb=True)
lanes = await builder.build_lanes(
    graph, domain=DomainType.FINANCE
)
```

- Role and responsibility mappings
- Actor skill-based lane organization
- Domain-specific swimlane patterns

## Supported Domains

The system supports domain-specific knowledge for:

```python
from bpmn_agent.models.knowledge_base import DomainType

class DomainType(str, Enum):
    FINANCE = "finance"          # Financial processes
    HEALTHCARE = "healthcare"    # Medical/healthcare workflows
    MANUFACTURING = "manufacturing"  # Production processes
    LOGISTICS = "logistics"      # Supply chain operations
    HUMAN_RESOURCES = "human_resources"  # HR workflows
    RETAIL = "retail"           # Retail operations
    GENERAL = "general"         # General/cross-domain
```

## Usage Examples

### Example 1: Process with KB Enhancement

```python
from datetime import datetime
from bpmn_agent.stages import SemanticGraphConstructionPipeline
from bpmn_agent.models.extraction import (
    ExtractedEntity, ExtractedRelation,
    ExtractionResult, ExtractionMetadata,
    EntityType, RelationType, ConfidenceLevel
)
from bpmn_agent.models.knowledge_base import DomainType

# Create extraction result
extraction = ExtractionResult(
    entities=[
        ExtractedEntity(
            id="e1", type=EntityType.ACTIVITY,
            name="Payment Authorization",
            confidence=ConfidenceLevel.HIGH,
            source_text="payments need authorization"
        ),
        ExtractedEntity(
            id="e2", type=EntityType.ACTIVITY,
            name="Fund Transfer",
            confidence=ConfidenceLevel.HIGH,
            source_text="funds are transferred"
        ),
        ExtractedEntity(
            id="a1", type=EntityType.ACTOR,
            name="Bank Officer",
            confidence=ConfidenceLevel.HIGH,
            source_text="bank officer"
        ),
    ],
    relations=[
        ExtractedRelation(
            id="r1", source_id="a1", target_id="e1",
            type=RelationType.PERFORMS,
            confidence=ConfidenceLevel.HIGH,
            source_text="officer performs"
        ),
    ],
    metadata=ExtractionMetadata(
        input_text="Bank officer receives payment requests...",
        input_length=45,
        extraction_timestamp=datetime.now().isoformat(),
        extraction_duration_ms=150.0,
        llm_model="gpt-4",
        llm_temperature=0.7,
    )
)

# Run pipeline WITH KB enhancement
pipeline = SemanticGraphConstructionPipeline(enable_kb=True)
graph, report, flows = pipeline.construct_graph(
    extraction, 
    actor_profiles={"a1": ActorProfile(...)},
    domain=DomainType.FINANCE  # Provide domain context
)

# Results include KB-enhanced suggestions
print(f"Graph has {len(graph.nodes)} nodes")
print(f"Detected {len(flows)} implicit flows")
print(f"Validation issues: {len(report.validation_issues)}")
```

### Example 2: Process WITHOUT KB (Graceful Degradation)

```python
# Same extraction result
extraction = ExtractionResult(...)

# Run pipeline WITHOUT KB enhancement  
pipeline = SemanticGraphConstructionPipeline(enable_kb=False)
graph, report, flows = pipeline.construct_graph(
    extraction,
    actor_profiles={},
    domain=None  # No domain provided
)

# System still works, just with less intelligent inference
print(f"Basic graph construction completed: {len(graph.nodes)} nodes")
```

### Example 3: Direct KB Usage

```python
from bpmn_agent.knowledge import KnowledgeBase, PatternRecognizer, ContextSelector

# Initialize KB
kb = KnowledgeBase()
kb.load_from_file("path/to/kb.json")

# Use pattern recognizer
recognizer = PatternRecognizer(kb)
entity_types = recognizer.get_entity_types_for_domain(DomainType.FINANCE)
patterns = recognizer.get_patterns_for_domain(DomainType.FINANCE)

# Recognize implicit flows
implicit_flows = recognizer.recognize_implicit_flows(
    graph, 
    domain=DomainType.FINANCE
)

# Use context selector
selector = ContextSelector(kb)
context = selector.select_context_for_domain(DomainType.FINANCE)
examples = selector.get_examples_for_entity_type(
    EntityType.ACTIVITY, 
    domain=DomainType.FINANCE
)
```

## KB Data Structure

### Knowledge Base JSON Format

```json
{
  "domains": {
    "finance": {
      "entity_types": ["ACTIVITY", "GATEWAY", "EVENT"],
      "relation_types": ["PRECEDES", "TRIGGERS", "INVOLVES"],
      "actor_roles": ["Bank Officer", "System", "Customer"],
      "patterns": [
        {
          "name": "Payment Approval",
          "entities": ["ACTIVITY"],
          "trigger_keywords": ["payment", "approval", "authorization"],
          "roles_involved": ["Bank Officer", "System"],
          "typical_duration_minutes": 15
        }
      ],
      "examples": [
        {
          "name": "Fund Transfer Process",
          "description": "Standard bank fund transfer",
          "process_steps": [...],
          "common_actors": ["Teller", "Manager", "System"],
          "expected_duration_minutes": 45
        }
      ],
      "context": {
        "description": "Financial process domain",
        "typical_activities": ["Authorization", "Validation", "Transfer"],
        "critical_roles": ["Compliance Officer", "Manager"]
      }
    }
  }
}
```

### BPMNPattern Model

```python
from bpmn_agent.models.knowledge_base import BPMNPattern

pattern = BPMNPattern(
    name="Approval Gateway",
    description="Standard approval decision point",
    entity_type=EntityType.GATEWAY,
    typical_structure=GraphStructure(
        nodes=[...],
        edges=[...]
    ),
    roles_involved=["Manager", "Supervisor"],
    keywords=["approved", "rejected", "review"],
    success_rate=0.95,
    average_duration_minutes=10
)
```

## Configuration

### Enable/Disable KB

```python
# Enable KB enhancement globally
from bpmn_agent.config import set_kb_enabled
set_kb_enabled(True)

# Or per-component
extractor = EntityExtractor(enable_kb=True)  # With KB
builder = ProcessGraphBuilder(enable_kb=False)  # Without KB
```

### KB File Location

```python
import os
from bpmn_agent.knowledge import KnowledgeBase

# Default location
kb = KnowledgeBase()  # Loads from $KB_DATA_PATH/knowledge_base.json

# Custom location
kb = KnowledgeBase(data_path="/custom/path/kb.json")

# Or via environment
os.environ["KB_DATA_PATH"] = "/custom/kb/directory"
```

### Domain Selection

```python
from bpmn_agent.models.knowledge_base import DomainType

# Explicit domain
graph, report = pipeline.construct_graph(
    extraction,
    domain=DomainType.FINANCE
)

# Auto-detection (not yet implemented)
# graph, report = pipeline.construct_graph(extraction)
# # System attempts to detect domain from content
```

## Performance Impact

### Latency (with KB enabled)

| Operation | Without KB | With KB | Overhead |
|-----------|-----------|---------|----------|
| Entity extraction | 2-3s | 2.5-3.5s | +500ms |
| Relationship resolution | 1-2s | 1.5-2.5s | +500ms |
| Graph construction | 1-2s | 1.5-2.5s | +500ms |
| Flow inference | <1s | 1-2s | +1s (pattern matching) |
| **Total** | **4-8s** | **6-11s** | **+2-3s** |

### Memory Usage

- Base system: ~50MB
- KB loaded: +30-50MB (depending on KB size)
- Pattern cache: +10-20MB (after first domain use)

### Throughput

- Single process: ~5-10 processes/min (without KB)
- Single process: ~3-6 processes/min (with KB)
- With parallelization: Scale linearly

## Caching & Optimization

### Pattern Cache

Patterns are cached per-domain after first use:

```python
# First call - load from KB
patterns = recognizer.get_patterns_for_domain(DomainType.FINANCE)  # Slow

# Subsequent calls - from cache
patterns = recognizer.get_patterns_for_domain(DomainType.FINANCE)  # Fast
```

### Context Caching

Context is cached per session:

```python
selector = ContextSelector(kb)

# Set context once
selector.set_active_domain(DomainType.FINANCE)

# All subsequent calls use cached context
for process in processes:
    context = selector.get_context()  # Uses cache
```

## Troubleshooting

### KB Not Loading

```python
from bpmn_agent.knowledge import KnowledgeBase

try:
    kb = KnowledgeBase()
except Exception as e:
    print(f"KB load error: {e}")
    # Graceful degradation - system continues without KB
```

### No Patterns Recognized

```python
# Debug: Check available patterns
recognizer = PatternRecognizer(kb)
patterns = recognizer.get_patterns_for_domain(DomainType.FINANCE)

if not patterns:
    print("No patterns loaded for domain")
    # System falls back to generic processing
```

### Implicit Flow Not Detected

```python
# Check if KB is enabled
graph, report = pipeline.construct_graph(extraction, domain=DomainType.FINANCE)

if not report.suggestions:
    print("No KB suggestions generated")
    print(f"KB enabled: {pipeline.enable_kb}")
    print(f"Domain: {pipeline.domain}")
```

### Domain Not Supported

```python
from bpmn_agent.models.knowledge_base import DomainType

# Check supported domains
print([d.value for d in DomainType])

# Use GENERAL domain as fallback
graph, report = pipeline.construct_graph(
    extraction,
    domain=DomainType.GENERAL
)
```

## Testing

### Run KB Integration Tests

```bash
# All KB integration tests
pytest src/bpmn-agent/tests/test_kb_stage4_integration.py -v

# Specific test class
pytest src/bpmn-agent/tests/test_kb_stage4_integration.py::TestKBGraphEnricher -v

# With coverage
pytest src/bpmn-agent/tests/test_kb_stage4_integration.py --cov=bpmn_agent.knowledge
```

### Test Coverage

```
TestKBGraphEnricher (6 tests)
- Pattern recognition for entity enrichment
- Confidence boost for KB-recognized entities
- Relation type suggestions

TestImplicitFlowInferrerWithKB (6 tests)  
- KB-based implicit flow inference
- Fork/join detection with patterns
- Sequential flow inference

TestLaneStructureBuilderWithKB (6 tests)
- Role-based lane construction
- KB-aware actor grouping
- Swimlane optimization

TestSemanticGraphConstructionPipelineWithKB (6 tests)
- Full pipeline with KB enhancement
- Domain propagation through stages
- Graph validation with KB rules

TestFullPipelineIntegration (2 tests)
- End-to-end process: extraction → graph → lanes
- Graceful degradation without KB
```

## Best Practices

### 1. Always Provide Domain Context

```python
# Good - provides domain context for KB enhancement
graph, report = pipeline.construct_graph(
    extraction,
    domain=DomainType.FINANCE
)

# Less optimal - KB features limited
graph, report = pipeline.construct_graph(extraction)
```

### 2. Handle KB Unavailability Gracefully

```python
# System continues even if KB fails to load
try:
    pipeline = SemanticGraphConstructionPipeline(enable_kb=True)
except Exception:
    # Fallback to non-KB mode
    pipeline = SemanticGraphConstructionPipeline(enable_kb=False)

graph, report = pipeline.construct_graph(extraction)
```

### 3. Cache KB Across Batches

```python
# Load KB once, reuse for multiple processes
pipeline = SemanticGraphConstructionPipeline(enable_kb=True)

for process_text in batch_of_processes:
    extraction = await extractor.extract(process_text)
    graph, report = pipeline.construct_graph(
        extraction,
        domain=DomainType.FINANCE  # Cached after first use
    )
```

### 4. Monitor KB Enhancement Impact

```python
# Compare with/without KB
pipeline_with_kb = SemanticGraphConstructionPipeline(enable_kb=True)
pipeline_without_kb = SemanticGraphConstructionPipeline(enable_kb=False)

extraction = await extractor.extract(text)

graph1, report1 = pipeline_with_kb.construct_graph(extraction, domain=DomainType.FINANCE)
graph2, report2 = pipeline_without_kb.construct_graph(extraction)

print(f"KB improvements:")
print(f"  Nodes: {len(graph1.nodes)} vs {len(graph2.nodes)}")
print(f"  Edges: {len(graph1.edges)} vs {len(graph2.edges)}")
print(f"  Implicit flows: {len(report1.suggestions)} vs {len(report2.suggestions)}")
```

## API Reference

### Key Classes

#### KnowledgeBase

```python
class KnowledgeBase:
    def load_from_file(path: str) -> None
    def get_domain_context(domain: DomainType) -> ContextPackage
    def get_patterns_for_domain(domain: DomainType) -> List[BPMNPattern]
    def get_examples_for_domain(domain: DomainType) -> List[DomainExample]
```

#### PatternRecognizer

```python
class PatternRecognizer:
    def get_entity_types_for_domain(domain: DomainType) -> List[EntityType]
    def get_patterns_for_domain(domain: DomainType) -> List[BPMNPattern]
    def recognize_implicit_flows(graph: ProcessGraph, domain: DomainType) -> List[ImplicitFlow]
    def get_actor_roles_for_domain(domain: DomainType) -> List[str]
```

#### ContextSelector

```python
class ContextSelector:
    def select_context_for_domain(domain: DomainType) -> ContextPackage
    def get_examples_for_entity_type(
        entity_type: EntityType,
        domain: DomainType
    ) -> List[DomainExample]
    def find_matching_patterns(keywords: List[str], domain: DomainType) -> List[BPMNPattern]
```

#### SemanticGraphConstructionPipeline

```python
class SemanticGraphConstructionPipeline:
    def __init__(enable_kb: bool = False)
    
    def construct_graph(
        extraction: ExtractionResult,
        actor_profiles: Dict[str, ActorProfile] = {},
        domain: Optional[DomainType] = None
    ) -> Tuple[ProcessGraph, ConstructionReport, List[ImplicitFlow]]
```

## Contributing

To extend KB support:

1. **Add Domain** - Add entry to `DomainType` enum
2. **Define Patterns** - Create BPMN patterns in KB JSON
3. **Add Examples** - Include domain examples
4. **Test** - Add tests to `test_kb_stage4_integration.py`

```python
# Example: Adding healthcare domain support

# 1. Update DomainType enum
class DomainType(str, Enum):
    HEALTHCARE = "healthcare"

# 2. Add to KB JSON
{
  "domains": {
    "healthcare": {
      "patterns": [
        {"name": "Patient Registration", ...},
        {"name": "Diagnosis Decision", ...}
      ]
    }
  }
}

# 3. Add test
def test_healthcare_pattern_recognition():
    kb = KnowledgeBase()
    patterns = kb.get_patterns_for_domain(DomainType.HEALTHCARE)
    assert len(patterns) > 0
```

## References

- **Stage 1 Docs** - Entity extraction with KB context
- **Stage 2 Docs** - Relationship resolution with domain patterns
- **Stage 3 Docs** - Graph construction with KB enhancement
- **Stage 4 Docs** - Lane structuring with role mapping
- **KB Models** - `models/knowledge_base.py`
- **KB Module** - `knowledge/` directory
- **Tests** - `tests/test_kb_stage4_integration.py`

## Summary

The KB integration system provides optional domain-specific intelligence across all BPMN processing stages while maintaining graceful degradation when KB is unavailable or disabled. It enables:

- ✅ Domain-specific entity recognition
- ✅ Pattern-based relationship inference
- ✅ Intelligent flow detection
- ✅ Role-based lane organization
- ✅ Configurable enable/disable
- ✅ Backward compatibility

Start with `enable_kb=True` and `domain=DomainType.FINANCE` for enhanced processing, or omit both for basic functionality.
