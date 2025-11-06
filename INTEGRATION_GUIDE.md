# BPMN Agent Integration Guide

Comprehensive guide for integrating the BPMN Agent into your applications and workflows.

## Table of Contents

1. [Quick Integration](#quick-integration)
2. [Processing Modes](#processing-modes)
3. [API Usage Examples](#api-usage-examples)
4. [Error Handling](#error-handling)
5. [Configuration Guide](#configuration-guide)
6. [Performance Tuning](#performance-tuning)
7. [Knowledge Base Integration](#knowledge-base-integration)
8. [CLI Usage](#cli-usage)
9. [Troubleshooting](#troubleshooting)

---

## Quick Integration

### Installation

```bash
pip install bpmn-agent
```

With development tools:

```bash
pip install "bpmn-agent[dev]"
```

### Basic Usage (5 lines)

```python
from bpmn_agent import BPMNAgent, AgentConfig, LLMConfig

config = AgentConfig(
    llm_config=LLMConfig.from_env()
)
agent = BPMNAgent(config)
xml, state = await agent.process("Your process description")
print(xml)  # Valid BPMN 2.0 XML
```

---

## Processing Modes

The agent supports multiple processing modes for different use cases:

### 1. Standard Mode (Default)

Executes all 5 pipeline stages:

```python
from bpmn_agent import ProcessingMode

config = AgentConfig(
    mode=ProcessingMode.STANDARD,  # Default
    llm_config=LLMConfig.from_env()
)
agent = BPMNAgent(config)

# Process all stages → Generate XML
xml, state = await agent.process("Process description")
```

**Output:** Complete BPMN 2.0 XML file

---

### 2. Analysis-Only Mode

Analyze process without generating XML (stages 1-4):

```python
config = AgentConfig(
    mode=ProcessingMode.ANALYSIS_ONLY,
    llm_config=LLMConfig.from_env()
)
agent = BPMNAgent(config)

# Process stages 1-4, skip XML generation
xml, state = await agent.process("Process description")
assert xml is None  # No XML output
assert len(state.stage_results) == 4  # 4 stages executed

# Access analysis results
entities = state.metrics.entities_extracted
relations = state.metrics.relations_extracted
graph_nodes = state.metrics.graph_nodes
```

**Use Cases:**
- Process discovery without diagram generation
- Entity extraction testing
- Relationship validation
- Graph quality assessment

---

### 3. Validation-Only Mode

Quick text validation without processing (stage 1 only):

```python
config = AgentConfig(
    mode=ProcessingMode.VALIDATION_ONLY,
    llm_config=LLMConfig.from_env()
)
agent = BPMNAgent(config)

# Only preprocess and validate text
xml, state = await agent.process("Process description")
assert xml is None  # No XML
assert len(state.stage_results) == 1  # Only stage 1

# Check text preprocessing metrics
text_length = state.metrics.input_text_length
token_count = state.metrics.input_token_count
chunks = len(state.stage_results[0].result.chunks)
```

**Use Cases:**
- Input validation
- Token counting
- Quick preprocessing checks
- Batch text validation

---

### 4. KB-Enhanced Mode

Use knowledge base domain patterns:

```python
config = AgentConfig(
    mode=ProcessingMode.KB_ENHANCED,
    enable_kb=True,
    kb_domain_auto_detect=True,  # Auto-detect domain
    llm_config=LLMConfig.from_env()
)
agent = BPMNAgent(config)

# Auto-detects domain (finance, hr, it, etc.)
xml, state = await agent.process(
    "Invoice processing workflow...",
    domain="finance"  # Optional override
)
```

**Benefits:**
- Domain-specific entity patterns
- Context-aware relationships
- Implicit flow inference
- Swimlane structure hints
- Better accuracy for known domains

---

## API Usage Examples

### Example 1: Simple Sequential Process

```python
from bpmn_agent import BPMNAgent, AgentConfig, LLMConfig

# Configure
config = AgentConfig(
    llm_config=LLMConfig(
        provider="ollama",
        base_url="http://localhost:11434",
        model="mistral",
    )
)

agent = BPMNAgent(config)

# Process
description = """
A customer places an order. The order is validated. 
If valid, payment is processed. After payment, 
the order is confirmed and sent to fulfillment.
"""

xml, state = await agent.process(
    description,
    process_name="Order Processing"
)

print(f"Generated BPMN XML ({len(xml)} chars)")
print(f"Entities: {state.metrics.entities_extracted}")
print(f"Relations: {state.metrics.relations_extracted}")
print(f"Graph nodes: {state.metrics.graph_nodes}")
```

---

### Example 2: Parallel Process with Gateways

```python
description = """
When an order is received, two operations happen in parallel:
1. Inventory is checked
2. Payment is validated

Both must succeed. If inventory is low, the order is rejected.
If payment fails, the order is rejected. 
If both succeed, order is confirmed.
"""

xml, state = await agent.process(description)

# Check execution metrics
print(f"Duration: {state.metrics.total_duration_ms}ms")
print(f"Completed stages: {state.metrics.completed_stages}")
print(f"Has errors: {len(state.errors) > 0}")

# Access graph structure
if state.stage_results:
    last_result = state.stage_results[-1]
    if last_result.status == StageStatus.COMPLETED:
        graph = last_result.result
        print(f"Graph complexity: {graph.complexity}")
        print(f"Is acyclic: {graph.is_acyclic}")
```

---

### Example 3: Finance Process with KB

```python
config = AgentConfig(
    mode=ProcessingMode.KB_ENHANCED,
    enable_kb=True,
    llm_config=LLMConfig.from_env()
)
agent = BPMNAgent(config)

description = """
Invoice processing:
1. Invoice received and registered
2. Amount verification against PO
3. If discrepancies found → hold for review
4. If verified → routing approval
5. If amount > 10K → director approval required
6. Otherwise → manager approval sufficient
7. After approval → schedule for payment
8. Payment processing per agreed terms
9. Transaction recorded in accounting
"""

xml, state = await agent.process(
    description,
    process_name="Invoice Processing",
    domain="finance"
)

# KB context applied
print(f"Domain: {state.input_domain}")
print(f"KB patterns applied: {bool(state.stage_results[-1].metrics.get('kb_patterns'))}")

with open("invoice_process.bpmn", "w") as f:
    f.write(xml)
```

---

### Example 4: Error Recovery

```python
from bpmn_agent import ErrorHandlingStrategy

config = AgentConfig(
    error_handling=ErrorHandlingStrategy.RECOVERY,  # Continue on errors
    llm_config=LLMConfig.from_env()
)
agent = BPMNAgent(config)

description = "Complex process with potential extraction issues..."

try:
    xml, state = await agent.process(description)
    
    # Check for errors during processing
    if state.errors:
        print("Warnings during processing:")
        for error in state.errors:
            print(f"  - {error}")
    
    # XML may still be generated despite warnings
    if xml:
        print("BPMN generated despite some errors")
        
except Exception as e:
    print(f"Fatal error: {e}")
```

---

## Error Handling

### Error Handling Strategies

```python
from bpmn_agent import ErrorHandlingStrategy

# Strategy 1: STRICT (stop on first error)
config = AgentConfig(
    error_handling=ErrorHandlingStrategy.STRICT,
    llm_config=LLMConfig.from_env()
)
# Stops at first error, returns None

# Strategy 2: RECOVERY (continue despite errors)
config = AgentConfig(
    error_handling=ErrorHandlingStrategy.RECOVERY,
    llm_config=LLMConfig.from_env()
)
# Attempts to continue, collects errors
```

### Checking State for Errors

```python
xml, state = await agent.process(description)

if state.is_failed:
    print("Processing failed:")
    for error in state.errors:
        print(f"  ERROR: {error}")
    print(f"Error count: {len(state.errors)}")

if state.warnings:
    print("Warnings:")
    for warning in state.warnings:
        print(f"  WARNING: {warning}")

# Check individual stage results
for result in state.stage_results:
    if result.status == StageStatus.FAILED:
        print(f"Stage {result.stage_name} failed: {result.error}")
    elif result.warnings:
        print(f"Stage {result.stage_name} warnings: {result.warnings}")
```

---

## Configuration Guide

### Environment Variables

```bash
# LLM Configuration
export LLM_PROVIDER=ollama                    # ollama, openai_compatible
export LLM_BASE_URL=http://localhost:11434   # API endpoint
export LLM_API_KEY=sk-...                    # For OpenAI/Azure
export LLM_MODEL=mistral                     # Model name
export LLM_TEMPERATURE=0.7                   # 0-2, default 0.7
export LLM_MAX_TOKENS=2048                   # Max response tokens
export LLM_TIMEOUT=60                        # Request timeout (seconds)
export LLM_MAX_RETRIES=3                     # Retry attempts
export LLM_CONTEXT_WINDOW=4096               # Context size

# Agent Configuration
export AGENT_MODE=standard                   # standard, analysis_only, validation_only, kb_enhanced
export AGENT_ENABLE_KB=true                  # true, false
export AGENT_KB_DOMAIN_AUTO_DETECT=true      # true, false
export AGENT_ERROR_HANDLING=recovery         # strict, recovery
export AGENT_ENABLE_LOGGING=true             # true, false

# Observability
export LOG_LEVEL=info                        # debug, info, warning, error
export ENABLE_JSON_LOGS=true                 # true, false
export ENABLE_TRACING=true                   # true, false
```

### Programmatic Configuration

```python
from bpmn_agent import (
    AgentConfig, LLMConfig, ProcessingMode, 
    ErrorHandlingStrategy, ObservabilityConfig, LogLevel
)

# LLM Configuration
llm_config = LLMConfig(
    provider="ollama",
    base_url="http://localhost:11434",
    model="mistral",
    temperature=0.7,
    max_tokens=2048,
    timeout=60,
    max_retries=3,
    context_window=4096,
)

# Observability Configuration
obs_config = ObservabilityConfig(
    service_name="bpmn-agent",
    log_level=LogLevel.INFO,
    json_logs=True,
    enable_tracing=True,
)

# Agent Configuration
agent_config = AgentConfig(
    llm_config=llm_config,
    mode=ProcessingMode.STANDARD,
    enable_kb=True,
    kb_domain_auto_detect=True,
    error_handling=ErrorHandlingStrategy.RECOVERY,
    enable_logging=True,
)

agent = BPMNAgent(agent_config)
```

---

## Performance Tuning

### Token Efficiency

```python
# Check token count before processing
from bpmn_agent.core import TokenCounter

counter = TokenCounter()
text = "Your process description..."

token_count = counter.count(text)
print(f"Text will use ~{token_count} tokens")

# Estimated cost (OpenAI GPT-4)
cost = (token_count / 1000) * 0.03  # $0.03 per 1K tokens
print(f"Estimated cost: ${cost:.4f}")
```

### Batch Processing

```python
import asyncio
from bpmn_agent import BPMNAgent, AgentConfig, LLMConfig

config = AgentConfig(llm_config=LLMConfig.from_env())
agent = BPMNAgent(config)

descriptions = [
    "Process 1 description...",
    "Process 2 description...",
    "Process 3 description...",
]

# Sequential (slower but less resource-intensive)
results = []
for desc in descriptions:
    xml, state = await agent.process(desc)
    results.append((xml, state))

# Parallel (faster but requires more resources)
tasks = [
    agent.process(desc)
    for desc in descriptions
]
results = await asyncio.gather(*tasks)

print(f"Processed {len(results)} diagrams")
```

### Throughput Optimization

```python
# Configure for throughput (lower accuracy, faster)
llm_config = LLMConfig(
    model="mistral",        # Faster model
    temperature=0.5,        # Lower = faster/more consistent
    max_tokens=1024,        # Smaller output
    timeout=30,             # Shorter timeout
)

config = AgentConfig(
    llm_config=llm_config,
    mode=ProcessingMode.ANALYSIS_ONLY,  # Skip XML generation
    error_handling=ErrorHandlingStrategy.RECOVERY,
)

agent = BPMNAgent(config)
```

### Accuracy Optimization

```python
# Configure for accuracy (slower, higher quality)
llm_config = LLMConfig(
    model="gpt-4",          # More capable model
    temperature=0.3,        # Lower = more deterministic
    max_tokens=4096,        # Larger output allowed
    timeout=120,            # Longer timeout for thinking
)

config = AgentConfig(
    llm_config=llm_config,
    mode=ProcessingMode.KB_ENHANCED,    # Use KB patterns
    enable_kb=True,
    kb_domain_auto_detect=True,
    error_handling=ErrorHandlingStrategy.STRICT,
)

agent = BPMNAgent(config)
```

---

## Knowledge Base Integration

### Using Domain-Specific Patterns

```python
config = AgentConfig(
    mode=ProcessingMode.KB_ENHANCED,
    enable_kb=True,
    llm_config=LLMConfig.from_env()
)
agent = BPMNAgent(config)

# Finance domain
description = "Invoice received, verified, approved, paid..."
xml, state = await agent.process(
    description,
    domain="finance"  # Finance patterns applied
)

# HR domain
description = "Employee applies for leave, manager approves..."
xml, state = await agent.process(
    description,
    domain="hr"  # HR patterns applied
)

# IT domain
description = "Incident created, investigated, resolved..."
xml, state = await agent.process(
    description,
    domain="it"  # IT patterns applied
)
```

### Implicit Flow Inference

```python
# KB automatically infers implicit flows
description = """
Customer submits order. System checks inventory and payment in parallel.
Both must succeed for order confirmation.
"""

xml, state = await agent.process(description)

# KB detected parallel pattern and added:
# - ParallelGateway for inventory+payment
# - Implicit synchronization
# - Proper flow convergence

print(f"KB patterns applied: {state.stage_results[-1].metrics.get('kb_suggestions')}")
```

---

## CLI Usage

### Command Line Interface

```bash
# Show help
bpmn-agent --help

# Process single file
bpmn-agent process --input description.txt --output process.bpmn

# Process with options
bpmn-agent process \
  --input description.txt \
  --output process.bpmn \
  --mode standard \
  --kb \
  --domain finance

# Validate BPMN file
bpmn-agent validate process.bpmn

# Show agent status
bpmn-agent status

# Show available domains
bpmn-agent domains
```

### Batch Processing

```bash
# Process multiple files
for file in descriptions/*.txt; do
    bpmn-agent process \
      --input "$file" \
      --output "diagrams/$(basename $file .txt).bpmn"
done
```

---

## State Tracking & Metrics

### Accessing Processing State

```python
xml, state = await agent.process(description)

# Session information
print(f"Session ID: {state.session_id}")
print(f"Start time: {state.start_time}")

# Processing stages
print(f"Total stages executed: {len(state.stage_results)}")
for result in state.stage_results:
    print(f"  {result.stage_name}: {result.status} ({result.duration_ms}ms)")

# Metrics summary
summary = state.summary()
print(f"Completion rate: {summary['completion_rate']:.1%}")
print(f"Is complete: {summary['is_complete']}")
print(f"Error count: {summary['error_count']}")
print(f"Duration: {summary['total_duration_ms']}ms")
```

### Detailed Metrics

```python
metrics = state.metrics

# Input metrics
print(f"Input text length: {metrics.input_text_length} chars")
print(f"Input tokens: {metrics.input_token_count}")

# Extraction metrics
print(f"Entities extracted: {metrics.entities_extracted}")
print(f"Relations extracted: {metrics.relations_extracted}")
print(f"Avg entity confidence: {metrics.avg_entity_confidence:.2%}")

# Resolution metrics
print(f"Coreferences resolved: {metrics.coreferences_resolved}")
print(f"Actors consolidated: {metrics.actors_consolidated}")

# Graph metrics
print(f"Graph nodes: {metrics.graph_nodes}")
print(f"Graph edges: {metrics.graph_edges}")
print(f"Cycles detected: {metrics.graph_cycles_detected}")

# Output metrics
print(f"Output XML size: {metrics.output_xml_size_bytes} bytes")
print(f"Output elements: {metrics.output_element_count}")
```

---

## Troubleshooting

### LLM Connection Issues

```python
from bpmn_agent.core import LLMClientFactory, LLMConfig

config = LLMConfig.from_env()
client = LLMClientFactory.create(config)

# Test connection
try:
    is_valid = await client.validate_connection()
    print(f"LLM connection: {'OK' if is_valid else 'FAILED'}")
except Exception as e:
    print(f"Connection error: {e}")
```

### Common Issues

**Issue:** "ModuleNotFoundError: No module named 'bpmn_agent'"

```bash
# Solution: Install package
pip install -e .
```

**Issue:** "Connection refused" (LLM)

```bash
# Solution: Check LLM is running
curl http://localhost:11434/api/tags

# Or configure different provider
export LLM_PROVIDER=openai_compatible
export LLM_BASE_URL=https://api.openai.com/v1
export LLM_API_KEY=sk-...
```

**Issue:** "Timeout waiting for LLM response"

```python
# Solution: Increase timeout
llm_config = LLMConfig(
    timeout=120,  # 2 minutes instead of 60
    max_retries=5,
)
```

**Issue:** "No entities extracted"

```python
# Solution: Enable KB or provide domain hint
config = AgentConfig(
    enable_kb=True,
    kb_domain_auto_detect=True,
)

xml, state = await agent.process(
    description,
    domain="finance"  # Explicit domain
)
```

### Debugging

```python
import logging
from loguru import logger

# Enable debug logging
logger.enable("bpmn_agent")
logger.add("debug.log", level="DEBUG")

# Enable JSON logging
from bpmn_agent.core import ObservabilityConfig, LogLevel

obs_config = ObservabilityConfig(
    log_level=LogLevel.DEBUG,
    json_logs=True,
)

# Process and check logs
xml, state = await agent.process(description)

# Check stage results for issues
for result in state.stage_results:
    if result.error:
        print(f"Error in {result.stage_name}: {result.error}")
    if result.warnings:
        print(f"Warnings in {result.stage_name}: {result.warnings}")
```

---

## Summary

| Feature | Standard | Analysis-Only | Validation | KB-Enhanced |
|---------|----------|---------------|------------|-------------|
| Text preprocessing | ✅ | ✅ | ✅ | ✅ |
| Entity extraction | ✅ | ✅ | ❌ | ✅ |
| Entity resolution | ✅ | ✅ | ❌ | ✅ |
| Graph construction | ✅ | ✅ | ❌ | ✅ |
| XML generation | ✅ | ❌ | ❌ | ✅ |
| KB patterns | ✅ | ✅ | ❌ | ✅ |
| Speed | Fast | Fast | Very Fast | Slower |
| Accuracy | High | High | Low | Very High |

---

**For more information, see:**
- `README.md` - Overview and quick start
- `KB_INTEGRATION.md` - Knowledge base details
- Tests in `tests/` - Usage examples

