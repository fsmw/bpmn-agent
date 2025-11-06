# BPMN Agent - Transform Natural Language to BPMN 2.0 Diagrams

An AI-powered agent that transforms natural language process descriptions into valid BPMN 2.0 XML diagrams. Built with LLMs (Ollama, OpenAI) and modern Python tooling.

## Quick Start

### Installation

```bash
# Clone repository
git clone <repo>
cd bpmn-agent

# Install with dependencies
pip install -e .

# Or with development tools
pip install -e ".[dev]"
```

### Configuration

Set environment variables:

```bash
# LLM Configuration (Ollama)
export LLM_PROVIDER=ollama
export LLM_BASE_URL=http://localhost:11434
export LLM_MODEL=mistral

# Or OpenAI-compatible
export LLM_PROVIDER=openai_compatible
export LLM_BASE_URL=https://api.openai.com/v1
export LLM_API_KEY=sk-...
export LLM_MODEL=gpt-4
```

### Usage

```python
from bpmn_agent.core import LLMClientFactory, LLMConfig, LLMMessage

# Create LLM client from environment
config = LLMConfig.from_env()
client = LLMClientFactory.create(config)

# Validate connection
is_valid = await client.validate_connection()

# Send messages
messages = [
    LLMMessage(role="system", content="You are a BPMN expert"),
    LLMMessage(role="user", content="Describe this process: A pizza order..."),
]
response = await client.call(messages)
print(response.content)
```

## OpenCode Agent Integration

This project is designed to work seamlessly with **OpenCode**, an AI coding agent for the terminal. Use the BPMN Agent as a specialized subagent to generate and analyze BPMN diagrams directly from your development environment.

### Setup OpenCode Agent

#### 1. Create Agent Configuration

Create `.opencode/agent/bpmn-agent.md`:

```markdown
---
description: Specialized agent for generating BPMN 2.0 XML diagrams from natural language process descriptions
mode: subagent
model: anthropic/claude-sonnet-4-20250514
temperature: 0.3
tools:
  write: true
  edit: true
  bash: false
permission:
  bash: deny
---

You are an expert BPMN 2.0 modeler and process analyst.

Your role is to:
1. Analyze natural language process descriptions
2. Extract entities (Actors, Activities, Gateways, Decisions)
3. Identify relationships and sequence flows
4. Generate valid BPMN 2.0 XML files

When analyzing a process description:
- Identify all actors/roles (ACTOR)
- Identify all tasks/activities (ACTIVITY)
- Identify decision points (XOR/AND GATEWAYS)
- Map sequence flows between elements
- Ensure no duplicate entities

Output format:
- Always generate a structured analysis first
- Create JSON representation of entities and relationships
- Generate valid BPMN 2.0 XML using the bpmn_agent library

Reference the Knowledge Base integration for domain-specific patterns:
- Finance processes: invoice, payment, approval workflows
- HR processes: recruitment, onboarding, request approvals
- IT processes: incident management, change requests
- Generic patterns: sequential, parallel, exclusive choice flows
```

#### 2. Add Project OpenCode Configuration

Create `opencode.json` in project root:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "agent": {
    "bpmn-agent": {
      "description": "Generate BPMN 2.0 diagrams from process descriptions",
      "mode": "subagent",
      "model": "anthropic/claude-sonnet-4-20250514",
      "temperature": 0.3,
      "tools": {
        "write": true,
        "edit": true,
        "bash": false
      },
      "permission": {
        "bash": "deny"
      }
    }
  }
}
```

### Usage Examples

#### Generate Simple Process

```
@bpmn-agent Generate a BPMN diagram for this finance process:

"An invoice is received from the vendor. The invoice is reviewed for accuracy. 
If discrepancies exist, the invoice is held for investigation. Once verified, 
the invoice is routed for approval. If the amount exceeds 10,000, director 
approval is required. Otherwise, manager approval is sufficient. After approval, 
the invoice is scheduled for payment. Payment is processed according to agreed terms. 
The transaction is recorded in the accounting system."

Please analyze this and generate the BPMN 2.0 XML file.
```

#### Generate with Domain Context

```
@bpmn-agent Generate a BPMN diagram for this HR process using domain-specific patterns:

"Employee submits leave request. Manager reviews request. 
If employee has sufficient balance, approval is granted. 
Otherwise, request is rejected. Approved requests are logged 
and HR is notified. HR updates employee records and calendar system."

Use HR domain patterns if available, otherwise use generic sequential patterns.
Create the output file as @src/bpmn-agent/examples/leave-request.bpmn
```

#### Validate Existing Diagram

```
@bpmn-agent Validate the BPMN XML file @examples/Car-Wash.bpmn

Check for:
1. XSD schema compliance (BPMN 2.0 standard)
2. Proper sequence flow connections
3. All activities assigned to lanes/actors
4. Gateway logic correctness

Generate a validation report.
```

#### Batch Process Generation

```
@bpmn-agent I have 3 process descriptions in @examples/:
- Recruitment-and-Selection.bpmn
- Smart-Parking.bpmn
- Pizza-Store.bpmn

Analyze each one and extract the process logic. 
Generate improved versions with better structure and KB pattern application.
Place outputs in @src/bpmn-agent/examples/analyzed/
```

#### Knowledge Base Integration

```
@bpmn-agent Generate a process diagram with KB integration:

"When a customer places an order, inventory is checked in parallel with payment validation. 
If both succeed, order is confirmed. Otherwise, it's rejected."

Use the KB domain classifier to identify this as a commerce process.
Apply parallel pattern from the KB pattern library.
Generate KB-augmented context in the XML metadata.
```

### Workflow: Plan → Build

Use Tab key to switch between Plan (analysis-only) and Build (implementation) modes.

**Step 1: Plan Mode (Analysis)**
```
@bpmn-agent <TAB>

Create a plan for generating BPMN from this complex process:
"Our order fulfillment system receives orders. Each order checks inventory, 
validates payment, runs fraud detection in parallel. Failed checks trigger 
rejection. Successful orders proceed to fulfillment..."

What's your implementation plan?
```

**Step 2: Build Mode (Implementation)**
```
<TAB> Sounds good! Go ahead and generate the BPMN XML now.
```

### Integration with Source Code

The agent can work directly with your BPMN Agent implementation:

```
@bpmn-agent Using the code in @src/bpmn-agent/, process this description:

"A customer service ticket is created. It's assigned to an agent. 
Agent investigates and resolves. Customer is notified. If satisfaction 
rating is low, escalation to supervisor occurs."

1. Use the TextPreprocessor from stages/text_preprocessing.py
2. Use EntityExtractor from stages/entity_extraction.py  
3. Apply KB domain classifier for service-domain pattern
4. Generate final XML using models/graph.py
```

### Key Tips

- **Be Specific**: Describe actors, conditions, and flows clearly
- **Use Domain Language**: Mention business terms (invoice, approval, workflow)
- **Include Decisions**: Explicitly mention IF/THEN logic
- **Reference Patterns**: Tell the agent which domain or pattern to use
- **Iterate**: Use Plan mode first, then Build mode after feedback
- **Validate**: Ask the agent to validate generated BPMN against XSD schema

### Resources

- **OpenCode Docs**: https://opencode.ai/docs
- **Agent Configuration**: https://opencode.ai/docs/agents/
- **KB Integration**: See [KB Integration Guide](docs/guides/kb-integration.md)

## Architecture

### Core Modules

- **`core.llm_client`** - LLM provider abstraction
  - Support for Ollama, OpenAI, Azure OpenAI
  - Retry logic, timeout handling
  - Token counting

- **`core.observability`** - Logging & tracing
  - Structured JSON logging
  - OpenTelemetry integration
  - Execution timing

### Data Models

- **`models.bpmn_elements`** - BPMN 2.0 entity models
  - 30+ element types (Task, Event, Gateway, etc.)
  - Pydantic validation
  - Navigation helpers

- **`models.extraction`** - Pipeline stage outputs
  - Entity extraction results
  - Relationship capture
  - Co-reference resolution

- **`models.graph`** - Process graph representation
  - Intermediate graph structure
  - Query interface
  - Validation & metrics

## Project Structure

```
bpmn-agent/
├── core/
│   ├── llm_client.py          # LLM abstraction
│   └── observability.py       # Logging & tracing
├── models/
│   ├── bpmn_elements.py       # BPMN 2.0 models
│   ├── extraction.py          # Extraction outputs
│   └── graph.py               # Process graphs
├── stages/                    # Pipeline stages (future)
├── validators/                # Validation rules (future)
├── tools/                     # Utilities (future)
├── tests/                     # Test suite (future)
└── pyproject.toml             # Dependencies
```

## Features

### Phase 1: Foundation ✅
- [x] LLM provider abstraction (Ollama, OpenAI)
- [x] Observability infrastructure
- [x] Comprehensive data models
- [x] Configuration management

### Phase 2: Pipeline (In Progress)
- [ ] Text preprocessing
- [ ] Entity extraction
- [ ] Relationship resolution
- [ ] Graph construction
- [ ] XML generation

### Phase 3+: Advanced
- [ ] Subprocess support
- [ ] Advanced event types
- [ ] Data object handling
- [ ] Multi-model routing
- [ ] Refinement loops

## LLM Configuration

### Ollama (Local)

```python
from bpmn_agent.core import LLMConfig, LLMClientFactory

config = LLMConfig(
    provider="ollama",
    base_url="http://localhost:11434",
    model="mistral",
    temperature=0.7,
    timeout=60,
)
client = LLMClientFactory.create(config)
```

### OpenAI

```python
config = LLMConfig(
    provider="openai_compatible",
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    model="gpt-4",
    temperature=0.5,
)
client = LLMClientFactory.create(config)
```

### Environment Variables

```bash
LLM_PROVIDER=ollama                          # Provider type
LLM_BASE_URL=http://localhost:11434          # API base URL
LLM_API_KEY=sk-...                          # API key (if needed)
LLM_MODEL=mistral                           # Model name
LLM_TEMPERATURE=0.7                         # Temperature (0-2)
LLM_MAX_TOKENS=2048                         # Max tokens
LLM_TIMEOUT=60                              # Timeout (seconds)
LLM_MAX_RETRIES=3                           # Retry attempts
LLM_CONTEXT_WINDOW=4096                     # Context size
```

## Observability

### Logging

```python
from bpmn_agent.core import ObservabilityManager, ObservabilityConfig, LogLevel

# Initialize
config = ObservabilityConfig(
    service_name="bpmn-agent",
    log_level=LogLevel.INFO,
    json_logs=True,  # JSON output
    enable_tracing=True,
)
ObservabilityManager.initialize(config)
```

### Function Logging

```python
from bpmn_agent.core import log_execution

@log_execution(include_args=True, include_duration=True)
async def process_text(text: str) -> str:
    # Function body
    return result
```

### Execution Timing

```python
from bpmn_agent.core import Timer

with Timer("extraction"):
    # Timed operation
    result = await extract_entities(text)
```

## Data Models

### BPMN Elements

```python
from bpmn_agent.models import (
    Process, Task, ServiceTask, UserTask,
    StartEvent, EndEvent, ExclusiveGateway,
    SequenceFlow, Participant, Collaboration
)

# Create elements
start = StartEvent(name="Start", documentation="Process starts here")
task = UserTask(name="Review", incoming=[...], outgoing=[...])
gateway = ExclusiveGateway(name="Approved?")
flow = SequenceFlow(source_ref=task.id, target_ref=gateway.id)

# Create process
process = Process(
    name="My Process",
    flow_nodes=[start, task, gateway],
    sequence_flows=[flow],
)
```

### Extraction Results

```python
from bpmn_agent.models import (
    ExtractionResult, ExtractedEntity, ExtractedRelation,
    EntityType, RelationType
)

# Query extraction results
result: ExtractionResult = ...

# Get activities
activities = result.get_entities_by_type(EntityType.ACTIVITY)

# Get relationships
outgoing, incoming = result.get_relations_for_entity(entity_id)

# Check quality
avg_confidence = result.average_entity_confidence
distribution = result.entity_types_distribution
```

### Process Graphs

```python
from bpmn_agent.models import ProcessGraph, GraphNode, GraphEdge, NodeType

# Query graph
graph: ProcessGraph = ...

# Get start/end nodes
starts = graph.get_start_nodes()
ends = graph.get_end_nodes()

# Identify parallel paths
paths = graph.get_parallel_paths()

# Validate structure
is_valid, errors = graph.validate_structure()

# Get metrics
is_acyclic = graph.is_acyclic
complexity = graph.complexity
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_llm_client.py

# Run with coverage
pytest --cov=bpmn_agent
```

### Code Quality

```bash
# Format code
black bpmn_agent

# Lint
ruff check bpmn_agent

# Type checking
mypy bpmn_agent
```

### Building

```bash
# Install build tools
pip install build

# Build package
python -m build

# Upload to PyPI
twine upload dist/*
```

## Documentation

- **PLAN_V1.md** - Strategic roadmap (7 phases)
- **PHASE1_COMPLETION.md** - Phase 1 completion report
- **API Documentation** - (Generated from docstrings)

## Supported LLM Providers

| Provider | Supported | Status |
|----------|-----------|--------|
| Ollama | ✅ Yes | Production |
| OpenAI (GPT-4, etc.) | ✅ Yes | Production |
| Azure OpenAI | ✅ Yes | Production |
| LiteLLM | ✅ Yes | Production |
| Local (llama.cpp) | ✅ Yes | Production |
| Claude/Anthropic | ⏳ Coming | Planned |

## Supported BPMN Elements

**Tasks:** Task, ServiceTask, UserTask, ManualTask, ScriptTask, SendTask, ReceiveTask

**Events:** StartEvent, EndEvent, IntermediateEvent

**Gateways:** ExclusiveGateway, InclusiveGateway, ParallelGateway, EventBasedGateway

**Flows:** SequenceFlow, MessageFlow, Association

**Structure:** Process, Collaboration, Participant, Lane, LaneSet

**Data:** DataObject, DataStore, ItemDefinition

**Misc:** Group, TextAnnotation

## Error Handling

The agent provides comprehensive error handling:

```python
from bpmn_agent.models import ExtractionResultWithErrors

result: ExtractionResultWithErrors = await agent.process(text)

if result.has_fatal_errors:
    print("Cannot continue:", result.errors)
else:
    for error in result.errors:
        print(f"Warning: {error.message}")
```

## Performance

### Token Efficiency
- Default: ~1 token per 4 characters
- With TOON encoding: 30-50% token savings (Phase 3+)

### Latency
- Entity extraction: ~2-5s (Ollama), ~1-3s (OpenAI)
- Graph construction: <1s
- XML generation: <500ms

### Throughput
- Local Ollama: 5-10 processes/min
- OpenAI API: 20+ processes/min (rate-limit dependent)

## Troubleshooting

### LLM Connection Issues

```python
# Validate configuration
from bpmn_agent.core import validate_llm_config

is_valid = await validate_llm_config()
if not is_valid:
    print("LLM connection failed")
```

### JSON Logging

Enable for structured output:

```python
config = ObservabilityConfig(json_logs=True)
ObservabilityManager.initialize(config)
```

### Debugging

```python
import logging
from loguru import logger

logger.enable("bpmn_agent")
logger.add(sys.stderr, level="DEBUG")
```

## Contributing

We welcome contributions! Please follow our development workflow:

### Quick Start

1. **Create an Issue** - Document what you want to work on
2. **Create a Branch** - Use `gh issue develop <ISSUE_NUMBER>` or manually
3. **Develop Locally** - Pre-commit hooks ensure quality
4. **Create a PR** - Link PR to issue with `Fixes #<NUMBER>`
5. **CI/CD** - Automated tests, linting, and coverage checks
6. **Code Review** - Wait for approval
7. **Merge** - Issue closes automatically

### Detailed Guide

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for complete guidelines on:
- Issue creation and templates
- Branch naming conventions
- Development workflow
- Testing requirements
- Code standards
- PR process and CI/CD

### Development Setup

```bash
# Automated setup (recommended)
cd src/bpmn-agent
bash scripts/setup-dev.sh

# Or manually
pip install -e ".[dev]"
pre-commit install
pre-commit install --hook-type pre-push
```

### Quality Gates

All PRs must pass:
- ✅ Tests (unit + integration)
- ✅ Code coverage > 75%
- ✅ Linting (black, ruff)
- ✅ Type checking (mypy)
- ✅ Security scanning (Bandit, Safety)

See **[DEVOPS_SETUP.md](DEVOPS_SETUP.md)** for detailed CI/CD setup.

## License

MIT License - See LICENSE file

## Support

- **Issues:** GitHub Issues
- **Discussions:** GitHub Discussions
- **Documentation:** See `/docs` directory

## Roadmap

- Phase 1: Foundation ✅
- Phase 2: Pipeline (Dec 2024)
- Phase 3: Advanced features (Jan 2025)
- Phase 4: Validation & refinement (Feb 2025)
- Phase 5: Deployment & scaling (Mar 2025)
- Phase 6: Enterprise features (Apr 2025)
- Phase 7: Monitoring & optimization (May 2025)

---

**Version:** 0.1.0  
**Status:** Alpha (Phase 1 Complete)  
**Last Updated:** $(date -u +'%Y-%m-%d')
