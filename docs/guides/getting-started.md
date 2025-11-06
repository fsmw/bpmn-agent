# Getting Started with BPMN Agent

Quick start guide to get the BPMN Agent up and running.

## Prerequisites

- Python 3.10+
- Virtual environment (`.venv`) - already set up
- Ollama (optional, for LLM processing) or OpenAI API key

## Installation

The project is already installed in development mode. To verify:

```bash
cd /home/fsmw/dev/bpmn/src/bpmn-agent

# Verify virtual environment setup
./verify_venv_setup.sh

# All systems should show ✓
```

## Quick Start

### 1. Check Agent Status

```bash
.venv/bin/python -m bpmn_agent.tools.cli health
```

Expected output:
```json
{
  "status": "healthy",
  "llm_provider": "ollama",
  "kb_enabled": true
}
```

### 2. Process Your First Workflow

```bash
# Simple order processing workflow
.venv/bin/python -m bpmn_agent.tools.cli process \
  "Customer places order. System validates order. If valid, confirm. If invalid, reject." \
  --output my_first_process.bpmn
```

### 3. Use Knowledge Base Enhancement

```bash
# Same workflow with KB patterns for better results
.venv/bin/python -m bpmn_agent.tools.cli process \
  "Customer places order. System validates order. If valid, confirm. If invalid, reject." \
  --mode kb_enhanced \
  --output my_first_process_kb.bpmn
```

### 4. Validate Generated BPMN

```bash
.venv/bin/python -m bpmn_agent.tools.cli validate my_first_process.bpmn
```

## Common Tasks

### Process with Domain Hints

```bash
# Tell agent the domain for better pattern matching
.venv/bin/python -m bpmn_agent.tools.cli process \
  "Payment initiated -> Fraud check -> Approve or Reject -> Send confirmation" \
  --mode kb_enhanced \
  --domain finance \
  --output payment_flow.bpmn
```

### Get Analysis Without BPMN Generation

```bash
# Fast analysis: domain detection + pattern matching
.venv/bin/python -m bpmn_agent.tools.cli process \
  "Your workflow description" \
  --mode analysis_only
```

### View Configuration

```bash
.venv/bin/python -m bpmn_agent.tools.cli info
```

## Python API Usage

### Basic Example

```python
from bpmn_agent import BPMNAgent, AgentConfig, ProcessingMode
from bpmn_agent.core.llm_client import LLMConfig, LLMProviderType

# Configure
config = AgentConfig(
    llm_config=LLMConfig(
        provider=LLMProviderType.OLLAMA,
        model="mistral"
    ),
    processing_mode=ProcessingMode.KB_ENHANCED,
    knowledge_base_enabled=True
)

# Create agent
agent = BPMNAgent(config)

# Process
import asyncio
result = asyncio.run(agent.process("Your workflow description"))

# Get result
print(result.bpmn_xml)
```

## Running Tests

```bash
# Run all tests
.venv/bin/pytest tests/ -v

# Run specific test file
.venv/bin/pytest tests/test_entity_extraction.py -v

# Run with coverage
.venv/bin/pytest tests/ --cov=bpmn_agent
```

## Project Structure

```
/home/fsmw/dev/bpmn/src/bpmn-agent/
├── bpmn_agent/                 # Main package
│   ├── core/                   # LLM + observability
│   ├── stages/                 # 5-stage pipeline
│   ├── models/                 # Data models
│   ├── knowledge/              # KB integration
│   ├── tools/                  # CLI interface
│   └── agent/                  # Orchestrator
├── tests/                      # Test suite (397 tests)
├── .venv/                      # Virtual environment (MUST use this)
└── README.md                   # Full documentation
```

## Important: Virtual Environment

**ALWAYS use the local `.venv`:**

```bash
# ✓ CORRECT
.venv/bin/python -m bpmn_agent.tools.cli process "..."
.venv/bin/pytest tests/

# ✗ WRONG - Never do this
python -m bpmn_agent.tools.cli process "..."
python3 script.py
```

Read [VENV Usage](../development/venv-usage.md) for details.

## Documentation

- **[README.md](README.md)** - Full project documentation
- **[KB_INTEGRATION_EXAMPLES.md](KB_INTEGRATION_EXAMPLES.md)** - Practical KB examples
- **[KB_INTEGRATION.md](KB_INTEGRATION.md)** - Technical KB details
- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** - Advanced integration
- **[VENV_USAGE.md](VENV_USAGE.md)** - Virtual environment setup
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - CLI command reference

## Troubleshooting

### Agent Not Healthy

```bash
# Check status
.venv/bin/python -m bpmn_agent.tools.cli health

# If unhealthy, check Ollama
# Option 1: Run Ollama
ollama serve

# Option 2: Use OpenAI instead
export OPENAI_API_KEY="your-key"
```

### Tests Failing

```bash
# Run with verbose output
.venv/bin/pytest tests/ -vv

# Check imports work
.venv/bin/python -c "from bpmn_agent import BPMNAgent; print('✓ Imports work')"
```

### Import Errors

```bash
# Verify package structure
.venv/bin/python -c "
from bpmn_agent.core.llm_client import LLMClientFactory
from bpmn_agent.models.bpmn_elements import Task
from bpmn_agent.knowledge.domain_classifier import DomainClassifier
print('✓ All imports working')
"
```

## Next Steps

1. ✅ Virtual environment verified
2. ✅ CLI commands working
3. ✅ Tests passing (397/397)
4. 📖 Explore examples in [KB Integration Examples](kb-integration-examples.md)
5. 🔧 Build your own workflows
6. 📚 Read full docs for advanced usage

## Support

For issues:
- Check documentation files
- Review test files for usage patterns
- See troubleshooting section above
- Check GitHub issues (if applicable)

---

**Happy BPMN diagramming! 🎉**

