# Knowledge Base Integration - Practical Examples

This guide provides concrete examples of using the BPMN Agent with Knowledge Base enhancements.

## Table of Contents
1. [CLI Examples](#cli-examples)
2. [Python API Examples](#python-api-examples)
3. [Domain-Specific Examples](#domain-specific-examples)
4. [Advanced Patterns](#advanced-patterns)

## CLI Examples

### Basic Knowledge Base Enhanced Processing

```bash
# Process with KB enhancement enabled
.venv/bin/python -m bpmn_agent.tools.cli process \
  "A customer submits an order. The system validates it. If valid, confirm and send to warehouse." \
  --mode kb_enhanced \
  --output order_process.bpmn
```

### Domain-Specific Processing

```bash
# Finance domain processing
.venv/bin/python -m bpmn_agent.tools.cli process \
  "Payment approval flow: Payment initiated, fraud check performed, if cleared approve, else reject." \
  --mode kb_enhanced \
  --domain finance \
  --output payment_flow.bpmn

# Healthcare domain processing
.venv/bin/python -m bpmn_agent.tools.cli process \
  "Patient admission: Check-in, vitals assessment, doctor consultation, discharge if stable." \
  --mode kb_enhanced \
  --domain healthcare \
  --output patient_admission.bpmn

# IT domain processing
.venv/bin/python -m bpmn_agent.tools.cli process \
  "Deployment pipeline: Code commit, automated tests, build, deploy to staging, manual approval, production deploy." \
  --mode kb_enhanced \
  --domain it \
  --output deployment_pipeline.bpmn
```

### Analysis-Only Mode

```bash
# Get pattern matching insights without generating BPMN
.venv/bin/python -m bpmn_agent.tools.cli process \
  "Order processing with quality checks" \
  --mode analysis_only
```

### Health Check

```bash
# Check agent status and KB integration
.venv/bin/python -m bpmn_agent.tools.cli health
```

Expected output:
```json
{
  "status": "healthy",
  "llm_provider": "ollama",
  "mode": "standard",
  "kb_enabled": true,
  "timestamp": "2025-11-06T12:43:57.649501"
}
```

## Python API Examples

### Basic Usage with KB

```python
from bpmn_agent import BPMNAgent, AgentConfig, ProcessingMode
from bpmn_agent.core.llm_client import LLMConfig, LLMProviderType

# Configure agent with KB
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

# Process text
text = "Customer places order, system validates, if valid confirm else reject"
result = await agent.process(text)

# Get BPMN XML
xml = result.bpmn_xml
print(xml)
```

### With Domain Classification

```python
from bpmn_agent.knowledge import DomainClassifier

classifier = DomainClassifier()

# Classify domain
text = "Payment transaction verification and approval"
domain = classifier.classify_domain(text)

print(f"Detected domain: {domain.primary_domain}")
print(f"Confidence: {domain.confidence}")
print(f"Complexity: {domain.complexity_level}")
```

### Pattern Recognition

```python
from bpmn_agent.knowledge import AdvancedPatternMatcher

matcher = AdvancedPatternMatcher()

# Find matching patterns
text = "Order processing: validate payment, check inventory, ship if available"
matches = matcher.find_patterns(text)

for match in matches:
    print(f"Pattern: {match.pattern_name}")
    print(f"Category: {match.category}")
    print(f"Confidence: {match.confidence}")
```

### Context-Aware Generation

```python
from bpmn_agent.knowledge import ContextOptimizer

optimizer = ContextOptimizer()

# Get optimized context for generation
text = "Insurance claim workflow"
context = optimizer.select_best_context(
    text=text,
    max_tokens=1000
)

print(f"Selected patterns: {len(context.selected_patterns)}")
print(f"Token count: {context.token_count}")
```

## Domain-Specific Examples

### Finance Domain

**Scenario:** Payment Processing Workflow

```
Raw Text:
"When a payment is received, verify the customer account. 
If the account is valid, process the payment. 
If payment exceeds limit, require approval. 
Once processed, send confirmation to customer."

With KB Enhancement:
- Detects: payment_processing, account_verification patterns
- Suggests: ExclusiveGateway for payment limit check
- Enriches: Adds confirmation notification task
```

CLI Example:
```bash
.venv/bin/python -m bpmn_agent.tools.cli process \
  "Payment received -> verify account -> check limit -> approve/reject -> send confirmation" \
  --mode kb_enhanced \
  --domain finance
```

### Healthcare Domain

**Scenario:** Patient Triage Process

```
Raw Text:
"Patient arrives at ED, initial assessment performed, 
vital signs checked. If critical, go to ICU. 
If stable, to regular ward. All documented in system."

With KB Enhancement:
- Detects: triage_assessment, vital_signs_check patterns
- Suggests: ParallelGateway for concurrent assessments
- Enriches: Adds documentation tasks
```

### Manufacturing Domain

**Scenario:** Quality Control Process

```
Raw Text:
"Product manufactured, quality check performed. 
If passes, package and ship. If fails, rework or scrap."

With KB Enhancement:
- Detects: quality_inspection, defect_handling patterns
- Suggests: ExclusiveGateway for pass/fail decision
- Enriches: Adds rework loop handling
```

### IT Domain

**Scenario:** Software Release Pipeline

```
Raw Text:
"Code committed, automated tests run. 
If pass, build artifact. If fail, notify team. 
Build artifacts sent to staging for QA approval."

With KB Enhancement:
- Detects: ci_cd_pipeline, automated_testing patterns
- Suggests: ParallelGateway for parallel test execution
- Enriches: Adds artifact storage, rollback handling
```

## Advanced Patterns

### Custom KB Context Injection

```python
from bpmn_agent import BPMNAgent
from bpmn_agent.knowledge import KnowledgeBase

kb = KnowledgeBase()
kb.load_patterns()

# Get domain-specific hints
hints = kb.get_examples_by_domain("finance")

# Process with injected context
result = await agent.process(
    text="Your workflow description",
    context_hints=hints
)
```

### Pattern Validation

```python
from bpmn_agent.models import BPMNPattern

# Validate pattern recognition
patterns = matcher.find_patterns(text)

for pattern in patterns:
    assert pattern.category in ["sequential", "decision", "parallel"]
    assert pattern.confidence >= 0.5
    assert len(pattern.description) > 0
```

### Complexity Analysis

```python
# Analyze workflow complexity
from bpmn_agent.knowledge import ComplexityAnalyzer

analyzer = ComplexityAnalyzer()

complexity = analyzer.analyze(text)
print(f"Complexity Level: {complexity.level}")  # LOW, MEDIUM, HIGH
print(f"Element Count: {complexity.estimated_elements}")
print(f"Suggested Splits: {complexity.suggested_splits}")
```

### Batch Processing with KB

```python
# Process multiple workflows with consistent KB context
workflows = [
    "Order processing flow",
    "Payment verification flow",
    "Inventory management flow"
]

for workflow_text in workflows:
    result = await agent.process(
        workflow_text,
        mode=ProcessingMode.KB_ENHANCED
    )
    
    # Save output
    xml = result.bpmn_xml
    filename = f"{workflow_text.replace(' ', '_')}.bpmn"
    with open(filename, 'w') as f:
        f.write(xml)
```

## KB Modes Comparison

| Feature | Standard | KB Enhanced | Analysis Only |
|---------|----------|-------------|---------------|
| LLM Processing | ✓ | ✓ | ✗ |
| Pattern Matching | ✗ | ✓ | ✓ |
| BPMN Generation | ✓ | ✓ | ✗ |
| Domain Detection | ✗ | ✓ | ✓ |
| Complexity Analysis | ✗ | ✓ | ✓ |
| Performance | Fast | Medium | Very Fast |
| Accuracy | Good | Excellent | Insights Only |

## Troubleshooting

### LLM Connection Issues

```bash
# Check health status
.venv/bin/python -m bpmn_agent.tools.cli health

# Expected: status: "healthy"
# If unhealthy, ensure Ollama is running:
# ollama serve
```

### KB Loading Issues

```python
# Verify KB is loaded
from bpmn_agent.models.knowledge_base import KnowledgeBase

kb = KnowledgeBase()
kb.load_patterns()

# Check loaded patterns
print(f"Loaded patterns: {len(kb.patterns)}")
for domain in kb.list_domains():
    print(f"  {domain}: {len(kb.get_examples_by_domain(domain))} patterns")
```

### Pattern Recognition Not Working

```python
# Check pattern files exist
import os
pattern_dir = "/home/fsmw/dev/bpmn/src/bpmn-agent/bpmn_agent/knowledge/patterns"
pattern_files = [f for f in os.listdir(pattern_dir) if f.endswith('.json')]
print(f"Pattern files found: {pattern_files}")
```

## Performance Tips

1. **Use KB Enhanced mode for better results** - Small performance cost, big accuracy gain
2. **Batch similar domain workflows** - Reuse KB context for efficiency
3. **Use analysis_only for insights** - No LLM calls, very fast domain detection
4. **Cache domain classifications** - Avoid re-classifying same text

## Next Steps

- See [KB_INTEGRATION.md](KB_INTEGRATION.md) for technical details
- See [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) for advanced integration
- See [README.md](README.md) for general project information
- See [VENV_USAGE.md](VENV_USAGE.md) for virtual environment setup

