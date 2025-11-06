# Demos and Executable Examples

This folder contains demonstration scripts and executable examples for the BPMN Agent project.

## 📋 Available Demos

### Orchestrator Demos

- **`orchestrator.py`** - Basic orchestrator demo
- **`phase3_orchestrator.py`** - Orchestrator demo with Phase 3 features
- **`phase3_tools.py`** - Phase 3 tools demo
- **`phase3_tools_working.py`** - Working Phase 3 tools demo

### Validation Demos

- **`validation_phase4_demo.sh`** - Phase 4 validation demonstration script

## 🚀 Execution

### Prerequisites

1. Make sure you have the virtual environment activated:
   ```bash
   cd /home/fsmw/dev/bpmn/src/bpmn-agent
   source .venv/bin/activate
   ```

2. Verify that dependencies are installed:
   ```bash
   pip install -e ".[dev]"
   ```

### Run Demos

```bash
# Basic orchestrator demo
python demo/orchestrator.py

# Phase 3 demo
python demo/phase3_orchestrator.py

# Phase 4 validation demo
bash demo/validation_phase4_demo.sh
```

## 📝 Notes

- Demos may require environment variable configuration (LLM_PROVIDER, etc.)
- Some demos may require external services (Ollama, OpenAI, etc.)
- Review comments in each file for more details

## 🔗 See Also

- [User Guides](../docs/guides/)
- [BPMN Examples](../examples/)
- [Main Documentation](../README.md)
