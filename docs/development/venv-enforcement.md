# Virtual Environment Enforcement Setup

## Changes Made to Guarantee Local .venv Usage

### 1. Project Structure Reorganization
- **Before:** Modules were in separate directories at project root
  - `/core/`, `/knowledge/`, `/models/`, etc.
- **After:** All modules now under `bpmn_agent/` namespace package
  - `/bpmn_agent/core/`, `/bpmn_agent/knowledge/`, etc.

**Why:** This creates a proper Python package structure that setuptools can discover and install correctly.

### 2. Import Path Fixes
- Fixed all relative imports to use full `bpmn_agent.*` namespace
- Changed imports like:
  - `from models.knowledge_base import ...` → `from bpmn_agent.models.knowledge_base import ...`
  - `from knowledge import ...` → `from bpmn_agent.knowledge import ...`

### 3. pyproject.toml Update
```toml
[tool.setuptools]
packages = [
    "bpmn_agent",
    "bpmn_agent.core",
    "bpmn_agent.knowledge",
    "bpmn_agent.models",
    "bpmn_agent.stages",
    "bpmn_agent.tools",
    "bpmn_agent.validators",
    "bpmn_agent.agent",
    "bpmn_agent.api"
]
```

### 4. CLI Enforcement Guard (bpmn_agent/tools/cli.py)
```python
# ENFORCE LOCAL VENV USAGE
_PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
_VENV_PATH = _PROJECT_ROOT / ".venv"
if sys.prefix != str(_VENV_PATH):
    # Exit with error and show instructions
    sys.exit(1)
```

**Effect:** Any attempt to run the CLI without the local `.venv` will fail with a clear error message.

### 5. Documentation Files Created
- **VENV_USAGE.md** - User guide on proper venv usage
- **VENV_ENFORCEMENT_SETUP.md** - This file, explaining the setup

### 6. Configuration Files
- **.env** - Environment variables for venv paths
- **.bashrc.local** - Bash aliases to redirect to local venv

## How It Works

### Installation
```bash
# The package is installed in editable mode
cd /home/fsmw/dev/bpmn/src/bpmn-agent
.venv/bin/pip install -e .
```

This tells pip to:
1. Use the local `.venv` Python
2. Install bpmn-agent as editable (links to source, not copied)
3. Discover all packages under `bpmn_agent/` directory

### Execution
```bash
# ALL commands must use the local venv path
.venv/bin/python -c "..."
.venv/bin/python -m bpmn_agent.tools.cli
.venv/bin/pytest tests/
```

### Protection Layer
The CLI will refuse to run if system Python is used:
```
❌ ERROR: Not using local virtual environment!
   Current: /usr/bin/python3
   Expected: /home/fsmw/dev/bpmn/src/bpmn-agent/.venv/bin/python
```

## Key Directories

```
/home/fsmw/dev/bpmn/src/bpmn-agent/
├── .venv/                          # Virtual environment (MUST use this)
├── bpmn_agent/                     # Main package (organized by function)
│   ├── __init__.py
│   ├── core/                       # Core LLM and observability
│   ├── knowledge/                  # KB integration and patterns
│   ├── models/                     # Data models and schemas
│   ├── stages/                     # 5-stage pipeline
│   ├── tools/                      # CLI entry point (has enforcement)
│   ├── validators/                 # BPMN validation
│   ├── agent/                      # Main agent orchestrator
│   └── api/                        # FastAPI routes (future)
├── tests/                          # Test suite
├── pyproject.toml                  # Package config
├── docs/development/venv-usage.md  # Usage instructions
└── VENV_ENFORCEMENT_SETUP.md      # This file

## For OpenCode Agent

From now on, I will ALWAYS use:
```bash
/home/fsmw/dev/bpmn/src/bpmn-agent/.venv/bin/python [command]
```

Never:
```bash
python3 [command]
python [command]
/usr/bin/python [command]
```

The enforcement is in place at the CLI level, so any attempt will be caught and will exit with a helpful error message.
