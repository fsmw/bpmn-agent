# ⚠️ IMPORTANT: Virtual Environment Usage

## NEVER use system Python for this project

This project has a local virtual environment (`.venv`) that **MUST** be used for all Python operations.

### Why?
- The `.venv` is isolated and won't contaminate the system Python
- All project dependencies are installed in `.venv`
- Using system Python breaks the project and can damage your system environment

### How to Use (CORRECT)

Always prefix Python commands with the venv path:

```bash
# Activate the venv (from project root)
source .venv/bin/activate

# OR use directly for single commands
.venv/bin/python -m bpmn_agent.tools.cli process "your text"
.venv/bin/pip install package_name
.venv/bin/pytest tests/
```

### How NOT to Use (WRONG - DON'T DO THIS)

```bash
# ❌ WRONG - Never use system Python
python -m bpmn_agent.tools.cli process "your text"
python3 script.py
pip install package_name
```

### For Agent Usage

When I (OpenCode) run commands in this project, I will **ALWAYS** use:

```bash
/home/fsmw/dev/bpmn/src/bpmn-agent/.venv/bin/python [command]
```

This is enforced in the CLI with a guard that will exit with an error if system Python is detected.

### Quick Reference

| Task | Command |
|------|---------|
| Run CLI | `.venv/bin/python -m bpmn_agent.tools.cli [cmd]` |
| Run tests | `.venv/bin/pytest tests/` |
| Install deps | `.venv/bin/pip install -r requirements.txt` |
| Activate shell | `source .venv/bin/activate` |

