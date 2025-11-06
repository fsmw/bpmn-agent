#!/bin/bash
# Verification script for venv enforcement setup

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_BIN="${PROJECT_ROOT}/.venv/bin"

echo "=== BPMN Agent Virtual Environment Verification ==="
echo ""

# Check 1: .venv exists
if [ -d "${PROJECT_ROOT}/.venv" ]; then
    echo "✓ .venv directory exists"
else
    echo "✗ .venv directory not found"
    exit 1
fi

# Check 2: Python in venv works
if "${VENV_BIN}/python" --version &>/dev/null; then
    echo "✓ .venv/bin/python is executable"
    "${VENV_BIN}/python" --version
else
    echo "✗ .venv/bin/python not working"
    exit 1
fi

# Check 3: Package structure
echo ""
echo "Package Structure:"
for dir in core knowledge models stages tools validators agent; do
    if [ -d "${PROJECT_ROOT}/bpmn_agent/$dir" ]; then
        echo "  ✓ bpmn_agent/$dir"
    else
        echo "  ✗ bpmn_agent/$dir MISSING"
    fi
done

# Check 4: Imports work
echo ""
echo "Testing imports..."
"${VENV_BIN}/python" -c "
from bpmn_agent.core.llm_client import LLMClientFactory
from bpmn_agent.models.bpmn_elements import Task
from bpmn_agent.knowledge.domain_classifier import DomainClassifier
print('✓ All imports working')
" || exit 1

# Check 5: CLI enforcement
echo ""
echo "Testing CLI enforcement..."
"${VENV_BIN}/python" -m bpmn_agent.tools.cli --help > /dev/null 2>&1 && \
    echo "✓ CLI enforcement working" || echo "✗ CLI failed"

# Check 6: Configuration files
echo ""
echo "Configuration Files:"
for file in .env .bashrc.local VENV_USAGE.md VENV_ENFORCEMENT_SETUP.md; do
    if [ -f "${PROJECT_ROOT}/$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file MISSING"
    fi
done

echo ""
echo "=== Verification Complete ==="
echo ""
echo "To run commands, always use:"
echo "  ${VENV_BIN}/python [command]"
