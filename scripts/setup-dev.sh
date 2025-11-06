#!/bin/bash
# Setup script for bpmn-agent development environment
# This script configures the local development environment with CI/CD tools

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🚀 Setting up bpmn-agent development environment..."
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo "📋 Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}❌ Python 3.10+ required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python $PYTHON_VERSION detected${NC}"

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}⚠️  Not in a virtual environment. Creating one...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
    echo -e "${YELLOW}Please activate it: source venv/bin/activate${NC}"
    echo -e "${YELLOW}Then run this script again.${NC}"
    exit 0
fi

echo -e "${GREEN}✅ Virtual environment active: $VIRTUAL_ENV${NC}"

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install project in development mode
echo ""
echo "📦 Installing project dependencies..."
cd "$PROJECT_ROOT"
pip install -e ".[dev]"

# Install pre-commit
echo ""
echo "🔧 Installing pre-commit hooks..."
if ! command -v pre-commit &> /dev/null; then
    pip install pre-commit
fi

pre-commit install
pre-commit install --hook-type pre-push

echo -e "${GREEN}✅ Pre-commit hooks installed${NC}"

# Verify installation
echo ""
echo "🧪 Running quick test to verify setup..."
if pytest -m "unit and not llm and not slow" --tb=short -q > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Tests pass${NC}"
else
    echo -e "${YELLOW}⚠️  Some tests failed, but setup is complete${NC}"
    echo "   Run 'pytest -m \"unit and not llm\"' to see details"
fi

# Check coverage
echo ""
echo "📊 Checking code coverage..."
if pytest -m "unit and not llm and not slow" --cov=bpmn_agent --cov-report=term-missing -q > /dev/null 2>&1; then
    COVERAGE=$(coverage report --format=total 2>/dev/null | tail -1 | awk '{print $NF}' | sed 's/%//' || echo "0")
    if [ ! -z "$COVERAGE" ] && [ "$COVERAGE" != "0" ]; then
        echo -e "${GREEN}✅ Current coverage: ${COVERAGE}%${NC}"
        if (( $(echo "$COVERAGE < 75" | bc -l 2>/dev/null || echo "0") )); then
            echo -e "${YELLOW}⚠️  Coverage is below 75% threshold${NC}"
        fi
    fi
fi

echo ""
echo -e "${GREEN}✅ Development environment ready!${NC}"
echo ""
echo "📝 Next steps:"
echo "   1. Create feature branch: git checkout -b feature/your-feature-name"
echo "   2. Make changes and commit (pre-commit will run automatically)"
echo "   3. Push and create PR for review"
echo ""
echo "🔍 Useful commands:"
echo "   - Run tests: pytest -m \"unit and integration and not llm\""
echo "   - Check coverage: pytest --cov=bpmn_agent --cov-report=html"
echo "   - Format code: black ."
echo "   - Lint code: ruff check ."
echo "   - Type check: mypy bpmn_agent/"
echo ""
echo "📚 Documentation:"
echo "   - See docs/development/devops-setup.md for CI/CD workflow details"
echo "   - See README.md for project overview"
