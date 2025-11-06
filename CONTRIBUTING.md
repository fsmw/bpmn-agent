# Contributing Guide

Thank you for your interest in contributing to `bpmn-agent`! This document describes our development process and how to work with Issues, Branches, Pull Requests, and CI/CD.

## 📋 Table of Contents

- [Development Workflow](#development-workflow)
- [Creating an Issue](#creating-an-issue)
- [Creating a Branch](#creating-a-branch)
- [Local Development](#local-development)
- [Creating a Pull Request](#creating-a-pull-request)
- [CI/CD and Quality Gates](#cicd-and-quality-gates)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Commits and Messages](#commits-and-messages)

---

## 🔄 Development Workflow

Our process follows a structured flow: **Issue → Branch → PR → CI → Merge**

```
1. Create Issue          → Document what will be done
2. Create Branch          → Work on a specific branch
3. Local Development      → Pre-commit hooks ensure quality
4. Create Pull Request    → Link PR to Issue
5. Automatic CI/CD        → Tests, linting, coverage
6. Code Review           → Review and approval
7. Merge                 → Automatic Issue closure
```

### Why this workflow?

- ✅ **Complete traceability**: Every change is linked to an issue
- ✅ **Guaranteed quality**: CI/CD prevents broken code in `main`
- ✅ **Automatic documentation**: Issues document decisions and context
- ✅ **Effective collaboration**: More informed code review
- ✅ **Metrics**: Visibility into project progress

---

## 📝 Creating an Issue

Before starting to code, **always create an Issue** that describes the work to be done.

### When to create an Issue

- ✅ New functionality
- ✅ Bug fixes
- ✅ Documentation improvements
- ✅ Refactoring
- ✅ Infrastructure tasks

### How to create an Issue

#### Option 1: From GitHub CLI

```bash
cd src/bpmn-agent
gh issue create --title "Descriptive title" --body "Detailed description" --label "enhancement"
```

#### Option 2: From GitHub Web

1. Go to https://github.com/fsmw/bpmn-agent/issues/new
2. Use the appropriate template (if available)
3. Complete title, description, labels

### Recommended Issue Template

```markdown
## 🎯 Objective
Clear description of the objective and context.

## 📋 Tasks
- [ ] Task 1
- [ ] Task 2
- [ ] Task 3

## 📁 Files to Modify
- `path/to/file.py`
- `tests/test_file.py`

## ✅ Success Criteria
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Tests pass
- [ ] Coverage > 75%

## 🔗 Related
- Issue: #X
- Plan: `FILE.md`

## 📝 Notes
Additional notes, considerations, etc.
```

### Available Labels

**By Type:**
- `bug` - Bug fixes
- `enhancement` - New functionality
- `documentation` - Documentation changes
- `refactoring` - Refactoring

**By Phase/Component:**
- `phase1`, `phase2`, `phase3`, `phase4`, `phase5`
- `ci-cd` - CI/CD improvements
- `testing` - Tests
- `validation` - Validation

**By Priority:**
- `priority:high` - High priority
- `priority:medium` - Medium priority
- `priority:low` - Low priority

---

## 🌿 Creating a Branch

Once you have an Issue, create a branch to work on it.

### Option 1: Use GitHub CLI (Recommended)

```bash
cd src/bpmn-agent
gh issue develop <ISSUE_NUMBER>
```

This automatically:
- Creates a branch linked to the issue (e.g., `issue-2-integrate-phase4-validation`)
- Checks out the branch
- Links the branch to the issue on GitHub

### Option 2: Create Manually

```bash
cd src/bpmn-agent
git checkout main
git pull origin main
git checkout -b feature/issue-<NUMBER>-short-description
```

### Branch Naming Convention

```
feature/issue-<NUMBER>-short-description
bugfix/issue-<NUMBER>-short-description
docs/issue-<NUMBER>-short-description
refactor/issue-<NUMBER>-short-description
```

Examples:
- `feature/issue-2-integrate-phase4-validation`
- `bugfix/issue-5-fix-xsd-validation-error`
- `docs/issue-8-update-readme`

---

## 💻 Local Development

### Initial Setup

```bash
cd src/bpmn-agent

# Automated setup (recommended)
bash scripts/setup-dev.sh

# Or manually
pip install -e ".[dev]"
pre-commit install
pre-commit install --hook-type pre-push
```

### Pre-commit Hooks

Hooks run automatically on each commit:

- ✅ **black** - Code formatting
- ✅ **ruff** - Linting and auto-fix
- ✅ **mypy** - Type checking
- ✅ **pytest-unit** - Fast unit tests
- ✅ **pytest-integration** - Integration tests
- ✅ **coverage-check** - Coverage verification (warning if < 75%)

### Run Hooks Manually

```bash
# All files
pre-commit run --all-files

# Only staged files
pre-commit run

# Specific hook
pre-commit run black --all-files
```

### Iterative Development

```bash
# 1. Make changes
vim bpmn_agent/file.py

# 2. Add changes
git add bpmn_agent/file.py

# 3. Commit (pre-commit runs automatically)
git commit -m "feat: add feature X (refs #<ISSUE_NUMBER>)"

# 4. If pre-commit fails, fix and commit again
# 5. Repeat until work is complete
```

---

## 🔀 Creating a Pull Request

When your work is ready, create a Pull Request.

### Prerequisites

- ✅ All tests pass locally
- ✅ Pre-commit hooks pass
- ✅ Coverage > 75%
- ✅ Code formatted and linted
- ✅ Commits with descriptive messages

### Create PR from Terminal

```bash
cd src/bpmn-agent

# Push branch
git push origin feature/issue-<NUMBER>-description

# Create PR linked to issue
gh pr create --title "feat: Descriptive title" --body "Fixes #<ISSUE_NUMBER>

## Changes
- Change 1
- Change 2
- Change 3

## Testing
- [x] Unit tests pass
- [x] Integration tests pass
- [x] Coverage > 75%

## Checklist
- [x] Code follows project style
- [x] Documentation updated if necessary
- [x] No breaking changes (or documented)

Fixes #<ISSUE_NUMBER>"
```

### Create PR from GitHub Web

1. Push your branch: `git push origin feature/issue-<NUMBER>-description`
2. Go to https://github.com/fsmw/bpmn-agent/compare
3. Select your branch
4. Complete the PR form
5. **Important**: Include `Fixes #<ISSUE_NUMBER>` in the description

### PR Template

```markdown
## Description
Brief description of changes.

Fixes #<ISSUE_NUMBER>

## Changes
- Change 1
- Change 2
- Change 3

## Testing
- [x] Unit tests pass
- [x] Integration tests pass
- [x] Coverage > 75%
- [x] Tested locally

## Checklist
- [x] Code follows project style (black, ruff)
- [x] Type checking passes (mypy)
- [x] Documentation updated if necessary
- [x] No breaking changes (or documented)
- [x] Pre-commit hooks pass

## Screenshots (if applicable)
...

## Additional Notes
...
```

### Keywords to Close Issues

Include one of these phrases in the PR to automatically close the issue:

- `Fixes #<NUMBER>` - Closes the issue when merged
- `Closes #<NUMBER>` - Same as Fixes
- `Resolves #<NUMBER>` - Same as Fixes
- `Related to #<NUMBER>` - Only links, doesn't close

---

## ✅ CI/CD and Quality Gates

### What happens when you create a PR?

GitHub Actions automatically runs:

#### 1. CI Workflow (`ci.yml`)
- ✅ Tests on Python 3.10, 3.11, 3.12
- ✅ Linting (black, ruff)
- ✅ Type checking (mypy)
- ✅ Unit tests
- ✅ Integration tests
- ✅ Coverage check (must be > 75%)

#### 2. Quality Workflow (`quality.yml`)
- ✅ Security scanning (Bandit)
- ✅ Dependency vulnerabilities (Safety)
- ✅ Code complexity (xenon, radon)
- ✅ License compliance
- ✅ SBOM generation

#### 3. Release Workflow (`release.yml`)
- Only runs when a `v*` tag is created

### Check CI Status

```bash
# View checks for current PR
gh pr checks

# View recent runs
gh run list

# View details of a specific run
gh run view <RUN_ID>
```

### What to do if CI fails?

1. **Review logs**: `gh run view <RUN_ID>` or from GitHub web
2. **Reproduce locally**: Run the same command that failed
3. **Fix the problem**
4. **Push new commit**: CI will run automatically

### Quality Gates (REQUIRED)

**The PR CANNOT be merged if:**

- ❌ Tests fail
- ❌ Coverage < 75%
- ❌ Linting fails
- ❌ Type checking fails
- ❌ Critical security issues

**The PR can be merged when:**

- ✅ All checks pass
- ✅ At least 1 review approval (if configured)
- ✅ Branch is up to date with `main`

---

## 📏 Code Standards

### Formatting

We use **black** with standard configuration:

```bash
black bpmn_agent/
```

### Linting

We use **ruff** for fast linting:

```bash
ruff check bpmn_agent/
ruff check --fix bpmn_agent/  # Auto-fix
```

### Type Checking

We use **mypy** for type safety:

```bash
mypy bpmn_agent/ --ignore-missing-imports
```

### Imports

Import order (ruff orders them automatically):

```python
# 1. Standard library
import os
from typing import List

# 2. Third-party
from pydantic import BaseModel
import requests

# 3. Local
from bpmn_agent.core import LLMClient
from bpmn_agent.models import Process
```

---

## 🧪 Testing

### Test Structure

```
tests/
├── unit/              # Fast unit tests
├── integration/      # Integration tests
├── test_*.py         # Tests by module
└── conftest.py       # Shared fixtures
```

### Run Tests

```bash
# All tests
pytest

# Only unit tests (fast)
pytest -m "unit and not llm and not slow"

# Only integration
pytest -m "integration and not llm"

# With coverage
pytest --cov=bpmn_agent --cov-report=term --cov-report=html

# Specific file
pytest tests/test_file.py

# Specific test
pytest tests/test_file.py::test_function
```

### Available Markers

- `@pytest.mark.unit` - Unit test
- `@pytest.mark.integration` - Integration test
- `@pytest.mark.llm` - Requires LLM (skipped in CI by default)
- `@pytest.mark.slow` - Slow test
- `@pytest.mark.e2e` - End-to-end test

### Coverage Requirements

- **Minimum**: 75% (line and branch)
- **Target**: 80%+
- **CI fails** if coverage < 75%

### Writing Tests

```python
import pytest
from bpmn_agent.core import LLMClient

@pytest.mark.unit
def test_basic_function():
    """Simple unit test."""
    result = function_to_test()
    assert result == expected

@pytest.mark.integration
async def test_integration():
    """Integration test."""
    client = LLMClient(...)
    result = await client.call(...)
    assert result is not None
```

---

## 📝 Commits and Messages

### Commit Convention

We use **Conventional Commits**:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Commit Types

- `feat`: New functionality
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting (no code changes)
- `refactor`: Refactoring
- `test`: Tests
- `chore`: Maintenance tasks
- `ci`: CI/CD changes
- `perf`: Performance improvements

### Examples

```bash
# Feature
git commit -m "feat(validation): add XSD validation (refs #2)"

# Bug fix
git commit -m "fix(xsd): fix parsing error (fixes #5)"

# Documentation
git commit -m "docs(readme): update installation instructions"

# Refactor
git commit -m "refactor(orchestrator): simplify validation logic"
```

### Linking Commits to Issues

```bash
# Closes the issue when PR is merged
git commit -m "feat: implement X (fixes #2)"

# Only references the issue
git commit -m "feat: implement X (refs #2)"
```

---

## 🔍 Code Review

### Review Process

1. **Create PR** → CI runs automatically
2. **Wait for approvals** → At least 1 reviewer (if configured)
3. **Address comments** → Make changes and push new commits
4. **Merge** → When all checks pass and there's approval

### What to Look for in a Review

- ✅ Code follows project standards
- ✅ Tests cover the changes
- ✅ No regressions
- ✅ Documentation updated
- ✅ Acceptable performance
- ✅ Security considered

---

## 🚀 Merge

### Merge Options

```bash
# Squash merge (recommended)
gh pr merge <PR_NUMBER> --squash --delete-branch

# Merge commit
gh pr merge <PR_NUMBER> --merge --delete-branch

# Rebase merge
gh pr merge <PR_NUMBER> --rebase --delete-branch
```

### After Merge

- ✅ Issue closes automatically (if PR has "Fixes #X")
- ✅ Branch is deleted automatically (if using `--delete-branch`)
- ✅ CI runs on `main` to verify everything still works

---

## 📚 Additional Resources

- **[DevOps Setup](docs/development/devops-setup.md)** - Detailed CI/CD setup
- **[GitHub Workflow](docs/development/github-workflow.md)** - Workflow best practices
- **README.md** - General project documentation

---

## ❓ Frequently Asked Questions

### Can I skip pre-commit hooks?

Yes, with `git commit --no-verify`, but **NOT recommended**. Hooks prevent problems before they reach CI.

### What if my PR fails in CI but passes locally?

1. Verify tool versions (black, ruff, mypy, pytest)
2. Verify Python version (CI uses 3.10, 3.11, 3.12)
3. Run `pre-commit run --all-files` locally
4. Review CI logs for specific details

### Can I create a PR without an Issue?

Technically yes, but **we recommend creating an Issue first** to document context and allow discussion before implementing.

### How do I update my branch with changes from main?

```bash
git checkout feature/issue-<NUMBER>-description
git fetch origin
git merge origin/main
# Or use rebase if you prefer
git rebase origin/main
```

---

## 🎯 Quick Summary

```bash
# 1. Create Issue
gh issue create --title "..." --body "..." --label "enhancement"

# 2. Create Branch
gh issue develop <ISSUE_NUMBER>

# 3. Development
# ... make changes ...
git add .
git commit -m "feat: ... (refs #<NUMBER>)"
git push origin feature/issue-<NUMBER>-description

# 4. Create PR
gh pr create --title "..." --body "Fixes #<NUMBER>"

# 5. Wait for CI and Review
gh pr checks

# 6. Merge (when ready)
gh pr merge <PR_NUMBER> --squash --delete-branch
```

---

**Thank you for contributing to bpmn-agent!** 🚀
