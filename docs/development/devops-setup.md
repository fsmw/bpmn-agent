# README for DevOps Integration

This document explains how to set up the development workflow for the bpmn-agent project using GitHub Actions and pre-commit hooks.

## Quick Setup

### Automated Setup (Recommended)

Run the setup script to configure everything automatically:
```bash
cd src/bpmn-agent
bash scripts/setup-dev.sh
```

This script will:
- ✅ Check Python version (3.10+)
- ✅ Create virtual environment if needed
- ✅ Install all dependencies
- ✅ Install pre-commit hooks
- ✅ Run quick tests to verify setup
- ✅ Check code coverage

### Manual Setup

#### 1. Pre-commit Hooks (Local Development Quality)

Install pre-commit hooks:
```bash
cd src/bpmn-agent
pre-commit install
pre-commit install --hook-type pre-push
```

### 2. GitHub Secrets Required

Set these in your GitHub repositorySettings > Secrets and variables > Actions:

```
# For LLM integration tests
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1  # optional
LLM_API_KEY=sk-...  # alternative to OPENAI_API_KEY
LLM_BASE_URL=https://api.llm-provider.com/v1  # optional

# For publishing
PYPI_API_TOKEN=pypi-...

# Optional for Docker publishing
DOCKER_USERNAME=your-docker-hub-username
DOCKER_TOKEN=docker-hub-token
```

### 3. Branch Protection Rules

In GitHub repository Settings > Branches > Branch protection rules:

For `main` branch:
- ✅ Require pull request reviews before merging (2 reviewers)
- ✅ Require status checks to pass before merging
  - ✅ CI (test)  
  - ✅ Code Quality
  - ✅ Security Scan
  - ✅ Lint/Format (optional)
- ✅ Require branches to be up to date before merging
- ✅ Require conversation resolution before merging

## Development Workflow

### Formal Workflow: Issue → Branch → PR → CI → Merge

We follow a structured workflow using GitHub Issues, Branches, and Pull Requests:

```
1. Create Issue          → Document what needs to be done
2. Create Branch          → Work on a specific branch
3. Develop Locally        → Pre-commit hooks ensure quality
4. Create Pull Request    → Link PR to Issue
5. CI/CD Automatic        → Tests, linting, coverage
6. Code Review            → Review and approval
7. Merge                  → Automatic issue closure
```

### Step-by-Step Workflow

#### Step 1: Create an Issue

```bash
# From terminal
cd src/bpmn-agent
gh issue create --title "Descriptive title" --body "Description" --label "enhancement"

# Or from GitHub web: https://github.com/fsmw/bpmn-agent/issues/new
```

**Why Issues First?**
- ✅ Documents context and requirements
- ✅ Enables discussion before implementation
- ✅ Provides traceability (Issue → Branch → PR → Merge)
- ✅ Automatically closes when PR is merged

#### Step 2: Create Branch from Issue

```bash
# Recommended: Use GitHub CLI (automatically links branch to issue)
gh issue develop <ISSUE_NUMBER>

# Or manually
git checkout main
git pull origin main
git checkout -b feature/issue-<NUMBER>-descripcion-corta
```

#### Step 3: Develop Locally

```bash
# Work on your changes
# ... edit files ...

# Pre-commit checks run automatically when you commit
git add .
git commit -m "feat: add new entity extraction logic (refs #<ISSUE_NUMBER>)"

# Pre-commit will run:
# - black (formatting)
# - ruff (linting + auto-fix)  
# - mypy (type checking)
# - pytest-unit (fast tests)
# - pytest-integration (component tests)
# - coverage-check (warns if < 75%)

# If any check fails, fix it and try again
git commit -m "feat: add new entity extraction logic (fixed tests)"
```

#### Step 4: Create Pull Request

```bash
# Push branch
git push origin feature/issue-<NUMBER>-descripcion

# Create PR linked to issue (use "Fixes #<NUMBER>" to auto-close issue)
gh pr create --title "feat: Título descriptivo" --body "Fixes #<ISSUE_NUMBER>

## Cambios
- Cambio 1
- Cambio 2

## Testing
- [x] Tests pasan
- [x] Coverage > 75%

Fixes #<ISSUE_NUMBER>"
```

**Important**: Include `Fixes #<ISSUE_NUMBER>` in PR description to automatically close the issue when merged.

#### Step 5: CI/CD Runs Automatically

When you create a PR, GitHub Actions automatically:
- ✅ Runs tests on Python 3.10, 3.11, 3.12
- ✅ Checks formatting, linting, type checking
- ✅ Verifies coverage > 75%
- ✅ Runs security scans

**PR cannot be merged until all checks pass.**

#### Step 6: Code Review

- Wait for at least 1 approval (if branch protection is configured)
- Address review comments
- Push new commits if needed

#### Step 7: Merge

```bash
# When ready, merge PR (squash recommended)
gh pr merge <PR_NUMBER> --squash --delete-branch
```

**Result:**
- ✅ Issue automatically closes (if PR had "Fixes #X")
- ✅ Branch automatically deleted
- ✅ All changes merged to `main`

### See Also

- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Complete contribution guide
- **[GitHub Workflow Guide](github-workflow.md)** - Best practices
- **[WORKFLOW_ISSUES_SETUP.md](../WORKFLOW_ISSUES_SETUP.md)** - Issues setup guide

### What Happens in CI

When you create PR, these GitHub Actions run:

1. **CI Workflow** (`ci.yml`)
   - Tests across Python 3.10, 3.11, 3.12
   - Runs unit + integration tests (no LLM tests by default)
   - Checks formatting, linting, type checking
   - Uploads coverage to Codecov

2. **Security Scan** (`quality.yml`)
   - Bandit security analysis
   - Safety dependency check
   - License compliance
   - Code quality metrics

3. **LLM Integration Tests** (`llm.yml`) - Only if:
   - Manual trigger (workflow_dispatch)
   - Daily schedule (cron)
   - PR touches core LLM/stage files

### PR Template (Use in PR Description)

```markdown
## Descripción
Breve descripción de los cambios.

Fixes #<ISSUE_NUMBER>

## Cambios
- Cambio 1
- Cambio 2
- Cambio 3

## Testing
- [x] Tests unitarios pasan
- [x] Tests de integración pasan
- [x] Coverage > 75%
- [x] Probado localmente

## Checklist
- [x] Código sigue estilo del proyecto (black, ruff)
- [x] Type checking pasa (mypy)
- [x] Documentación actualizada si es necesario
- [x] Sin breaking changes (o documentados)
- [x] Pre-commit hooks pasan

## Testing Local

```bash
# Quick test before pushing
pytest -m "unit and integration and not llm and not slow"

# Full test suite
pytest --cov=bpmn_agent --cov-report=term
```
```

**Important**: Always include `Fixes #<ISSUE_NUMBER>` to automatically close the issue when the PR is merged.

### Release Process

```bash
# 1. Ensure all tests pass and version bumped in pyproject.toml
git checkout main
git pull origin main

# 2. Tag the release
git tag v0.5.0  # version matching pyproject.toml
git push origin v0.5.0

# 3. Release workflow runs automatically:
# - Builds package
# - Runs tests
# - Creates GitHub release
# - Publishes to PyPI
# - Builds Docker image (optional)
```

## File Modifications for Project Integration

### 1. pyproject.toml Updates

Add to your `pyproject.toml` (already in current version):

```toml
[tool.bandit]
exclude_dirs = ["tests"]
skips = ["B101"]  # Skip assert_usage test

[tool.cyclonedx]
output_file = "sbom.json"

[tool.towncrier]
filename = "CHANGELOG.md"
directory = "news"
title_format = "{name} v{version} ({project_date})"
```

### 2. Environment Setup Scripts

Create scripts/setup-dev.sh:
```bash
#!/bin/bash

set -e

echo "🚀 Setting up bpmn-agent development environment..."

# Install uv if not available
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    pip install uv
fi

# Install pre-commit
echo "Installing pre-commit hooks..."
pre-commit install
pre-commit install --hook-type pre-push

# Install development dependencies
echo "Installing development dependencies..."
pip install -e ".[dev]"

# Verify installation
echo "🧪 Running quick test to verify setup..."
pytest -m "unit and not llm and not slow" --tb=short

echo "✅ Development environment ready!"
echo ""
echo "Next steps:"
echo "1. Create feature branch: git checkout -b feature/your-name"
echo "2. Make changes and commit (pre-commit will run automatically)"
echo "3. Push and create PR for review"
```

## Troubleshooting Common Issues

### Pre-commit Issues

**Problem**: Pre-commit hook fails on mypy
```bash
# Solution: Update mypy configuration or add type imports
# In pyproject.toml, adjust mypy ignore_missing_imports for problematic modules
```

**Problem**: Tests timeout in CI
```bash
# Solution: Check test markers and ensure no LLM calls in unit tests
pytest -m "unit and not llm" --timeout=30
```

### GitHub Actions Issues

**Problem**: Tests fail due to missing Ollama service
- CI handles this by skipping LLM tests unless secrets are available
- Unit/integration tests don't require real LLM calls (use mocks)

**Problem**: Repository doesn't have access to secrets in fork PRs
- Only repository secrets are available to PRs from the same repo
- For fork PRs, LLM integration tests will be skipped automatically

### Branch Protection Issues

**Problem**: PR can't merge due to missing status checks
1. Ensure CI workflows run successfully
2. Check that branch protection rules only require checks that actually exist
3. Use status check names that match your workflow names

## Performance Considerations

### Test Execution Time

Target test execution times:
- **Unit tests**: < 2 minutes
- **Integration tests**: < 5 minutes  
- **Full CI suite on PR**: < 15 minutes
- **LLM integration tests**: < 20 minutes (optional)

### Optimization Tips

1. **Use pytest-xdist** for parallel testing:
   ```bash
   pip install pytest-xdist
   pytest -n auto -m "unit"
   ```

2. **Cache LLM responses** in integration tests
   ```python
   @pytest.fixture
   def cached_llm_client():
       # Return client with response caching
   ```

3. **Use markers wisely** - don't mark tests as `slow` unless necessary

## Next Steps

### Week 1: Basic Setup
- [ ] Install pre-commit hooks locally
- [ ] Set up branch protection on main
- [ ] Test with demo PR
- [ ] Configure required GitHub secrets

### Week 2: Advanced Features  
- [ ] Set up Codecov coverage tracking
- [ ] Configure security scanning notifications
- [ ] Add performance benchmarking
- [ ] Set up automated dependency updates

### Week 3: Monitoring & Alerts
- [ ] Set up GitHub issue templates
- [ ] Configure Slack/email notifications for build failures
- [ ] Add dependency vulnerability scanning
- [ ] Set up automated documentation deployment

## Support

For issues with this setup:
1. Check workflow logs in GitHub Actions tab
2. Look at pre-commit output locally
3. Verify all secrets are properly configured
4. Check that your local environment matches CI (Python version, dependencies)