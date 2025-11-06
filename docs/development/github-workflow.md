# GitHub Workflow Guide for BPMN-Agent

This guide explains the complete development workflow using GitHub Actions, pre-commit hooks, and best practices for collaborative development.

## 🎯 Overview

Our workflow has three quality gates:

1. **Pre-commit Hooks** (Local): Fast checks before you commit
2. **Pull Request CI** (GitHub): Full testing on PR creation
3. **Merge Rules** (Branch Protection): Enforces quality before merging

## 📋 Required One-Time Setup

### 1. Local Environment Setup

```bash
cd src/bpmn-agent

# Install pre-commit hooks (runs locally on every commit)
pre-commit install
pre-commit install --hook-type pre-push

# Verify setup
pre-commit run --all-files  # Should pass all checks
```

### 2. GitHub Repository Settings

Go to your repository → **Settings** → **Branches** → **Branch protection rules**:

**Rule for `main` branch:**
- ✅ **Require pull request reviews before merging** (2 reviewers)
- ✅ **Require status checks to pass before merging**
  - ✅ **CI (test)**  
  - ✅ **Code Quality**
  - ✅ **Security Scan**
- ✅ **Require branches to be up to date before merging**
- ✅ **Require conversation resolution before merging**

### 3. GitHub Secrets Setup

**Repository Settings** → **Secrets and variables** → **Actions**:

```
LLM Testing:
OPENAI_API_KEY=sk-your-openai-key
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-llm-provider-key
LLM_BASE_URL=https://your-llm-provider.com/v1

Package Publishing:
PYPI_API_TOKEN=pypi-your-pypi-token

Optional for Docker:
DOCKER_USERNAME=your-docker-username
DOCKER_TOKEN=your-docker-token
```

## 🔄 Daily Development Workflow

### Step 1: Start Feature Work

```bash
# Always start from updated main
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-description

# Example branch naming:
# feature/entity-extraction-improvement
# fix/llm-client-connection-retry
# docs/api-documentation-update
```

### Step 2: Development Process

```bash
# Work on your changes
# ... edit files ...

# Pre-commit hooks will run automatically when you commit
git add .
git commit -m "feat: add parallel gateway detection logic"

# Pre-commit runs these checks:
# 1. black --check . (code formatting)
# 2. ruff check . (linting + auto-fix)
# 3. mypy bpmn_agent/ (type checking)
# 4. pytest -m "unit and not llm and not slow" (fast tests)
# 5. pytest -m "integration and not llm and not slow" (component tests)

# If any check fails, fix and try again:
git commit -m "feat: add parallel gateway detection logic (fix tests)"
```

### Step 3: Quality Gates Before Push

```bash
# Optional: Run full test suite before pushing (catch issues early)
make test-fast    # or:
pytest -m "unit and integration and not llm and not slow" --cov=bpmn_agent

# Push feature branch
git push origin feature/your-feature-description
```

### Step 4: Create Pull Request

1. Go to GitHub → Open new PR from your branch to `main`
2. Fill in PR description (use template below)
3. **Automated CI runs immediately:**

**CI Workflow (ci.yml)**:
- ✅ Tests on Python 3.10, 3.11, 3.12
- ✅ Unit + Integration tests (no LLM calls)
- ✅ Formatting, linting, type checking
- ✅ Coverage upload to Codecov

**Quality Workflow (quality.yml)**:
- ✅ Security scanning (Bandit + Safety)
- ✅ Code quality metrics
- ✅ License compliance

**Optional LLM Tests (llm.yml)**:
- ℹ️ Only runs if secrets available or manual trigger
- ℹ️ Tests real LLM integrations (Ollama, OpenAI)

### Step 5: Code Review & Merge

Your PR must pass all status checks before merging:

```
✅ CI (test)
✅ Code Quality  
✅ Security Scan
✅ 2/2 reviewers approved
✅ Up to date with main branch
✅ Conversation resolved
```

## 📝 Pull Request Template

```markdown
## Description
Brief description of what this PR changes and why.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## How Has This Been Tested?
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing with example processes
- [ ] Performance testing (if applicable)

## PR Checklist
- [ ] Code follows project style guidelines (pre-commit passes)
- [ ] Self-review of the code
- [ ] Code is well-documented with TODOs resolved
- [ ] No breaking changes without documentation
- [ ] Environment variables documented if new
- [ ] Error messages are user-friendly
- [ ] Edge cases considered and tested

## Testing Commands for Reviewers
```bash
# Quick validation (runs < 2 minutes)
pytest -m "unit and integration and not llm" --tb=short

# Full test suite (if you have LLM set up)
pytest -m "not slow" --tb=short
```

## Screenshots (if applicable)
Add screenshots for UI changes or test output.

## Additional Notes
Any additional context for the reviewers.
```

## 🚀 Release Workflow

When ready to release:

```bash
# 1. Ensure all changes are in main
git checkout main
git pull origin main

# 2. Update version in pyproject.toml
# Change: version = "0.5.0" → version = "0.5.1"

# 3. Commit version bump
git commit -m "chore: bump version to 0.5.1" 
git push origin main

# 4. Create and push tag
git tag v0.5.1
git push origin v0.5.1

# 5. Release workflow automatically:
# - Builds package
# - Runs comprehensive tests
# - Creates GitHub release with changelog
# - Publishes to PyPI
# - Builds Docker image
```

## 🧪 Development Testing Commands

### Quick Validation (Local)

```bash
# Fast feedback (under 2 minutes)
pytest -m "unit and not llm" --tb=short

# Component integration (under 5 minutes) 
pytest -m "unit and integration and not llm" --tb=short

# Full mock tests (under 10 minutes)
pytest -m "not llm and not slow" --tb=short --cov=bpmn_agent
```

### LLM Testing (Requires Setup)

```bash
# If you have Ollama running locally:
export LLM_PROVIDER=ollama
export LLM_BASE_URL=http://localhost:11434
export LLM_MODEL=mistral
pytest -m "llm" --tb=short

# If you have OpenAI API key:
export LLM_PROVIDER=openai
export LLM_API_KEY=sk-your-key
pytest -m "llm" --tb=short
```

### Code Quality Commands

```bash
# Individual code quality checks
black --check .              # Formatting
ruff check .                  # Linting
mypy bpmn_agent/             # Type checking
bandit -r bpmn_agent/        # Security scanning
```

## 🐛 Common Issues & Solutions

### Pre-commit Hook Issues

**Problem**: Black formatting fails
```bash
# Auto-fix:
black .
git commit -m "fix: apply black formatting"
```

**Problem**: MyPy type checking fails
```bash
# Check specific file:
mypy bpmn_agent/core/llm_client.py

# Common fixes:
# 1. Add proper imports: from typing import List, Optional
# 2. Add type hints to function signatures
# 3. Add "# type: ignore" for external dependencies
```

**Problem**: pytest tests fail
```bash
# Run with verbose output to see what's failing:
pytest -m "integration" -v --tb=long

# Run specific test file:
pytest tests/test_agent.py -v

# Common fixes:
# 1. Check imports and fixtures
# 2. Verify test marks match test behavior
# 3. Check that mocks are properly configured
```

### CI/CD Issues

**Problem**: GitHub Actions fails on type checking
- Solution: Ensure your local Python version matches CI (3.11 is primary)
- Check that all dependencies are installed in CI

**Problem**: LLM integration tests fail in PR
- This is expected - LLM tests only run with secrets or manual trigger
- Use mocked tests for regular PR validation

**Problem**: Security scan shows vulnerabilities
- Solution: Update problematic dependencies: `pip install --upgrade package-name`
- For acceptable risks, document in PR description

## 📊 Performance Benchmarks

The CI/CD pipeline is optimized for speed:

- **Pre-commit**: < 30 seconds (unit tests)
- **PR CI**: < 10 minutes (full matrix)
- **LLM Tests**: < 15 minutes (optional)
- **Security Scan**: < 5 minutes
- **Total PR validation**: < 20 minutes

### Optimization Tips

1. **Use pytest markers wisely** - mark truly slow tests as `slow`
2. **Mock external services** - don't make real API calls in unit tests
3. **Run tests in parallel** locally with `pytest -n auto`
4. **Cache dependencies** - GitHub Actions automatically caches pip packages

## 🔐 Security Best Practices

### Branch Protection Rules

- **Require PR reviews**: Prevent direct commits to main
- **Require status checks**: Ensure CI passes
- **Require up-to-date branches**: Prevent stale merges

### Secret Management

- Never commit API keys or secrets
- Use GitHub secrets for any authentication
- Test with mocked services when possible

### Code Quality

- All code must pass type checking
- Security scanning must pass (or risks be documented)
- Dependencies scanned for vulnerabilities

## 📚 Advanced Features

### Manual LLM Testing

You can trigger LLM tests manually in GitHub Actions:

1. Go to **Actions** tab
2. Select **LLM Integration Tests** workflow  
3. Click **Run workflow** button
4. Choose branch and run

### Performance Tracking

The workflow automatically tracks test performance:
- Test execution time trends
- Coverage percentage changes
- Code quality metrics over time

### Fork Contributing

For contributors from forks:
- Pre-commit hooks work locally ✅
- Most CI tests run ✅
- LLM tests skipped (no access to secrets) ✅
- Can still contribute bug fixes and features ✅

## 🆘 Getting Help

### Troubleshooting Checklist

1. **Local pre-commit issues?**
   ```bash
   pre-commit run --all-files  # Debug locally
   pre-commit uninstall       # Reset if needed
   pre-commit install         # Reinstall
   ```

2. **CI tests failing?**
   - Check Actions tab for detailed logs
   - Compare with your local test results
   - Verify Python version matches

3. **Can't merge PR?**
   - Check all status checks are green
   - Ensure branch protection requirements met
   - Verify code review requests resolved

### Support Channels

- **GitHub Issues**: Report CI/CD problems
- **Team Chat**: Quick help with workflow questions  
- **Documentation**: Check `DEVOPS_SETUP.md` for technical details

---

**Remember**: This workflow ensures high code quality while maintaining developer productivity. The automated checks prevent most issues before they reach production, while pre-commit hooks provide immediate feedback during development.