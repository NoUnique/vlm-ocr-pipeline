# CI/CD Pipeline

This document describes the Continuous Integration and Continuous Deployment (CI/CD) workflows for the VLM OCR Pipeline project.

## Overview

The project uses GitHub Actions for automated testing, linting, type checking, and documentation deployment. All workflows are defined in `.github/workflows/`.

## Workflows

### 1. Main CI Workflow (`.github/workflows/ci.yml`)

Runs on every push to `main`/`develop` branches and on pull requests.

#### Jobs

**Lint and Format Check**
- Checks code formatting with `ruff format --check`
- Runs linting with `ruff check`
- Ensures code style consistency

**Type Check**
- Runs `pyright` for static type checking
- Validates type annotations across the codebase
- Uses Node.js 20 for `npx pyright`

**Test**
- Runs pytest test suite
- Tests on Python 3.11 and 3.12
- Uploads coverage reports to Codecov (optional)
- Uses matrix strategy for multi-version testing

**Build Documentation**
- Builds MkDocs documentation with `--strict` flag
- Catches documentation errors before deployment
- Validates all documentation links and references

**All Checks Passed**
- Final gate that requires all previous jobs to succeed
- Provides clear status for PR mergeability

#### Trigger Events
```yaml
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]
  workflow_dispatch:  # Manual trigger
```

---

### 2. PR Checks Workflow (`.github/workflows/pr-checks.yml`)

Provides automated analysis and labeling for pull requests.

#### Jobs

**PR Information**
- Analyzes changed files (Python, tests, docs, workflows)
- Counts lines added/deleted
- Posts summary comment on PR
- Updates existing comment on new commits

**Size Label**
- Automatically labels PRs by size:
  - `size/XS`: < 50 lines
  - `size/S`: 50-200 lines
  - `size/M`: 200-500 lines
  - `size/L`: 500-1000 lines
  - `size/XL`: > 1000 lines

#### Permissions
```yaml
permissions:
  pull-requests: write
  contents: read
  checks: write
```

---

### 3. Documentation Deployment (`.github/workflows/docs.yml`)

Automatically deploys documentation to GitHub Pages.

#### Trigger Conditions
- Changes to `docs/**`
- Changes to `mkdocs.yml`
- Changes to workflow file itself
- Manual dispatch

#### Deployment
- Builds documentation with MkDocs
- Deploys to `gh-pages` branch
- Serves at: `https://<username>.github.io/<repo>/`

---

## Setup Instructions

### 1. Enable GitHub Actions

GitHub Actions should be enabled by default. Verify in repository Settings → Actions.

### 2. Configure GitHub Pages

1. Go to repository Settings → Pages
2. Source: Deploy from a branch
3. Branch: `gh-pages` / `root`
4. Save

### 3. Add Secrets (Optional)

For coverage reports:
```
Settings → Secrets → Actions → New repository secret
Name: CODECOV_TOKEN
Value: <your-codecov-token>
```

### 4. Branch Protection (Recommended)

Protect `main` branch with required status checks:

1. Settings → Branches → Add rule
2. Branch name pattern: `main`
3. Require status checks before merging:
   - ✅ Lint and Format Check
   - ✅ Type Check
   - ✅ Test (3.11)
   - ✅ Test (3.12)
   - ✅ Build Documentation
   - ✅ All Checks Passed
4. Require pull request reviews
5. Save changes

---

## Local Development Workflow

### Pre-commit Checks

Before committing, run the pre-commit script:

```bash
./scripts/pre-commit-check.sh
```

This runs the same checks as CI:
1. Code formatting check
2. Linting
3. Type checking
4. Documentation build

### Manual CI Checks

Run individual checks locally:

```bash
# Format check
uv run ruff format --check .

# Lint
uv run ruff check .

# Type check
npx pyright

# Test
uv run pytest tests/ -v

# Build docs
uv run mkdocs build --strict
```

### Fix Issues

```bash
# Auto-fix formatting
uv run ruff format .

# Auto-fix linting issues
uv run ruff check . --fix

# View type errors (no auto-fix)
npx pyright
```

---

## CI/CD Best Practices

### Commit Messages

Use conventional commits format:
```
feat: add new feature
fix: resolve bug
docs: update documentation
test: add test coverage
refactor: improve code structure
perf: performance improvement
ci: update CI/CD configuration
```

### Pull Requests

1. Keep PRs focused and small (prefer size/S or size/M)
2. Ensure all CI checks pass before requesting review
3. Add tests for new features
4. Update documentation for user-facing changes
5. Respond to automated PR comments

### Testing

- Write tests for all new features
- Maintain >80% test coverage
- Include both unit and integration tests
- Use fixtures for test data

### Type Safety

- Add type annotations to all functions
- Use `from __future__ import annotations` for forward references
- Run `npx pyright` before committing
- Fix type errors, don't suppress them unnecessarily

---

## Troubleshooting

### CI Fails but Local Checks Pass

1. Ensure you're using the same Python version (3.11)
2. Check that all dependencies are in `requirements.txt`
3. Clear local cache: `rm -rf .venv .ruff_cache .pytest_cache`
4. Reinstall: `uv venv --python 3.11 && uv pip install -r requirements.txt`

### Type Check Fails on CI

- CI uses `npx pyright` (not global install)
- Ensure Node.js types are consistent
- Check for platform-specific type issues

### Tests Timeout

- CI has 6-hour timeout per workflow
- Individual jobs have 360-minute timeout
- Long-running tests should use appropriate fixtures
- Consider mocking external API calls

### Documentation Build Fails

- Run `uv run mkdocs build --strict` locally
- Check for broken links in markdown files
- Verify all referenced files exist
- Ensure proper YAML frontmatter in docs

---

## Monitoring

### View Workflow Status

- Repository → Actions tab
- Click on workflow run to see details
- Each job shows detailed logs

### Status Badges

Add to README.md:

```markdown
[![CI](https://github.com/username/repo/actions/workflows/ci.yml/badge.svg)](https://github.com/username/repo/actions/workflows/ci.yml)
```

### Notifications

Configure in GitHub Settings → Notifications:
- Email notifications for failed workflows
- Slack/Discord integration (via webhooks)

---

## Future Improvements

### Planned Enhancements

1. **Code Coverage Enforcement**
   - Fail CI if coverage drops below threshold
   - Per-file coverage reports

2. **Security Scanning**
   - Dependency vulnerability scanning (Dependabot)
   - Secret scanning
   - SAST (Static Application Security Testing)

3. **Performance Regression Testing**
   - Benchmark tests in CI
   - Compare performance against main branch

4. **Automatic Dependency Updates**
   - Renovate or Dependabot
   - Auto-merge minor updates if tests pass

5. **Release Automation**
   - Automatic changelog generation
   - Semantic versioning
   - PyPI package publishing

---

## See Also

- [Pre-commit Script](../scripts/pre-commit-check.sh)
- [CLAUDE.md](../CLAUDE.md) - Development guidelines
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
