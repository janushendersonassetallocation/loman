# Rhiza Tools Reference

A comprehensive quick reference for all tools used in Rhiza-based projects.

## Table of Contents

- [Essential Commands](#essential-commands)
- [Make Commands](#make-commands)
- [UV (Package Manager)](#uv-package-manager)
- [Git Workflows](#git-workflows)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Documentation](#documentation)
- [Release Management](#release-management)
- [AI-Powered Workflows](#ai-powered-workflows)
- [Environment Management](#environment-management)
- [Troubleshooting](#troubleshooting)

---

## Essential Commands

These are the commands you'll use most frequently:

| Command | Description |
|---------|-------------|
| `make help` | Show all available make targets |
| `make install` | Install dependencies and set up environment |
| `make test` | Run all tests with coverage |
| `make fmt` | Format and lint code |
| `make tutorial` | Interactive tutorial for new developers |

---

## Make Commands

### Development Workflow

| Command | Description | Usage |
|---------|-------------|-------|
| `make install` | Install dependencies and setup environment | First-time setup or after dependency changes |
| `make clean` | Clean build artifacts and stale branches | When you need a fresh start |
| `make test` | Run all tests with coverage | Before committing changes |
| `make fmt` | Format and lint code with auto-fix | Before committing changes |
| `make all` | Run all CI checks locally | Before pushing to remote |

### Code Quality

| Command | Description |
|---------|-------------|
| `make deptry` | Check for unused/missing dependencies |
| `make pre-commit` | Run all pre-commit hooks |
| `make typecheck` | Run type checking with ty |
| `make security` | Run security scans (pip-audit and bandit) |
| `make docs-coverage` | Check documentation coverage |

### Testing

| Command | Description |
|---------|-------------|
| `make test` | Run all tests with coverage |
| `make benchmark` | Run performance benchmarks |
| `make hypothesis-test` | Run property-based tests |

### Documentation

| Command | Description |
|---------|-------------|
| `make docs` | Generate API documentation with pdoc |
| `make book` | Build companion book |
| `make mkdocs-serve` | Serve MkDocs site with live reload |
| `make mkdocs-build` | Build MkDocs documentation site |

### Template Management

| Command | Description |
|---------|-------------|
| `make sync` | Sync with upstream template |
| `make summarise-sync` | Preview sync changes without applying |
| `make validate` | Validate project structure |
| `make readme` | Update README.md with current help output |

### Release Management

| Command | Description | Options |
|---------|-------------|---------|
| `make publish` | Bump version, tag, and push (all-in-one) | `DRY_RUN=1` for preview |
| `make bump` | Bump version (prompts for level) | `BUMP=major/minor/patch` |
| `make release` | Create and push release tag | `DRY_RUN=1` for preview |
| `make release-status` | Show release workflow status | |

### Docker

| Command | Description |
|---------|-------------|
| `make docker-build` | Build Docker image |
| `make docker-run` | Run Docker container |
| `make docker-clean` | Remove Docker image |

### Notebooks

| Command | Description |
|---------|-------------|
| `make marimo` | Start Marimo notebook server |
| `make marimushka` | Export Marimo notebooks to HTML |
| `make marimo-validate` | Validate all Marimo notebooks |

### Presentations

| Command | Description |
|---------|-------------|
| `make presentation` | Generate slides from PRESENTATION.md |
| `make presentation-pdf` | Generate PDF presentation |
| `make presentation-serve` | Serve presentation interactively |

### GitHub Integration

| Command | Description |
|---------|-------------|
| `make gh-install` | Install GitHub CLI and extensions |
| `make view-prs` | List open pull requests |
| `make view-issues` | List open issues |
| `make failed-workflows` | List recent failing workflow runs |
| `make whoami` | Check GitHub auth status |
| `make workflow-status` | Show release workflow status |
| `make latest-release` | Show latest GitHub release info |

### Git LFS

| Command | Description |
|---------|-------------|
| `make lfs-install` | Install and configure git-lfs |
| `make lfs-pull` | Download all git-lfs files |
| `make lfs-track` | List patterns tracked by git-lfs |
| `make lfs-status` | Show git-lfs file status |

### AI-Powered Workflows

| Command | Description |
|---------|-------------|
| `make copilot` | Open GitHub Copilot CLI |
| `make claude` | Open Claude Code interactive prompt |
| `make analyse-repo` | AI analysis of repository |
| `make summarise-changes` | Summarize changes since last release |

### Meta Commands

| Command | Description |
|---------|-------------|
| `make help` | Display help message |
| `make version-matrix` | Show supported Python versions |
| `make print-VARIABLE` | Print value of any Makefile variable |
| `make print-logo` | Display Rhiza logo |

---

## UV (Package Manager)

UV is a fast, reliable Python package manager. Always use `uv` commands instead of calling `.venv/bin/python` directly.

### Basic Commands

| Command | Description |
|---------|-------------|
| `uv --version` | Check UV version |
| `uv add package` | Add a dependency |
| `uv add --dev package` | Add a dev dependency |
| `uv remove package` | Remove a dependency |
| `uv sync` | Sync dependencies with lock file |
| `uv lock` | Update lock file without installing |
| `uv tree` | Show dependency tree |

### Running Python

| Command | Description |
|---------|-------------|
| `uv run python script.py` | Run Python script |
| `uv run pytest` | Run pytest |
| `uv run python -m module` | Run Python module |
| `uv run python -c "code"` | Run Python code inline |

### Running External Tools

| Command | Description |
|---------|-------------|
| `uvx tool-name` | Run tool without installing |
| `uvx --from package tool` | Run tool from specific package |
| `uvx --with dep1 --with dep2 tool` | Run tool with extra dependencies |

### Environment Management

| Command | Description |
|---------|-------------|
| `uv venv` | Create virtual environment |
| `uv venv --python 3.13` | Create venv with specific Python version |
| `uv python list` | List available Python versions |
| `uv python install 3.13` | Install specific Python version |

### Information Commands

| Command | Description |
|---------|-------------|
| `uv pip list` | List installed packages |
| `uv pip show package` | Show package information |
| `uv pip check` | Verify package compatibility |

---

## Git Workflows

### Basic Operations

| Command | Description |
|---------|-------------|
| `git status` | Show working tree status |
| `git add .` | Stage all changes |
| `git commit -m "msg"` | Commit with message |
| `git push` | Push to remote |
| `git pull` | Pull from remote |

### Branching

| Command | Description |
|---------|-------------|
| `git branch` | List branches |
| `git branch name` | Create branch |
| `git checkout -b name` | Create and switch to branch |
| `git switch name` | Switch to branch |
| `git merge branch` | Merge branch into current |

### Useful Flags

| Command | Description |
|---------|-------------|
| `git --no-pager status` | Show status without pager |
| `git --no-pager diff` | Show diff without pager |
| `git --no-pager log` | Show log without pager |
| `git commit --amend` | Amend last commit |
| `git push --force-with-lease` | Safe force push |

### Inspection

| Command | Description |
|---------|-------------|
| `git log --oneline -10` | Show last 10 commits (one line each) |
| `git log --graph --oneline` | Show commit graph |
| `git show commit-hash` | Show commit details |
| `git diff` | Show unstaged changes |
| `git diff --staged` | Show staged changes |
| `git blame file` | Show who changed each line |

### Cleanup

| Command | Description |
|---------|-------------|
| `git clean -fd` | Remove untracked files |
| `git reset --hard` | Reset to last commit (destructive!) |
| `git checkout -- file` | Discard changes to file |
| `git restore file` | Restore file from index |

---

## Testing

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
uv run pytest tests/path/to/test.py

# Run specific test function
uv run pytest tests/path/to/test.py::test_function_name

# Run tests with verbose output
uv run pytest -v

# Run tests with print statements visible
uv run pytest -v -s

# Run tests matching a pattern
uv run pytest -k "pattern"

# Run only failed tests from last run
uv run pytest --lf

# Run tests in parallel
uv run pytest -n auto

# Stop after first failure
uv run pytest -x

# Show test durations
uv run pytest --durations=10
```

### Coverage

```bash
# Run tests with coverage
make test

# Generate HTML coverage report
uv run pytest --cov --cov-report=html

# Show coverage for specific package
uv run pytest --cov=package_name

# Fail if coverage below threshold
uv run pytest --cov --cov-fail-under=90
```

### Property-Based Testing

```bash
# Run property-based tests
make hypothesis-test

# Run with more examples
uv run pytest --hypothesis-show-statistics
```

---

## Code Quality

### Formatting and Linting

```bash
# Format and lint with auto-fix
make fmt

# Check only (no changes)
uv run ruff check .

# Format only
uv run ruff format .

# Show which files would be formatted
uv run ruff format --check .
```

### Type Checking

```bash
# Run type checking
make typecheck

# Type check specific file
uv run ty check path/to/file.py
```

### Security Scanning

```bash
# Run all security scans
make security

# Run pip-audit only
uv run pip-audit

# Run bandit only
uv run bandit -r src/
```

### Dependency Checking

```bash
# Check for unused/missing dependencies
make deptry

# Check for outdated packages
uv pip list --outdated
```

---

## Documentation

### Generating Docs

```bash
# Generate API documentation
make docs

# Build companion book
make book

# Build MkDocs site
make mkdocs-build

# Serve docs with live reload
make mkdocs-serve
```

### Documentation Coverage

```bash
# Check documentation coverage
make docs-coverage

# Check specific threshold
interrogate src/ --fail-under 80
```

---

## Release Management

### Version Bumping

```bash
# Bump version (interactive prompt)
make bump

# Bump patch version (0.0.X)
make bump BUMP=patch

# Bump minor version (0.X.0)
make bump BUMP=minor

# Bump major version (X.0.0)
make bump BUMP=major

# Preview without making changes
make bump DRY_RUN=1
```

### Creating Releases

```bash
# Full release (bump + tag + push)
make publish

# Just create and push tag
make release

# Preview release
make release DRY_RUN=1

# Check release status
make release-status

# View latest release
make latest-release
```

---

## AI-Powered Workflows

### GitHub Copilot

```bash
# Open Copilot CLI
make copilot

# Or use directly
gh copilot suggest "command description"
gh copilot explain "command to explain"
```

### Claude Code

```bash
# Open Claude interactive prompt
make claude
```

### Repository Analysis

```bash
# Analyze repository structure
make analyse-repo

# Summarize changes since last release
make summarise-changes
```

---

## Environment Management

### Virtual Environment

```bash
# Create virtual environment
uv venv

# Activate environment (if needed manually)
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Deactivate environment
deactivate
```

### Python Version

```bash
# Check current Python version
python --version
uv run python --version

# List available Python versions
uv python list

# Install specific Python version
uv python install 3.13

# Use specific Python version
uv venv --python 3.13
```

### Environment Variables

```bash
# Load from .rhiza/.env (automatically done by Makefile)
# Or load manually:
source .rhiza/.env

# Print environment variable
echo $VARIABLE_NAME

# Set temporarily
VARIABLE=value make target

# Print Makefile variable
make print-VARIABLE_NAME
```

---

## Troubleshooting

### Common Issues

#### "Command not found: uv"

```bash
# Install uv
make install-uv

# Or manually
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### "Python version mismatch"

```bash
# Clean and reinstall
make clean
make install
```

#### "Import errors after adding dependency"

```bash
# Sync dependencies
uv sync

# Or full reinstall
make install
```

#### "Tests failing unexpectedly"

```bash
# Clear pytest cache
rm -rf .pytest_cache/

# Run tests with verbose output
uv run pytest -v -s
```

#### "Pre-commit hooks failing"

```bash
# Install pre-commit
uv run pre-commit install

# Run all hooks manually
make fmt

# Run specific hook
uv run pre-commit run hook-name --all-files
```

#### "Git LFS files not downloading"

```bash
# Install and setup git-lfs
make lfs-install

# Download all files
make lfs-pull
```

### Debug Commands

```bash
# Check UV configuration
uv --version
uv pip list

# Check Python environment
uv run python -c "import sys; print(sys.executable)"
uv run python -c "import sys; print(sys.path)"

# Check make variables
make print-PYTHON_VERSION
make print-UV_BIN
make print-VENV

# Check git status
git status
git --no-pager diff

# Verify installation
make install
make test
```

### Getting Help

```bash
# Show all make targets
make help

# Show uv help
uv --help
uv add --help

# Show git help
git help
git help command

# Show pytest help
uv run pytest --help

# Read documentation
cat docs/QUICK_REFERENCE.md
cat docs/EXTENDING_RHIZA.md
```

---

## Quick Reference Cards

### Daily Development

```bash
# Morning: Update and sync
git pull
make install

# Development cycle
# 1. Write code
# 2. Test changes
make test

# 3. Format code
make fmt

# 4. Commit
git add .
git commit -m "feat: description"

# 5. Push
git push
```

### Before Committing

```bash
make fmt          # Format and lint
make test         # Run tests
make all          # Run all CI checks
```

### Before Releasing

```bash
make test         # Verify tests pass
make all          # Run all CI checks
make publish      # Bump, tag, and push
```

### Adding Dependencies

```bash
# Add runtime dependency
uv add package-name

# Add dev dependency
uv add --dev package-name

# Verify and test
make test
```

### When Things Break

```bash
# Clean everything
make clean

# Fresh install
make install

# Run tests to verify
make test

# If still broken, check:
git status
git --no-pager diff
make print-PYTHON_VERSION
```

---

## Command Equivalents

### Make vs Direct Commands

| Make Command | Equivalent Direct Command |
|--------------|---------------------------|
| `make install` | `uv sync --all-extras` |
| `make test` | `uv run pytest --cov` |
| `make fmt` | `uv run ruff format . && uv run ruff check --fix .` |
| `make docs` | `uv run pdoc --html --output-dir docs/api src/` |

**Always prefer make commands** when available, as they include additional logic and hooks.

---

## Environment Variables

Common environment variables used by Rhiza:

| Variable | Description | Default |
|----------|-------------|---------|
| `PYTHON_VERSION` | Python version to use | From `.python-version` |
| `RHIZA_VERSION` | Rhiza version | From `.rhiza/.rhiza-version` |
| `UV_BIN` | Path to uv binary | Auto-detected or `./bin/uv` |
| `VENV` | Virtual environment path | `.venv` |
| `COVERAGE_FAIL_UNDER` | Minimum coverage threshold | 90 |
| `DRY_RUN` | Preview mode for releases | (unset) |
| `BUMP` | Version bump type | (prompt) |

Set in Makefile or pass to commands:

```bash
# Override in command
make test COVERAGE_FAIL_UNDER=80

# Set in Makefile (before include line)
COVERAGE_FAIL_UNDER = 80

# Set in local.mk
PYTHON_VERSION = 3.12
```

---

## Tips and Best Practices

### General

- **Always use `make` targets when available** (they include hooks and validation)
- **Always use `uv run` for Python commands** (never call `.venv/bin/python` directly)
- **Use `DRY_RUN=1` for safe previews** of releases and bumps
- **Run `make fmt` before every commit** to maintain code quality
- **Run `make test` frequently** during development
- **Check `make help`** when you forget a command

### Development

- Use `git --no-pager` commands in scripts to avoid interactive pagers
- Create `local.mk` for personal shortcuts (it's gitignored)
- Use hooks (`post-install::`, etc.) for custom setup steps
- Keep the root `Makefile` small and focused

### Dependencies

- Let uv handle Python version management (don't use pyenv/asdf)
- Use `uv add` instead of manually editing `pyproject.toml`
- Commit both `pyproject.toml` and `uv.lock` to git
- Run `make deptry` regularly to check for unused dependencies

### Testing

- Write tests alongside code changes
- Use `pytest -k pattern` to run specific tests during development
- Check coverage with `make test`
- Use `--lf` flag to re-run only failed tests

### Documentation

- Document functions with docstrings
- Keep README.md updated with `make readme`
- Generate API docs with `make docs`
- Check documentation coverage with `make docs-coverage`

---

## See Also

- [Quick Reference](../guides/QUICK_REFERENCE.md) - Condensed command reference
- [Extending Rhiza](../guides/EXTENDING_RHIZA.md) - Extending and customizing Rhiza
- [Contributing Guide](../CONTRIBUTING.md) - How to contribute
- [README](../README.md) - Project overview

---

*Last updated: 2026-02-15*
