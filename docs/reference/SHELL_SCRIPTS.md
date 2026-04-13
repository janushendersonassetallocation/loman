# Shell Scripts Documentation

This document describes the shell scripts used in the Rhiza project, their purpose, error handling, and troubleshooting guides.

## Overview

Rhiza uses minimal, focused shell scripts (3 scripts, ~150 total lines) for critical automation tasks. All scripts follow strict bash practices:

- ✅ **Strict error handling**: `set -euo pipefail` (fail on error, undefined variables, pipe failures)
- ✅ **Descriptive error messages**: Clear error descriptions with remediation steps
- ✅ **Recovery options**: Fallback strategies for non-critical failures
- ✅ **Comprehensive testing**: Automated test suite validates functionality

## Shell Scripts

### 1. `.devcontainer/bootstrap.sh`

**Purpose**: Initializes the development environment in DevContainers and GitHub Codespaces.

**Location**: `.devcontainer/bootstrap.sh` (92 lines)

**What it does**:
1. Reads Python version from `.python-version`
2. Configures UV package manager environment variables
3. Installs project dependencies via `make install`
4. Installs Marimo notebook tool (with fallback)
5. Sets up pre-commit hooks (with fallback)

**Error Handling**:

The script includes an `error_with_recovery()` function that provides:
- Clear error descriptions
- Suggested remediation steps
- Contextual help for common issues

**Recovery Options**:

| Component | Behavior on Failure | Recovery |
|-----------|---------------------|----------|
| `.python-version` missing | ❌ Fatal error | Shows remediation: ensure file exists |
| `make install` fails | ❌ Fatal error | Shows steps: check connectivity, disk space |
| Marimo installation fails | ⚠️ Warning, continues | User can install manually later |
| Pre-commit setup fails | ⚠️ Warning, continues | User can install manually later |

**Environment Variables**:

```bash
INSTALL_DIR="${INSTALL_DIR:-./bin}"  # UV installation directory
UV_BIN="${INSTALL_DIR}/uv"           # Path to uv binary
UVX_BIN="${INSTALL_DIR}/uvx"         # Path to uvx binary
UV_VENV_CLEAR=1                      # Clear venv on install
UV_LINK_MODE=copy                    # Use copy mode for cross-filesystem support
PYTHON_VERSION                       # Read from .python-version file
```

**Example Output**:

```bash
✓ Using Python version from .python-version: 3.13
📦 Installing project dependencies...
✓ Dependencies installed successfully
📓 Installing Marimo notebook tool...
✓ Marimo installed successfully
🔧 Setting up pre-commit hooks...
✓ Pre-commit hooks configured successfully
✅ Bootstrap completed successfully!
```

**Common Issues**:

| Issue | Symptom | Solution |
|-------|---------|----------|
| Internet connectivity | `make install` fails downloading packages | Check network, retry |
| Insufficient disk space | Installation fails with write errors | Run `df -h`, free up space |
| `.python-version` missing | Error on startup | Ensure file exists in repo root |
| UV installation fails | Commands not found | Check UV installer, verify PATH |

### 2. `.github/hooks/session-start.sh`

**Purpose**: Validates the development environment before GitHub Copilot agent sessions begin.

**Location**: `.github/hooks/session-start.sh` (37 lines)

**What it does**:
1. Verifies UV package manager is available
2. Confirms virtual environment (`.venv`) exists
3. Checks that virtual environment is activated

**Error Handling**:

Each validation includes:
- `[ERROR]` Clear error message
- `[INFO]` Specific remediation steps
- `[INFO]` Alternative solutions

**Example Errors**:

```bash
# UV not found
[copilot-hook] [ERROR] uv not found
[copilot-hook] [INFO] Remediation: Run 'make install' to set up the environment
[copilot-hook] [INFO] Alternative: Ensure uv is in PATH or ./bin/uv exists

# Virtual environment missing
[copilot-hook] [ERROR] .venv not found
[copilot-hook] [INFO] Remediation: Run 'make install' to create the virtual environment
[copilot-hook] [INFO] Details: The .venv directory should contain Python dependencies

# Virtual environment not activated (warning only)
[copilot-hook] [WARN] .venv/bin is not on PATH
[copilot-hook] [INFO] Note: The agent may not use the correct Python version
[copilot-hook] [INFO] Remediation: Ensure .venv/bin is added to PATH before running the agent
```

**Usage**:

Called automatically by `.github/hooks/hooks.json` at session start:

```json
{
  "sessionStart": ".github/hooks/session-start.sh"
}
```

### 3. `.github/hooks/session-end.sh`

**Purpose**: Runs quality gates after GitHub Copilot agent sessions complete.

**Location**: `.github/hooks/session-end.sh` (35 lines)

**What it does**:
1. Formats code using `make fmt`
2. Runs tests using `make test`
3. Reports success or failure with detailed guidance

**Error Handling**:

Each quality gate provides:
- `[ERROR]` Clear failure message
- `[INFO]` Detailed remediation steps
- `[INFO]` Common solutions

**Example Errors**:

```bash
# Formatting failure
[copilot-hook] [ERROR] Formatting check failed
[copilot-hook] [INFO] Remediation: Review the formatting errors above
[copilot-hook] [INFO] Common fixes:
[copilot-hook]    - Run 'make fmt' locally to see detailed errors
[copilot-hook]    - Check for syntax errors in modified files
[copilot-hook]    - Ensure all files follow project style guidelines

# Test failure
[copilot-hook] [ERROR] Tests failed
[copilot-hook] [INFO] Remediation: Review the test failures above
[copilot-hook] [INFO] Common fixes:
[copilot-hook]    - Run 'make test' locally to see detailed output
[copilot-hook]    - Check if new code broke existing functionality
[copilot-hook]    - Verify test assertions match expected behavior
[copilot-hook]    - Review test logs in _tests/ directory
```

**Usage**:

Called automatically by `.github/hooks/hooks.json` at session end:

```json
{
  "sessionEnd": ".github/hooks/session-end.sh"
}
```

## Testing

### Test Suite: `.rhiza/tests/shell/test_scripts.sh`

**Purpose**: Validates shell script correctness and error handling.

**Location**: `.rhiza/tests/shell/test_scripts.sh` (265 lines)

**What it tests**:
- ✅ Proper shebang (`#!/bin/bash`)
- ✅ Strict error handling (`set -euo pipefail`)
- ✅ Error recovery functions
- ✅ Remediation messages
- ✅ Bash syntax validation
- ✅ Script behavior in valid environments

**Running Tests**:

```bash
# Normal mode (summary only)
bash .rhiza/tests/shell/test_scripts.sh

# Verbose mode (show each test)
bash .rhiza/tests/shell/test_scripts.sh --verbose
```

**Example Output**:

```bash
=== Shell Script Test Suite ===
Repository: /home/runner/work/rhiza/rhiza

Testing: session-start.sh
```bash
=== Shell Script Test Suite ===
Repository: /home/runner/work/rhiza/rhiza

Testing: session-start.sh
✓ PASS: session-start.sh has bash shebang
✓ PASS: session-start.sh uses strict error handling
✓ PASS: session-start.sh normal mode runs validation

Testing: session-end.sh
✓ PASS: session-end.sh has bash shebang
✓ PASS: session-end.sh uses strict error handling

Testing: bootstrap.sh
[... more tests ...]

=== Test Summary ===
Tests run: 13
Tests passed: 13
All tests passed!
```

**Test Framework**:

The test suite uses a custom bash-based framework with:
- `assert_equal()` - Compare expected vs actual values
- `assert_contains()` - Check if output contains text
- `assert_exit_code()` - Verify exit codes

No external dependencies required (no bats-core needed).

## Best Practices

### For Script Authors

1. **Always use strict mode**:
   ```bash
   set -euo pipefail
   IFS=$'\n\t'
   ```

2. **Provide descriptive errors with remediation**:
   ```bash
   echo "[script] [ERROR] Something failed"
   echo "[script] [INFO] Remediation: Try this solution"
   echo "[script] [INFO] Alternative: Or try this"
   ```

3. **Use clear exit codes**:
   - `0` - Success
   - `1` - Fatal error
   - For warnings, print message but don't exit

4. **Validate syntax**:
   ```bash
   bash -n script.sh  # Syntax check
   shellcheck script.sh  # Linting (if available)
   ```

### For Script Users

1. **Read error messages carefully**:
   - Look for `[ERROR]` lines
   - Follow `[INFO]` Remediation steps in order

2. **Check environment variables**:
   ```bash
   echo $INSTALL_DIR
   echo $UV_BIN
   ```

3. **Run tests after modifications**:
   ```bash
   bash .rhiza/tests/shell/test_scripts.sh --verbose
   ```

## Troubleshooting

### Bootstrap Script Issues

#### Issue: "make install" fails

**Symptoms**:
```
[ERROR] Dependency installation failed
```

**Solutions**:
1. Check internet connectivity: `ping github.com`
2. Check disk space: `df -h`
3. Manually run `make install` to see detailed errors
4. Check UV installation: `command -v uv` or `ls -la ./bin/uv`
5. Check `.python-version` file exists

#### Issue: Marimo installation warning

**Symptoms**:
```
[WARN] Marimo installation failed (non-critical)
```

**Solutions**:
1. This is non-critical; bootstrap continues
2. Install manually later: `uv tool install marimo`
3. Verify UV is working: `uv --version`

#### Issue: Pre-commit setup warning

**Symptoms**:
```
[WARN] Pre-commit hook installation failed (non-critical)
```

**Solutions**:
1. This is non-critical; bootstrap continues
2. Install manually later: `uvx pre-commit install`
3. Ensure `.pre-commit-config.yaml` exists

### Session Hook Issues

#### Issue: "uv not found"

**Symptoms**:
```
[copilot-hook] [ERROR] uv not found
```

**Solutions**:
1. Run `make install` to set up environment
2. Check if `./bin/uv` exists: `ls -la ./bin/uv`
3. Check if `uv` is in PATH: `command -v uv`
4. Try manual UV installation: `curl -LsSf https://astral.sh/uv/install.sh | sh`

#### Issue: ".venv not found"

**Symptoms**:
```
[copilot-hook] [ERROR] .venv not found
```

**Solutions**:
1. Run `make install` to create virtual environment
2. Check if `.venv` directory was deleted: `ls -la .venv`
3. Verify `pyproject.toml` exists

#### Issue: Virtual environment not activated

**Symptoms**:
```
[copilot-hook] [WARN] .venv/bin is not on PATH
```

**Solutions**:
1. This is a warning, not an error (session continues)
2. Manually activate: `source .venv/bin/activate`
3. Ensure copilot-setup-steps.yml activates venv
4. Check PATH: `echo $PATH | grep .venv`

#### Issue: Formatting fails

**Symptoms**:
```
[copilot-hook] [ERROR] Formatting check failed
```

**Solutions**:
1. Run `make fmt` locally to see detailed errors
2. Check for syntax errors in modified files
3. Review ruff configuration in `ruff.toml`
4. Manually format: `uv run ruff format .`

#### Issue: Tests fail

**Symptoms**:
```
[copilot-hook] [ERROR] Tests failed
```

**Solutions**:
1. Run `make test` locally for detailed output
2. Check test logs in `_tests/html-report/`
3. Verify new code doesn't break existing tests
4. Run specific test: `uv run pytest tests/path/to/test.py`

## Advanced Usage

### Custom Error Handlers

When creating new scripts, use the `error_with_recovery()` pattern:

```bash
error_with_recovery() {
    local step="$1"
    local error_msg="$2"
    local remediation="$3"
    
    echo "[ERROR] $step failed"
    echo "   Details: $error_msg"
    echo "   [INFO] Suggested fix: $remediation"
    return 1
}

# Usage
if ! some_command; then
    error_with_recovery \
        "Step name" \
        "What went wrong" \
        "How to fix it"
fi
```

### Extending Test Suite

To add new tests to `.rhiza/tests/shell/test_scripts.sh`:

```bash
# Test N: Description
output=$(bash "$REPO_ROOT/path/to/script.sh" 2>&1) || true
assert_contains "$output" "expected text" "test description"
```

### Integration with CI/CD

Shell scripts integrate with GitHub Actions:

- `.github/workflows/copilot-setup-steps.yml` - Runs `make install`
- `.github/hooks/hooks.json` - Calls session hooks
- `.github/workflows/*.yml` - Can use scripts in CI

## References

- [Bash Strict Mode](http://redsymbol.net/articles/unofficial-bash-strict-mode/)
- [Shell Style Guide](https://google.github.io/styleguide/shellguide.html)
- [UV Documentation](https://docs.astral.sh/uv/)
- [GitHub Codespaces](https://docs.github.com/en/codespaces)
- [VS Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)

## Version History

- **v0.7.5** - Initial shell script documentation
  - Added recovery options to bootstrap.sh
  - Improved error messaging with remediation steps
  - Created comprehensive test suite
  - Added this documentation
- **v0.8.0** - Simplified shell scripts
  - Removed dry-run mode functionality
  - Reduced test count from 19 to 13 tests
  - Updated documentation to reflect changes
