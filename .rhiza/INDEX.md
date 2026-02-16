# Rhiza Index

Quick reference to all utilities, makefiles, and resources in the `.rhiza/` directory.

## 📁 Directory Structure

```
.rhiza/
├── rhiza.mk              # Core makefile logic (153 lines)
├── .rhiza-version        # Current Rhiza version
├── .cfg.toml             # Configuration file
├── .env                  # Environment variables
├── template-bundles.yml  # Template bundle definitions
├── make.d/               # Makefile extensions (auto-loaded)
├── requirements/         # Python dependencies
├── scripts/              # Shell scripts and utilities
├── templates/            # Project templates
├── tests/                # Test suite
├── docs/                 # Internal documentation
└── assets/               # Static assets
```

## 🔧 Makefiles (`.rhiza/make.d/`)

| File | Size | Purpose | Section |
|------|------|---------|---------|
| `agentic.mk` | 3.1K | AI agent integrations (copilot, claude) | Agentic Workflows |
| `book.mk` | 4.7K | Documentation book generation | Book |
| `bootstrap.mk` | 4.3K | Installation and environment setup | Bootstrap |
| `custom-env.mk` | 290B | Example environment customizations | - |
| `custom-task.mk` | 423B | Example custom tasks | Custom Tasks |
| `docker.mk` | 1.1K | Docker build and run targets | Docker |
| `docs.mk` | 3.9K | Documentation generation (pdoc) | Documentation |
| `github.mk` | 6.0K | GitHub CLI integrations | GitHub Helpers |
| `lfs.mk` | 3.0K | Git LFS management | Git LFS |
| `marimo.mk` | 2.9K | Marimo notebook support | Marimo Notebooks |
| `presentation.mk` | 3.3K | Presentation building (Marp) | Presentation |
| `quality.mk` | 860B | Code quality and formatting | Quality and Formatting |
| `releasing.mk` | 2.0K | Release and versioning | Releasing and Versioning |
| `test.mk` | 5.1K | Testing infrastructure | Development and Testing |

**Total**: 14 makefiles, ~41KB

## 📦 Requirements (`.rhiza/requirements/`)

| File | Purpose |
|------|---------|
| `docs.txt` | Documentation generation dependencies (pdoc) |
| `marimo.txt` | Marimo notebook dependencies |
| `tests.txt` | Testing dependencies (pytest, coverage) |
| `tools.txt` | Development tools (pre-commit, python-dotenv) |

See [requirements/README.md](requirements/README.md) for details.

## 🧪 Test Suite (`.rhiza/tests/`)

| Directory | Purpose |
|-----------|---------|
| `api/` | Makefile target validation (dry-run tests) |
| `deps/` | Dependency health checks |
| `integration/` | End-to-end workflow tests |
| `structure/` | Static project structure assertions |
| `sync/` | Template sync and content validation |
| `utils/` | Test infrastructure utilities |

**Total**: 23 Python test files

See [tests/README.md](tests/README.md) for details.

## 📚 Documentation (`.rhiza/docs/`)

| File | Purpose |
|------|---------|
| `ASSETS.md` | Asset management documentation |
| `CONFIG.md` | Configuration file documentation |
| `LFS.md` | Git LFS setup and usage |
| `PRIVATE_PACKAGES.md` | Private package authentication |
| `RELEASING.md` | Release process documentation |
| `TOKEN_SETUP.md` | GitHub token setup |
| `WORKFLOWS.md` | GitHub Actions workflows |

## 🎨 Assets (`.rhiza/assets/`)

- `rhiza-logo.svg` - Rhiza logo graphic

## 📋 Templates (`.rhiza/templates/`)

- `minibook/` - Minimal documentation book template

## 🔌 Template Bundles

Defined in `template-bundles.yml`:

| Bundle | Description | Files |
|--------|-------------|-------|
| `core` | Core Rhiza infrastructure | 43 files |
| `github` | GitHub Actions workflows | CI/CD |
| `tests` | Testing infrastructure | pytest, coverage |
| `marimo` | Interactive notebooks | Marimo support |
| `book` | Documentation generation | Book building |
| `docker` | Docker containerization | Dockerfile |
| `lfs` | Git LFS support | Large files |
| `presentation` | Presentation building | reveal.js |
| `gitlab` | GitLab CI/CD | GitLab workflows |
| `devcontainer` | VS Code DevContainer | Dev environment |
| `legal` | Legal documentation | LICENSE, CODE_OF_CONDUCT |

## 🎯 Key Make Targets

### Bootstrap
- `make install` - Install dependencies
- `make install-uv` - Ensure uv/uvx is installed
- `make clean` - Clean artifacts and stale branches

### Development
- `make test` - Run test suite
- `make fmt` - Format code
- `make docs` - Generate documentation

### AI Agents
- `make copilot` - GitHub Copilot interactive prompt
- `make claude` - Claude Code interactive prompt
- `make analyse-repo` - Update REPOSITORY_ANALYSIS.md

### Documentation
- `make book` - Build documentation book
- `make marimo` - Start Marimo server
- `make presentation` - Generate presentation slides

### Docker
- `make docker-build` - Build Docker image
- `make docker-run` - Run container

### GitHub
- `make view-prs` - List open pull requests
- `make view-issues` - List open issues
- `make failed-workflows` - List failing workflows

### Quality
- `make fmt` - Format code with ruff
- `make lint` - Lint code
- `make deptry` - Check dependencies

### Releasing
- `make release` - Create a release
- `make bump` - Bump version

## 🔗 Related Documentation

- [Architecture Diagrams & Naming Conventions](../docs/ARCHITECTURE.md) - Visual architecture overview and detailed naming conventions
- [Makefile Cookbook](make.d/README.md) - Common patterns and recipes
- [Test Suite Guide](tests/README.md) - Testing conventions
- [Customization Guide](../docs/CUSTOMIZATION.md) - How to customize Rhiza
- [Quick Reference](../docs/QUICK_REFERENCE.md) - Common commands
