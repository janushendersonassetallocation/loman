## Makefile (repo-owned)
# Keep this file small. It can be edited without breaking template sync.

DOCFORMAT=google
DEFAULT_AI_MODEL=claude-sonnet-4.5
LOGO_FILE=.rhiza/assets/rhiza-logo.svg
GH_AW_ENGINE ?= copilot  # Default AI engine for gh-aw workflows (copilot, claude, or codex)

# Always include the Rhiza API (template-managed)
include .rhiza/rhiza.mk

# Optional: developer-local extensions (not committed)
-include local.mk

# Wire typecheck into make validate
post-validate::
	@$(MAKE) typecheck

## Custom targets

##@ Security

.PHONY: security
security: install ## run security scans (pip-audit and bandit)
	@printf "${BLUE}[INFO] Running pip-audit for dependency vulnerabilities...${RESET}\n"
	@${UVX_BIN} pip-audit
	@printf "${BLUE}[INFO] Running bandit security scan...${RESET}\n"
	@${UVX_BIN} bandit -r ${SOURCE_FOLDER} -ll -q -c pyproject.toml

##@ Quality

.PHONY: semgrep
semgrep: install ## run Semgrep static analysis (numpy rules)
	@printf "${BLUE}[INFO] Running Semgrep (numpy rules)...${RESET}\n"
	@if [ -d ${SOURCE_FOLDER} ]; then \
		${UVX_BIN} semgrep --config .rhiza/semgrep.yml ${SOURCE_FOLDER}; \
	else \
		printf "${YELLOW}[WARN] SOURCE_FOLDER '${SOURCE_FOLDER}' not found, skipping semgrep.${RESET}\n"; \
	fi

.PHONY: adr
adr: install-gh-aw ## Create a new Architecture Decision Record (ADR) using AI assistance
	@echo "Creating a new ADR..."
	@echo "This will trigger the adr-create workflow."
	@echo ""
	@read -p "Enter ADR title (e.g., 'Use PostgreSQL for data storage'): " title; \
	echo ""; \
	read -p "Enter brief context (optional, press Enter to skip): " context; \
	echo ""; \
	if [ -z "$$title" ]; then \
		echo "Error: Title is required"; \
		exit 1; \
	fi; \
	if [ -z "$$context" ]; then \
		gh workflow run adr-create.md -f title="$$title"; \
	else \
		gh workflow run adr-create.md -f title="$$title" -f context="$$context"; \
	fi; \
	echo ""; \
	echo "✅ ADR creation workflow triggered!"; \
	echo ""; \
	echo "The workflow will:"; \
	echo "  1. Generate the next ADR number"; \
	echo "  2. Create a comprehensive ADR document"; \
	echo "  3. Update the ADR index"; \
	echo "  4. Open a pull request for review"; \
	echo ""; \
	echo "Check workflow status: gh run list --workflow=adr-create.md"; \
	echo "View latest run: gh run view"

