# Extending Rhiza

This guide provides comprehensive examples and best practices for extending and customizing Rhiza-based projects.

## Table of Contents

- [Overview](#overview)
- [Extension Points](#extension-points)
- [Common Patterns](#common-patterns)
- [Advanced Techniques](#advanced-techniques)
- [Real-World Examples](#real-world-examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [CodeQL Configuration](#codeql-configuration)
- [Documentation Customization](#documentation-customization)

---

## Overview

Rhiza's modular design allows you to extend functionality without modifying template-managed files. This ensures:

- ✅ **Template sync compatibility** - Your customizations survive template updates
- ✅ **Clean separation** - Project-specific code stays separate from framework code  
- ✅ **Easy maintenance** - Changes are localized and well-organized
- ✅ **Team flexibility** - Developers can have personal local overrides

### Where to Add Customizations

| Location | Purpose | Committed to Git? |
|----------|---------|-------------------|
| Root `Makefile` | Project-wide customizations | ✅ Yes |
| `local.mk` | Developer-local overrides | ❌ No (gitignored) |
| `.rhiza/.env` | Environment variables | ✅ Yes (optional) |

**Important**: Never modify files in `.rhiza/` directly—they are template-managed and will be overwritten during sync operations.

---

## Extension Points

### 1. Makefile Hooks

Hooks allow you to inject custom logic into standard workflows using double-colon syntax (`::`).

#### Available Hooks

| Hook | When It Runs | Common Use Cases |
|------|--------------|------------------|
| `pre-install` | Before `make install` | Install system dependencies, validate environment |
| `post-install` | After `make install` | Additional setup, generate files, configure services |
| `pre-sync` | Before template sync | Backup files, validate state |
| `post-sync` | After template sync | Regenerate files, apply local patches |
| `pre-validate` | Before project validation | Custom checks, pre-validation setup |
| `post-validate` | After project validation | Additional validation steps |
| `pre-bump` | Before version bump | Update version in additional files |
| `post-bump` | After version bump | Generate changelogs, update documentation |
| `pre-release` | Before creating release | Final checks, build artifacts |
| `post-release` | After creating release | Deploy, notify team, update external systems |

#### Hook Syntax

```makefile
# In root Makefile (before include line)

# Single-line hook
post-install::
	@echo "Running custom setup..."

# Multi-line hook
post-install::
	@echo "Installing additional tools..."
	@uv run pip install my-private-package
	@echo "Setup complete!"

# Multiple hooks (they accumulate)
post-install::
	@./scripts/setup-database.sh

post-install::
	@./scripts/configure-services.sh
```

### 2. Custom Make Targets

Add new make targets in your root `Makefile` to create project-specific commands.

#### Basic Target

```makefile
# In root Makefile (before include line)

##@ Custom Tasks
my-task: ## Description shown in make help
	@echo "Running my custom task..."
	@uv run python scripts/my_script.py
```

#### Target with Dependencies

```makefile
##@ Custom Tasks
deploy: test docs ## Deploy application (runs tests and docs first)
	@echo "Deploying application..."
	@./scripts/deploy.sh
```

#### Target with Variables

```makefile
##@ Custom Tasks
train: ## Train ML model (use ENV=prod for production)
	@echo "Training model in $(ENV) environment..."
	@uv run python scripts/train.py --env=$(ENV)
```

### 3. Environment Variables

Override default variables or add new ones.

#### In Root Makefile

```makefile
# In root Makefile (before include line)

# Override default Python version
PYTHON_VERSION = 3.12

# Override coverage threshold
COVERAGE_FAIL_UNDER = 80

# Add custom variable
MY_API_URL = https://api.example.com

# Export for use in recipes
export MY_API_URL

# Include Rhiza
include .rhiza/rhiza.mk
```

#### In .rhiza/.env

```bash
# .rhiza/.env
DATABASE_URL=postgresql://localhost/mydb
API_KEY=your-api-key
LOG_LEVEL=DEBUG
```

These are automatically loaded by the Makefile.

---

## Common Patterns

### Pattern 1: Installing System Dependencies

Ensure system packages are available before running your application.

```makefile
# Root Makefile

pre-install::
	@echo "Checking system dependencies..."
	@if ! command -v graphviz >/dev/null 2>&1; then \
		echo "Installing graphviz..."; \
		if command -v brew >/dev/null 2>&1; then \
			brew install graphviz; \
		elif command -v apt-get >/dev/null 2>&1; then \
			sudo apt-get update && sudo apt-get install -y graphviz; \
		else \
			echo "Please install graphviz manually."; \
			exit 1; \
		fi \
	fi
```

### Pattern 2: Database Setup

Set up a database during installation.

```makefile
# Root Makefile

post-install::
	@echo "Setting up database..."
	@uv run python scripts/init_db.py
	@echo "Running migrations..."
	@uv run alembic upgrade head

##@ Database
db-migrate: ## Create a new database migration
	@uv run alembic revision --autogenerate -m "$(MSG)"

db-upgrade: ## Apply database migrations
	@uv run alembic upgrade head

db-downgrade: ## Rollback last migration
	@uv run alembic downgrade -1

db-reset: ## Reset database (WARNING: destructive!)
	@echo "Resetting database..."
	@uv run python scripts/reset_db.py
	@$(MAKE) db-upgrade
```

### Pattern 3: Configuration Files

Generate configuration files from templates.

```makefile
# Root Makefile

post-install::
	@echo "Generating configuration files..."
	@if [ ! -f config/local.yaml ]; then \
		cp config/local.yaml.template config/local.yaml; \
		echo "Created config/local.yaml - please customize it"; \
	fi

##@ Configuration
config-validate: ## Validate configuration files
	@uv run python scripts/validate_config.py

config-show: ## Show current configuration
	@uv run python -c "from myapp.config import settings; print(settings.model_dump_json(indent=2))"
```

### Pattern 4: Building Assets

Compile frontend assets or other build artifacts.

```makefile
# Root Makefile

post-install::
	@echo "Building frontend assets..."
	@npm install
	@npm run build

##@ Build
build-assets: ## Build frontend assets
	@echo "Building assets..."
	@npm run build

watch-assets: ## Watch and rebuild assets on change
	@npm run watch

clean-assets: ## Clean built assets
	@rm -rf static/dist/
```

### Pattern 5: Multi-Environment Support

Support different environments (dev, staging, production).

```makefile
# Root Makefile

# Default environment
ENV ?= dev

##@ Deployment
deploy-dev: ## Deploy to development
	@$(MAKE) deploy ENV=dev

deploy-staging: ## Deploy to staging
	@$(MAKE) deploy ENV=staging

deploy-prod: ## Deploy to production
	@$(MAKE) deploy ENV=prod

deploy: test ## Deploy to $(ENV) environment
	@echo "Deploying to $(ENV) environment..."
	@uv run python scripts/deploy.py --env=$(ENV)
```

### Pattern 6: Development Servers

Run development servers with auto-reload.

```makefile
# Root Makefile

##@ Development
dev: ## Start development server with auto-reload
	@echo "Starting development server..."
	@uv run uvicorn myapp.main:app --reload --host 0.0.0.0 --port 8000

dev-worker: ## Start background worker
	@echo "Starting background worker..."
	@uv run celery -A myapp.worker worker --loglevel=info

dev-all: ## Start all development services
	@echo "Starting all services..."
	@$(MAKE) -j dev dev-worker
```

### Pattern 7: Data Management

Tasks for managing data, seeds, fixtures.

```makefile
# Root Makefile

##@ Data Management
seed-db: ## Seed database with sample data
	@echo "Seeding database..."
	@uv run python scripts/seed_data.py

import-data: ## Import data from file (use FILE=path/to/file)
	@echo "Importing data from $(FILE)..."
	@uv run python scripts/import_data.py $(FILE)

export-data: ## Export data to file (use FILE=path/to/file)
	@echo "Exporting data to $(FILE)..."
	@uv run python scripts/export_data.py $(FILE)

backup-db: ## Backup database
	@echo "Creating backup..."
	@mkdir -p backups
	@uv run python scripts/backup_db.py backups/backup-$$(date +%Y%m%d-%H%M%S).sql
```

### Pattern 8: Code Generation

Generate boilerplate code from templates.

```makefile
# Root Makefile

##@ Code Generation
new-model: ## Create new model (use NAME=ModelName)
	@echo "Generating model $(NAME)..."
	@uv run python scripts/generate_model.py $(NAME)

new-api: ## Create new API endpoint (use NAME=endpoint_name)
	@echo "Generating API endpoint $(NAME)..."
	@uv run python scripts/generate_api.py $(NAME)

new-test: ## Create test file (use NAME=test_name)
	@echo "Generating test file $(NAME)..."
	@uv run python scripts/generate_test.py $(NAME)
```

---

## Advanced Techniques

### Conditional Logic

Execute different logic based on environment or system.

```makefile
# Root Makefile

# Detect operating system
UNAME_S := $(shell uname -s)

pre-install::
ifeq ($(UNAME_S),Darwin)
	@echo "Installing macOS dependencies..."
	@brew install graphviz
else ifeq ($(UNAME_S),Linux)
	@echo "Installing Linux dependencies..."
	@sudo apt-get install -y graphviz
else
	@echo "Unsupported OS: $(UNAME_S)"
	@exit 1
endif

# Conditional based on environment variable
post-install::
ifdef CI
	@echo "Running in CI environment, skipping interactive setup"
else
	@echo "Running local setup..."
	@./scripts/interactive_setup.sh
endif
```

### Parallel Execution

Run multiple tasks in parallel.

```makefile
# Root Makefile

##@ Development
dev-all: ## Start all services in parallel
	@$(MAKE) -j4 dev-api dev-worker dev-frontend dev-db

dev-api:
	@uv run uvicorn myapp.api:app --reload

dev-worker:
	@uv run celery -A myapp.worker worker

dev-frontend:
	@npm run dev

dev-db:
	@docker-compose up postgres
```

### Dynamic Targets

Generate targets dynamically.

```makefile
# Root Makefile

# Define environments
ENVIRONMENTS := dev staging prod

# Generate deploy target for each environment
$(foreach env,$(ENVIRONMENTS),\
	$(eval deploy-$(env): ;\
		@echo "Deploying to $(env)..." ;\
		@./scripts/deploy.sh $(env)))

# Now you can run: make deploy-dev, make deploy-staging, make deploy-prod
```

### Function Calls

Use make functions for reusable logic.

```makefile
# Root Makefile

# Function to check if command exists
define check_command
	@command -v $(1) >/dev/null 2>&1 || { \
		echo "Error: $(1) is not installed"; \
		exit 1; \
	}
endef

##@ Validation
check-tools: ## Verify required tools are installed
	$(call check_command,docker)
	$(call check_command,kubectl)
	$(call check_command,helm)
	@echo "All required tools are installed ✓"
```

### Including External Makefiles

Modularize large Makefiles.

```makefile
# Root Makefile

# Include custom modules
-include makefiles/docker.mk
-include makefiles/kubernetes.mk
-include makefiles/terraform.mk

# Include Rhiza
include .rhiza/rhiza.mk
```

Then create `makefiles/docker.mk`:

```makefile
# makefiles/docker.mk

##@ Docker
docker-dev: ## Run development environment in Docker
	@docker-compose -f docker-compose.dev.yml up

docker-prod: ## Build production Docker image
	@docker build -t myapp:$(VERSION) .
```

---

## Real-World Examples

### Example 1: Machine Learning Project

```makefile
# Root Makefile for ML project

# ML-specific variables
DATA_DIR := data
MODELS_DIR := models
EXPERIMENT_NAME ?= default

post-install::
	@echo "Downloading datasets..."
	@uv run python scripts/download_data.py

##@ Machine Learning
train: ## Train model (use MODEL=model_name EXPERIMENT=name)
	@echo "Training $(MODEL) model..."
	@uv run python scripts/train.py \
		--model $(MODEL) \
		--experiment $(EXPERIMENT_NAME) \
		--data-dir $(DATA_DIR) \
		--output-dir $(MODELS_DIR)

evaluate: ## Evaluate model (use MODEL=model_name)
	@echo "Evaluating $(MODEL) model..."
	@uv run python scripts/evaluate.py \
		--model $(MODEL) \
		--data-dir $(DATA_DIR)/test

predict: ## Run inference (use MODEL=model_name INPUT=file)
	@uv run python scripts/predict.py \
		--model $(MODELS_DIR)/$(MODEL) \
		--input $(INPUT)

tune: ## Hyperparameter tuning (use MODEL=model_name)
	@echo "Tuning hyperparameters for $(MODEL)..."
	@uv run python scripts/tune.py --model $(MODEL)

experiment-track: ## Start MLflow tracking server
	@uv run mlflow server --host 0.0.0.0 --port 5000

##@ Data
download-data: ## Download datasets
	@uv run python scripts/download_data.py

preprocess-data: ## Preprocess raw data
	@uv run python scripts/preprocess.py \
		--input $(DATA_DIR)/raw \
		--output $(DATA_DIR)/processed

validate-data: ## Validate data quality
	@uv run python scripts/validate_data.py

# Include Rhiza
include .rhiza/rhiza.mk
```

### Example 2: Web API Project

```makefile
# Root Makefile for Web API project

# API configuration
API_HOST ?= 0.0.0.0
API_PORT ?= 8000
WORKERS ?= 4

post-install::
	@echo "Setting up database..."
	@uv run alembic upgrade head
	@echo "Loading fixtures..."
	@uv run python scripts/load_fixtures.py

##@ Development
dev: ## Start development server
	@uv run uvicorn myapi.main:app \
		--reload \
		--host $(API_HOST) \
		--port $(API_PORT)

dev-debug: ## Start development server with debugging
	@uv run python -m debugpy --listen 5678 \
		-m uvicorn myapi.main:app \
		--reload \
		--host $(API_HOST) \
		--port $(API_PORT)

shell: ## Open interactive Python shell with app context
	@uv run python scripts/shell.py

##@ Database
db-create: ## Create database
	@uv run python scripts/create_db.py

db-drop: ## Drop database (WARNING: destructive!)
	@echo "Are you sure? [y/N] " && read ans && [ $${ans:-N} = y ]
	@uv run python scripts/drop_db.py

db-migrate: ## Create new migration (use MSG="message")
	@uv run alembic revision --autogenerate -m "$(MSG)"

db-upgrade: ## Apply migrations
	@uv run alembic upgrade head

db-downgrade: ## Rollback migration
	@uv run alembic downgrade -1

db-seed: ## Seed database with test data
	@uv run python scripts/seed_db.py

##@ Production
start: ## Start production server
	@uv run gunicorn myapi.main:app \
		--workers $(WORKERS) \
		--worker-class uvicorn.workers.UvicornWorker \
		--bind $(API_HOST):$(API_PORT)

##@ API Testing
api-test: ## Run API tests
	@uv run pytest tests/api/ -v

api-load-test: ## Run load tests (use USERS=100 DURATION=60)
	@uv run locust \
		-f tests/load/locustfile.py \
		--users $(USERS) \
		--spawn-rate 10 \
		--run-time $(DURATION)s \
		--headless \
		--host http://localhost:$(API_PORT)

# Include Rhiza
include .rhiza/rhiza.mk
```

### Example 3: Data Pipeline Project

```makefile
# Root Makefile for data pipeline project

# Pipeline configuration
AIRFLOW_HOME := $(CURDIR)/airflow
export AIRFLOW_HOME

post-install::
	@echo "Initializing Airflow..."
	@uv run airflow db init
	@uv run airflow users create \
		--username admin \
		--password admin \
		--firstname Admin \
		--lastname User \
		--role Admin \
		--email admin@example.com

##@ Airflow
airflow-webserver: ## Start Airflow webserver
	@uv run airflow webserver --port 8080

airflow-scheduler: ## Start Airflow scheduler
	@uv run airflow scheduler

airflow-worker: ## Start Airflow worker
	@uv run airflow celery worker

airflow: ## Start all Airflow components
	@$(MAKE) -j3 airflow-webserver airflow-scheduler airflow-worker

##@ Pipelines
list-dags: ## List all DAGs
	@uv run airflow dags list

trigger-dag: ## Trigger DAG (use DAG=dag_id)
	@uv run airflow dags trigger $(DAG)

test-dag: ## Test DAG (use DAG=dag_id)
	@uv run airflow dags test $(DAG) $$(date +%Y-%m-%d)

backfill: ## Backfill DAG (use DAG=dag_id START=YYYY-MM-DD END=YYYY-MM-DD)
	@uv run airflow dags backfill $(DAG) \
		--start-date $(START) \
		--end-date $(END)

##@ Data
validate-schema: ## Validate data schemas
	@uv run python scripts/validate_schemas.py

check-data-quality: ## Run data quality checks
	@uv run python scripts/check_quality.py

# Include Rhiza
include .rhiza/rhiza.mk
```

---

## Best Practices

### 1. Documentation

- **Add comments** to explain complex logic
- **Use `##` syntax** for target descriptions (they appear in `make help`)
- **Group related targets** with `##@` section headers
- **Document variables** at the top of your Makefile

```makefile
# Root Makefile

# Configuration variables
API_HOST ?= 0.0.0.0  # Host for API server
API_PORT ?= 8000     # Port for API server
LOG_LEVEL ?= info    # Logging level (debug, info, warning, error)

##@ API Server
dev: ## Start development server (use API_PORT=8080 to override port)
	@echo "Starting API on $(API_HOST):$(API_PORT)..."
	@uv run uvicorn app.main:app --host $(API_HOST) --port $(API_PORT) --log-level $(LOG_LEVEL)
```

### 2. Error Handling

- **Check for required variables**
- **Validate prerequisites**
- **Provide helpful error messages**

```makefile
##@ Deployment
deploy: ## Deploy to server (requires SERVER and ENV variables)
ifndef SERVER
	$(error SERVER is not set. Use: make deploy SERVER=prod-01 ENV=production)
endif
ifndef ENV
	$(error ENV is not set. Use: make deploy SERVER=prod-01 ENV=production)
endif
	@echo "Deploying to $(SERVER) in $(ENV) environment..."
	@./scripts/deploy.sh $(SERVER) $(ENV)

check-docker:
	@command -v docker >/dev/null 2>&1 || { \
		echo "Error: docker is not installed"; \
		echo "Install from: https://docs.docker.com/get-docker/"; \
		exit 1; \
	}
```

### 3. Idempotency

Make targets idempotent (safe to run multiple times).

```makefile
post-install::
	@echo "Creating config file..."
	@if [ ! -f config/local.yaml ]; then \
		cp config/local.yaml.template config/local.yaml; \
	else \
		echo "Config already exists, skipping..."; \
	fi

db-create:
	@echo "Creating database..."
	@uv run python -c "from myapp.db import create_db; create_db(if_not_exists=True)"
```

### 4. DRY Principle

Avoid repetition by using variables and functions.

```makefile
# Bad: Repetitive
deploy-dev:
	@echo "Deploying to dev..."
	@./scripts/deploy.sh dev
	@./scripts/notify.sh dev

deploy-staging:
	@echo "Deploying to staging..."
	@./scripts/deploy.sh staging
	@./scripts/notify.sh staging

# Good: DRY
ENV ?= dev

deploy: ## Deploy to $(ENV) environment
	@echo "Deploying to $(ENV)..."
	@./scripts/deploy.sh $(ENV)
	@./scripts/notify.sh $(ENV)

deploy-dev: ENV=dev
deploy-dev: deploy

deploy-staging: ENV=staging
deploy-staging: deploy
```

### 5. Testing Custom Targets

Test your custom targets work correctly.

```makefile
##@ Testing
test-makefile: ## Test custom make targets
	@echo "Testing custom targets..."
	@$(MAKE) print-API_HOST
	@$(MAKE) print-API_PORT
	@echo "Targets exist: deploy, dev, db-migrate"
	@echo "All tests passed ✓"
```

### 6. Use Silent Prefix

Use `@` to suppress command echoing for cleaner output.

```makefile
# Bad: Noisy output
deploy:
	echo "Starting deployment..."
	./scripts/deploy.sh
	echo "Deployment complete!"

# Good: Clean output
deploy:
	@echo "Starting deployment..."
	@./scripts/deploy.sh
	@echo "Deployment complete!"
```

### 7. Provide Defaults

Provide sensible defaults for variables.

```makefile
# Defaults with override capability
HOST ?= localhost
PORT ?= 8000
ENV ?= dev
LOG_LEVEL ?= info

##@ Development
dev: ## Start development server
	@uv run uvicorn app:main --host $(HOST) --port $(PORT) --log-level $(LOG_LEVEL)
```

---

## Troubleshooting

### Issue: Hook Not Running

**Problem**: Your `post-install` hook doesn't seem to execute.

**Solution**: Ensure you're using double-colon syntax and it's defined before the `include` line:

```makefile
# Root Makefile

# ✅ Correct
post-install::
	@echo "Running custom setup..."

include .rhiza/rhiza.mk
```

### Issue: Variable Not Recognized

**Problem**: Variable defined in Makefile not available in target.

**Solution**: Define variables before the `include` line and use `export` if needed:

```makefile
# Root Makefile

# Define before include
MY_VAR = value
export MY_VAR  # Export if needed in sub-shells

include .rhiza/rhiza.mk

target:
	@echo "Variable: $(MY_VAR)"
```

### Issue: Target Not in Help

**Problem**: Custom target doesn't appear in `make help`.

**Solution**: Add `##` comment and ensure it comes after the target name and colon:

```makefile
##@ Custom Tasks
my-target: ## This description will appear in help
	@echo "Running target..."
```

### Issue: Circular Dependency

**Problem**: Get error about circular dependency.

**Solution**: Check for dependency loops:

```makefile
# ❌ Bad: Circular dependency
a: b
b: a

# ✅ Good: Linear dependency
a: b
b: c
c:
	@echo "Base target"
```

### Issue: Command Not Found in CI

**Problem**: Command works locally but fails in CI.

**Solution**: Check if command is available and install if needed:

```makefile
pre-install::
ifdef CI
	@echo "Installing CI dependencies..."
	@apt-get update && apt-get install -y build-essential
endif
```

---

## CodeQL Configuration

The CodeQL workflow (`.github/workflows/rhiza_codeql.yml`) performs security analysis on your code. However, **CodeQL requires GitHub Advanced Security**, which is:

- ✅ **Available for free** on public repositories
- ⚠️ **Requires GitHub Enterprise license** for private repositories

### Automatic Behavior

By default, the CodeQL workflow:

- **Runs automatically** on public repositories
- **Skips automatically** on private repositories (unless you have Advanced Security)

### Controlling CodeQL

You can override the default behavior using a repository variable:

1. Go to your repository → **Settings** → **Secrets and variables** → **Actions** → **Variables** tab
2. Create a new repository variable named `CODEQL_ENABLED`
3. Set the value:
   - `true` - Force CodeQL to run (use if you have Advanced Security on a private repo)
   - `false` - Disable CodeQL entirely (e.g., if it's causing issues)

For private repositories with Advanced Security enabled:

```bash
gh variable set CODEQL_ENABLED --body "true"
```

For users without Advanced Security, no action is needed. To disable it completely:

```bash
gh variable set CODEQL_ENABLED --body "false"
```

Or remove the workflow file entirely:

```bash
git rm .github/workflows/rhiza_codeql.yml
git commit -m "Remove CodeQL workflow"
```

---

## Documentation Customization

### Project Logo

The API documentation includes a logo in the sidebar. Override the default logo (`assets/rhiza-logo.svg`) by setting the `LOGO_FILE` variable in your `Makefile` or `local.mk`:

```makefile
LOGO_FILE := assets/my-custom-logo.png
```

### Custom pdoc Templates

Customize the look and feel of the API documentation by providing your own Jinja2 templates. Place custom templates in the `book/pdoc-templates` directory.

For example, to override the main module template, create `book/pdoc-templates/module.html.jinja2`.

See the [pdoc documentation on templates](https://pdoc.dev/docs/pdoc.html#edit-pdocs-html-template) for full details on overriding specific parts of the documentation.

For more details on customizing the documentation book, see [BOOK.md](BOOK.md).

---

## See Also

- [Quick Reference](QUICK_REFERENCE.md) - Command quick reference
- [Tools Reference](../reference/TOOLS_REFERENCE.md) - Comprehensive tool documentation
- [Makefile Cookbook](.rhiza/make.d/README.md) - Make recipes
- [rhiza-education Lesson 10: Customising Safely](https://github.com/Jebel-Quant/rhiza-education/blob/main/lessons/10-customizing-safely.md) - Tutorial overview of extension mechanisms and the template-managed file rule

---

*Last updated: 2026-02-15*
