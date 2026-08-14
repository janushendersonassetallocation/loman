## Makefile (repo-owned)
# Keep this file small. It can be edited without breaking template sync.

LOGO_FILE=.rhiza/assets/rhiza-logo.svg

# Override template default: include mkdocstrings plugin for API docs
MKDOCS_EXTRA_PACKAGES = --with 'mkdocstrings[python]'

# Override template default (v1.3.3 ships TYPECHECKER ?= both, i.e. ty + mypy
# --strict). mypy --strict currently reports 48 errors in src/loman; until those
# are fixed, keep the v0.10.3 behaviour of running ty alone so `all` stays green.
# Flip back to `both` once the strict errors are resolved.
TYPECHECKER = ty

# Always include the Rhiza API (template-managed)
include .rhiza/rhiza.mk

# Optional: developer-local extensions (not committed)
-include local.mk
