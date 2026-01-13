## Makefile (repo-owned)
# Keep this file small. It can be edited without breaking template sync.

# Always include the Rhiza API (template-managed)
include .rhiza/rhiza.mk

# Optional: repo extensions (committed)
-include .rhiza/make.d/*.mk

# Optional: developer-local extensions (not committed)
-include local.mk
