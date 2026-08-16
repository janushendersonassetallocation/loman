## Makefile (repo-owned)
# Keep this file small. It can be edited without breaking template sync.

DEFAULT_AI_MODEL=claude-sonnet-4.6
LOGO_FILE=.rhiza/assets/rhiza-logo.svg
GH_AW_ENGINE ?= copilot  # Default AI engine for gh-aw workflows (copilot, claude, or codex)

# Override template default: fix quoting bug and typo (mkdocstring -> mkdocstrings)
# mkdocs_graphviz is pinned below 2: in 2.0 it stopped being a Markdown extension and became
# an MkDocs plugin, dropping the makeExtension entry point that `markdown_extensions` needs.
# We build with zensical, which only shims a fixed set of MkDocs plugins (autorefs,
# mkdocstrings, glightbox, macros, search) and would silently ignore the graphviz plugin,
# rendering every ```dot block as a plain code fence. Unpin once zensical can run it.
MKDOCS_EXTRA_PACKAGES = --with-editable . --with 'mkdocstrings[python]' --with 'mkdocs_graphviz<2'

# Always include the Rhiza API (template-managed)
include .rhiza/rhiza.mk

# Optional: developer-local extensions (not committed)
-include local.mk
