## local.mk (repo-owned) -- make targets that are Loman's rather than the template's.
#
# rhiza 1.x replaced the make framework with the rhiza-task CLI. The Makefile it
# syncs is a shim that forwards every unmatched goal to that CLI, and nothing
# includes `.rhiza/make.d/` any more, so the two files that used to live there ---
# `00-additional-deps.mk` and `20-compat-tests.mk` --- moved here. `-include
# local.mk` is the seam core deliberately leaves open, and an explicit rule beats
# the shim's catch-all pattern rule, so anything defined here wins over a
# forwarded task of the same name.
#
# A `##` comment puts a target under "Repo-owned targets" in `make help`.

.PHONY: install-graphviz test-compat

##@ Loman Custom Tasks

# Loman renders graphs by shelling out to Graphviz's `dot`, so the visualization
# and widget tests cannot run without it. The installer lives in a script rather
# than inline here because it retries, falls back between package managers and
# verifies the result --- none of which is readable as a shell one-liner inside
# a make recipe.
#
# In scripts/ and not bin/: .gitignore excludes bin, which is where uv and other
# fetched programs land. A script placed there is silently never committed.
#
# `make test` no longer reaches this target, because `test` is the CLI's now: it
# needs `install`, which needs `setup`, which runs `local-setup.sh` --- and that
# calls the same script. This stays as the direct entry point for CI jobs that
# invoke pytest themselves instead of going through `make test`.
install-graphviz:  ## Install graphviz if not present, retrying transient failures
	@bash scripts/install-graphviz.sh

##@ Loman Compatibility Testing

# `make test` runs the suite against whatever the lock file resolves to, which
# is the newest supported pandas. Loman supports pandas 2 as well, and the two
# differ in the default temporal resolution --- nanoseconds versus microseconds.
# That difference is exactly what the serialization format records rather than
# assumes, so it needs a check that actually runs against the other end of the
# range; otherwise the support is accidental and breaks silently.
#
# The pin below is an exact series glob, not a lower bound. `--with "pandas>=2.0"`
# reads like "test the old one" and is not: uv resolves the newest version
# satisfying it, so both legs of the matrix installed the same pandas 2.3.3 and
# the job reported green while testing one version twice.
#
# Kept out of `make test` deliberately. It builds a separate environment, so
# folding it into the default target would roughly double the time of the loop
# people run most often. CI runs it as its own job (.github/workflows/compatibility.yml).

# Oldest supported pandas series, matching the floor declared in pyproject.toml.
# Names the series only: the pin below turns it into an exact `==2.2.*`.
PANDAS_MIN ?= 2.2

# The framework used to define this; the shim does not, so it is spelled out.
TESTS_FOLDER ?= tests

# Only the modules whose behaviour depends on the pandas version. The rest of
# the suite is version-independent and already covered by `make test`. The
# property tests are included because the value model --- datetime resolution
# above all --- is precisely what differs between the two pandas versions.
COMPAT_TESTS = \
  $(TESTS_FOLDER)/test_serialization.py \
  $(TESTS_FOLDER)/test_serialization_behaviour.py \
  $(TESTS_FOLDER)/test_containers.py \
  $(TESTS_FOLDER)/test_compression.py \
  $(TESTS_FOLDER)/test_byos.py \
  $(TESTS_FOLDER)/test_pandas_compat.py \
  $(TESTS_FOLDER)/test_serialization_properties.py

# Depends on `$(UV)` rather than the CLI's `install`: `uv run --isolated`
# resolves its own environment from pyproject.toml, so the project venv is not
# needed and building it first would only cost time. The shim defines `$(UV)`
# and bootstraps it from `$(UVX)` when the machine has neither.
test-compat: $(UV)  ## Run the version-sensitive tests against the oldest supported pandas
	@printf '[Loman] Testing against pandas ~=$(PANDAS_MIN)...\n'
	@$(UV) run --isolated \
	  --with "pandas==$(PANDAS_MIN).*" \
	  --with pytest \
	  --with hypothesis \
	  pytest $(COMPAT_TESTS) -q

##@ Releasing and Versioning

# Restored from the deleted .rhiza/make.d/releasing.mk. rhiza 1.x dropped these
# targets: the task runner has no release task at all, and the flow moved to a
# Claude Code skill (`/rhiza:release`). The underlying tool did not move --- it is
# still the `rhiza-tools` CLI these targets call --- so the local flow stays
# reproducible from a shell, without an editor in the loop.
#
# .github/workflows/rhiza_release.yml did NOT need excluding from the sync. Its
# trigger is unchanged from 0.x (`push: tags: v*`, plus a new workflow_call entry
# point that does not alter it), so a tag pushed by `make release` starts the same
# workflow it always did.

# Pinned, not floored. The 0.x file asked for `rhiza-tools>=0.3.3`, which resolves
# to whatever is newest --- the same unpinned-tool trap that had the typecheck gate
# reporting ty's release date rather than the code. 0.8.1 renamed a flag these
# targets pass (`--with-bump` became `--bump TYPE`), so an open floor would have
# broken `make publish` silently somewhere between 0.3.3 and now.
RHIZA_TOOLS ?= rhiza-tools==0.8.1

# `-c pyproject.toml`, because rhiza-tools still defaults to `.rhiza/.cfg.toml`,
# which 1.x deletes. Without it every target fails with a bare ENOENT naming a path
# nothing in the repo mentions any more. The bumpversion config is the
# [tool.bumpversion] table in pyproject.toml now.
RHIZA_TOOLS_ARGS = -c pyproject.toml

# DRY_RUN=1 previews without applying, as before.
_DRY_RUN := $(if $(DRY_RUN),--dry-run,)

# Bump type for `publish`; 0.8.1 wants it explicitly where the old CLI prompted.
BUMP ?= PATCH

.PHONY: bump release publish rollback release-status

# Prompts for the bump type and has no flag to supply one, so it needs a terminal
# --- as it did under 0.x. `publish BUMP=...` is the scriptable path.
bump: $(UVX)  ## Bump the project version, prompting for the type (DRY_RUN=1)
	@$(UVX) "$(RHIZA_TOOLS)" bump $(RHIZA_TOOLS_ARGS) $(_DRY_RUN)
	@[ -n "$(DRY_RUN)" ] || $(UV) lock

# One behaviour did change, and it is the tool's doing rather than the template's.
# Under rhiza-tools 0.3.3 `release` tagged the version already in pyproject.toml and
# `publish` was the bump-then-tag variant. 0.8.1 states that "a release always bumps
# the version before tagging" and offers no way to tag without bumping, so the two
# targets below are now the same command: `release` lets it ask for the bump type,
# `publish` passes one. Tagging an unbumped version means tagging a commit by hand.
release: $(UVX)  ## Bump (prompting for the type), tag and push (DRY_RUN=1)
	@$(UVX) "$(RHIZA_TOOLS)" release $(RHIZA_TOOLS_ARGS) $(_DRY_RUN)

publish: $(UVX)  ## Bump, tag and push without prompting (BUMP=PATCH|MINOR|MAJOR, DRY_RUN=1)
	@$(UVX) "$(RHIZA_TOOLS)" release --bump $(BUMP) -y $(RHIZA_TOOLS_ARGS) $(_DRY_RUN)

rollback: $(UVX)  ## Undo a release tag and/or version bump (DRY_RUN=1)
	@$(UVX) "$(RHIZA_TOOLS)" rollback $(RHIZA_TOOLS_ARGS) $(_DRY_RUN)

# The two halves the 0.x target paged together. Both are rhiza-task tasks now, so
# they come from the shim's catch-all rather than being reimplemented here.
release-status:  ## Show the release workflow status and the latest release
	@{ $(MAKE) --no-print-directory workflow-status; printf '\n'; \
	   $(MAKE) --no-print-directory latest-release; } 2>&1 | $${PAGER:-cat}
