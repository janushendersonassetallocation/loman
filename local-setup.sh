#!/usr/bin/env bash
# Repo-owned environment provisioning, run by rhiza-task's `setup` task.
#
# `setup` is what `install` needs, and `install` is what `test` needs, so this
# runs before any gate that builds the venv --- which is the job the deleted
# `pre-install::` hook in .rhiza/make.d/00-additional-deps.mk used to do under
# the 0.x make framework. rhiza 1.x replaced that framework with the rhiza-task
# CLI, and this file, at a fixed name in the repository root, is the seam it
# leaves for a repo's own provisioning.
#
# The hook must be executable: rhiza-task fails rather than skipping when it
# exists without the execute bit, on the grounds that a hook someone wrote and
# expected to run is not something to pass over quietly.
set -euo pipefail

# Loman renders computation graphs by shelling out to Graphviz's `dot`, so the
# visualization and widget tests cannot run without it. The installer is a
# separate script because it retries transient failures, falls back between
# package managers, and verifies `dot` actually appeared rather than trusting
# the installer's exit status.
exec bash "$(dirname "$0")/scripts/install-graphviz.sh"
