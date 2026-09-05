#!/bin/sh
# The repository's own environment hook, run by `rhiza-task`'s `setup` task.
#
# Repo-owned: the name and location are fixed by the CLI (repository root,
# `local-setup.sh`), but nothing upstream ships or overwrites the file. It is not in
# `.rhiza/template.lock`'s `files:` block.
#
# **When it runs.** `setup` is a prerequisite of `install`, and `install` is a
# prerequisite of essentially every gate — so this runs on local `make test`, on every
# CI job that invokes `uvx rhiza-task <gate>`, and on the devcontainer's `make install`,
# with no workflow edit between them. It must therefore be fast and idempotent: the
# common case is that graphviz is already there and this exits within a few
# milliseconds.
#
# **What it provisions.** graphviz, for `loman`'s plot rendering —
# `docs/notebooks/notebook-extras.py` builds a `loman.Computation` and draws it,
# which shells out to `dot`. Nothing in `src/` or `tests/` needs it.
#
# **Why a script and not a `system-packages = [...]` setting.** The CLI decides *when*
# provisioning happens; the repository decides *what*, because only the repository knows
# which platforms it builds on and how a package is spelled on each. `graphviz` happens
# to be the same word to apt and to brew; plenty of packages are not, and a list cannot
# express "download this tarball".
#
# **This replaces `.rhiza/scripts/customisations/build-extras.sh`**, which did the same
# job for the make layer and was orphaned when v1.4.0 retired it — nothing had called it
# since. The old file was apt-only and assumed `sudo`; this one handles both, and both
# root and non-root.
#
# **A missing package manager is a warning, not a failure.** Every gate reaches this
# hook, and graphviz is needed by one notebook — so failing here would block `make test`
# on a machine that has no use for the dependency. What it must not do is fail *quietly*:
# the warning names what is missing and what will break.
#
# **Windows cannot run this at all.** `rhiza-task`'s `setup` execs the hook directly, and
# Windows refuses to start a `.sh`, which the task reports as a failure. See CLAUDE.md.
set -eu

if command -v dot >/dev/null 2>&1; then
    echo "[INFO] graphviz already present: $(dot -V 2>&1)"
    exit 0
fi

# `sudo` is absent in a root container and required on a GitHub runner. Resolve it once
# rather than guessing per-branch.
if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
elif command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
else
    echo "[WARN] graphviz is missing and this is a non-root user with no sudo."
    echo "[WARN] Install it by hand; loman's plots will not render until you do."
    exit 0
fi

echo "[INFO] graphviz not found; installing"

if command -v apt-get >/dev/null 2>&1; then
    # Debian/Ubuntu: the GitHub runners and the devcontainer image.
    DEBIAN_FRONTEND=noninteractive $SUDO apt-get update -qq
    DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y --no-install-recommends graphviz
elif command -v brew >/dev/null 2>&1; then
    # macOS. brew refuses to run under sudo, so call it as the invoking user.
    brew install graphviz
elif command -v dnf >/dev/null 2>&1; then
    $SUDO dnf install -y graphviz
elif command -v apk >/dev/null 2>&1; then
    $SUDO apk add --no-cache graphviz
else
    echo "[WARN] no supported package manager found (apt-get, brew, dnf, apk)."
    echo "[WARN] Install graphviz by hand; loman's plots will not render until you do."
    exit 0
fi

# The package manager exiting 0 is not the same as `dot` being on PATH, and a hook that
# reports success without checking is the silent-green outcome this hook exists to remove.
if command -v dot >/dev/null 2>&1; then
    echo "[INFO] graphviz installed: $(dot -V 2>&1)"
else
    echo "[ERROR] the package manager succeeded but 'dot' is still not on PATH" >&2
    exit 1
fi
