#!/usr/bin/env bash
# Install the Graphviz `dot` binary, which Loman's renderer shells out to.
#
# Extracted from the make target it replaces because that target had three
# problems, each of which turned a transient upstream hiccup into a confusing
# build failure:
#
#   1. It never retried. Chocolatey's community feed answered a `504 Gateway
#      Timeout` on 2026-08-09 and broke CI on master.
#   2. winget was only reached when Chocolatey was *absent*, never when it was
#      present and failed --- which is the case on every GitHub Windows runner.
#   3. It ignored the installer's exit status and never checked afterwards that
#      `dot` had actually appeared, so the build carried on to a test run that
#      reported 107 failures across three files. Nothing in that output said
#      "Graphviz is missing"; it read like a code regression.
#
# So: retry transient failures, fall back between package managers, and make
# the absence of `dot` a single loud error at install time rather than a
# cascade at test time.
set -uo pipefail

ATTEMPTS="${GRAPHVIZ_INSTALL_ATTEMPTS:-3}"
BACKOFF_SECONDS="${GRAPHVIZ_INSTALL_BACKOFF:-10}"

log() { printf '[graphviz] %s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

# Chocolatey installs into a shim directory that an already-running shell may
# not have on PATH yet, so look there too before declaring failure.
dot_present() {
    have dot && return 0
    for candidate in \
        "/c/ProgramData/chocolatey/bin/dot.exe" \
        "/c/Program Files/Graphviz/bin/dot.exe" \
        "/c/Program Files (x86)/Graphviz/bin/dot.exe"; do
        if [ -x "${candidate}" ]; then
            PATH="$(dirname "${candidate}"):${PATH}"
            export PATH
            log "found dot at ${candidate}; added its directory to PATH"
            return 0
        fi
    done
    return 1
}

install_once() {
    case "$(uname -s)" in
        Darwin)
            if have brew; then
                log "installing via Homebrew"
                brew install graphviz
                return $?
            fi
            log "Homebrew not found; cannot install automatically" >&2
            return 1
            ;;
        Linux)
            if have apt-get; then
                log "installing via apt-get"
                if [ "$(id -u)" -eq 0 ]; then
                    apt-get update && apt-get install -y graphviz
                else
                    sudo apt-get update && sudo apt-get install -y graphviz
                fi
                return $?
            fi
            if have dnf; then
                log "installing via dnf"
                sudo dnf install -y graphviz
                return $?
            fi
            log "no supported Linux package manager found" >&2
            return 1
            ;;
        *)
            # Windows: try both, in order, rather than only whichever is first
            # on PATH. A failure from one is exactly when the other matters.
            local installed=1
            if have choco; then
                log "installing via Chocolatey"
                choco install graphviz -y --no-progress && installed=0
            fi
            if [ "${installed}" -ne 0 ] && have winget; then
                log "Chocolatey did not supply dot; trying winget"
                winget install --id Graphviz.Graphviz -e --silent \
                    --accept-package-agreements --accept-source-agreements && installed=0
            fi
            if [ "${installed}" -ne 0 ] && ! have choco && ! have winget; then
                log "neither Chocolatey nor winget is available" >&2
            fi
            return "${installed}"
            ;;
    esac
}

main() {
    if dot_present; then
        log "already installed: $(dot -V 2>&1 | head -1)"
        return 0
    fi

    log "not found; installing (up to ${ATTEMPTS} attempts)"
    local attempt=1
    while [ "${attempt}" -le "${ATTEMPTS}" ]; do
        if [ "${attempt}" -gt 1 ]; then
            local pause=$(((attempt - 1) * BACKOFF_SECONDS))
            log "attempt ${attempt} of ${ATTEMPTS}, after ${pause}s"
            sleep "${pause}"
        fi
        install_once
        # The installer's exit status is a hint; whether `dot` runs is the fact.
        if dot_present; then
            log "installed: $(dot -V 2>&1 | head -1)"
            return 0
        fi
        attempt=$((attempt + 1))
    done

    cat >&2 <<'EOF'
[graphviz] ERROR: Graphviz's `dot` is not available after all install attempts.

Loman renders computation graphs by shelling out to `dot`, so the visualization
and widget tests cannot run without it. This is an environment problem, not a
test failure: continuing would report a hundred or more failures that all say
"InvocationException: GraphViz's executables not found".

Install it and try again:
  macOS         brew install graphviz
  Debian/Ubuntu sudo apt-get install graphviz
  Windows       choco install graphviz   (or: winget install Graphviz.Graphviz)
EOF
    return 1
}

main "$@"
