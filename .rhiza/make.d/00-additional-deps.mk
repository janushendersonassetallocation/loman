## .rhiza/make.d/00-additional-deps.mk - Custom Repository Tasks
# This file is repo-owned: it is not in .rhiza/template.lock, so `make sync`
# leaves it alone.

.PHONY: pre-install install-graphviz

##@ Loman Custom Tasks

# Loman renders graphs by shelling out to Graphviz's `dot`, so the visualization
# and widget tests cannot run without it. The installer lives in a script rather
# than inline here because it retries, falls back between package managers and
# verifies the result --- none of which is readable as a shell one-liner inside
# a make recipe.
#
# In scripts/ and not bin/: .gitignore excludes bin, which is where uv and other
# fetched programs land. A script placed there is silently never committed.
install-graphviz:  ## Install graphviz if not present, retrying transient failures
	@bash scripts/install-graphviz.sh

pre-install:: ## Custom pre-install tasks for Loman
	@printf "${BLUE}[Loman] Running custom pre-install tasks...${RESET}\n"
	@$(MAKE) install-graphviz
