## .rhiza/make.d/10-custom-task.mk - Custom Repository Tasks
# This file example shows how to add new targets.

.PHONY: pre-install

##@ Loman Custom Tasks

install-graphviz:  ## Install graphviz if not present
	@if ! command -v dot >/dev/null 2>&1; then \
		echo "Graphviz not found. Attempting installation..."; \
		if [ "$$(uname)" = "Darwin" ]; then \
			if command -v brew >/dev/null 2>&1; then \
				echo "Installing via Homebrew..."; \
				brew install graphviz; \
			else \
				echo "Error: Homebrew not found. Please install Graphviz manually." >&2; \
				exit 1; \
			fi; \
		elif command -v apt-get >/dev/null 2>&1; then \
			echo "Installing via apt-get..."; \
			if [ "$$(id -u)" -eq 0 ]; then \
				apt-get update && apt-get install -y graphviz; \
			else \
				sudo apt-get update && sudo apt-get install -y graphviz; \
			fi; \
		else \
			echo "Warning: Could not detect a supported package manager. Please install Graphviz manually." >&2; \
			exit 1; \
		fi; \
	else \
		echo "graphviz is already installed, skipping installation."; \
	fi

pre-install:: ## Custom pre-install tasks for Loman
	@printf "${BLUE}[Loman] Running custom pre-install tasks...${RESET}\n"
	@$(MAKE) install-graphviz

