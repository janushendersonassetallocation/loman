#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y graphviz

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/setup-uv.sh"

echo "🚀 Generic Python .devcontainer environment ready!"
echo "🔧 Pre-commit hooks installed for code quality"
echo "📊 Graphviz installed"
echo "Marimo installed"
