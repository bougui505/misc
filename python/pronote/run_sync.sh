#!/usr/bin/env bash
set -euo pipefail

# Définir le PATH pour uv
export PATH="$HOME/.local/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Exécution de la synchronisation avec uv
uv run sync_to_gcalendar.py "$@"
