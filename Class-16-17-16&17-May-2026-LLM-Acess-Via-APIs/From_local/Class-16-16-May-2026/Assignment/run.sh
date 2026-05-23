#!/usr/bin/env bash
# Launches the app with the PROJECT venv's Python (not global/conda streamlit).
# Usage: ./run.sh   (from the Assignment folder)
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/../../env/bin/python"

if [ ! -x "$PY" ]; then
  echo "Project venv python not found at: $PY" >&2
  exit 1
fi

exec "$PY" -m streamlit run "$HERE/app.py" "$@"
