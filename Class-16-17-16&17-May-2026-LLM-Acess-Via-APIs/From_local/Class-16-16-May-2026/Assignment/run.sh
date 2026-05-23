#!/usr/bin/env bash
# Launches the app with the PROJECT venv's Python (not global/conda streamlit).
# Usage: ./run.sh   (from the Assignment folder)
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Walk up from this script to find the repo-root venv at <repo>/env/bin/python
PY=""
DIR="$HERE"
while [ "$DIR" != "/" ]; do
  if [ -x "$DIR/env/bin/python" ]; then
    PY="$DIR/env/bin/python"
    break
  fi
  DIR="$(dirname "$DIR")"
done

if [ -z "$PY" ]; then
  echo "Project venv python not found in any parent of: $HERE" >&2
  echo "Expected at <repo-root>/env/bin/python" >&2
  exit 1
fi

exec "$PY" -m streamlit run "$HERE/app.py" "$@"
