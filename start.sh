#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="$PYTHON_BIN"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON="$ROOT_DIR/.venv/bin/python"
else
  PYTHON="python3"
fi

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  printf 'Python executable not found: %s\n' "$PYTHON" >&2
  printf 'Create the virtual environment as described in README.md, or set PYTHON_BIN.\n' >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  printf 'npm was not found. Install Node.js and npm as described in README.md.\n' >&2
  exit 1
fi

if [[ ! -x "$ROOT_DIR/frontend/node_modules/.bin/vite" ]]; then
  printf 'Frontend dependencies are missing. Run "npm ci" in frontend/ first.\n' >&2
  exit 1
fi

printf 'Starting FoodAdvisor API at http://localhost:8000\n'
(cd "$ROOT_DIR" && "$PYTHON" -m uvicorn app:app --app-dir "$ROOT_DIR/backend" --host 0.0.0.0 --port 8000 --reload) &
API_PID=$!

printf 'Starting FoodAdvisor web app at http://localhost:5173\n'
(cd "$ROOT_DIR/frontend" && npm run dev -- --host 0.0.0.0) &
FRONTEND_PID=$!

cleanup() {
  trap - EXIT INT TERM
  kill "$API_PID" "$FRONTEND_PID" 2>/dev/null || true
  wait "$API_PID" 2>/dev/null || true
  wait "$FRONTEND_PID" 2>/dev/null || true
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait -n "$API_PID" "$FRONTEND_PID"
