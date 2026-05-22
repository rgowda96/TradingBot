#!/bin/bash
# SessionStart hook: install Python dependencies for the trading bot so the
# session can run bot.py immediately.
set -euo pipefail

# Only needed in the Claude Code web/remote environment.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

python3 -m pip install --quiet --disable-pip-version-check -r requirements.txt

echo "session-start: Python dependencies installed."
