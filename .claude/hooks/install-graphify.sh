#!/usr/bin/env bash
# Make sure the graphify CLI is available (cloud sessions start without it).
command -v graphify >/dev/null 2>&1 && exit 0
export PATH="$HOME/.local/bin:$PATH"
command -v graphify >/dev/null 2>&1 && exit 0

if command -v uv >/dev/null 2>&1; then
  uv tool install -q graphifyy >/dev/null 2>&1
elif command -v pipx >/dev/null 2>&1; then
  pipx install graphifyy >/dev/null 2>&1
else
  pip install -q --user graphifyy >/dev/null 2>&1 \
    || pip install -q --user --break-system-packages graphifyy >/dev/null 2>&1
fi

if [ -n "$CLAUDE_ENV_FILE" ]; then
  echo "export PATH=\"$HOME/.local/bin:\$PATH\"" >> "$CLAUDE_ENV_FILE"
fi
exit 0
