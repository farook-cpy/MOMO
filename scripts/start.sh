#!/data/data/com.termux/files/usr/bin/sh
# Start ZIA headless daemon in Termux with wake lock
set -eu

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

if command -v termux-wake-lock >/dev/null 2>&1; then
    termux-wake-lock
fi

exec python "$REPO_DIR/main.py" start --daemon
