#!/data/data/com.termux/files/usr/bin/sh
# Configure Termux:Boot startup script (~/.termux/boot/start-zia)
set -eu

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BOOT_DIR="$HOME/.termux/boot"
mkdir -p "$BOOT_DIR"

cat > "$BOOT_DIR/start-zia" <<EOF
#!/data/data/com.termux/files/usr/bin/sh
termux-wake-lock
sshd
sleep 5
cd "$REPO_DIR"
python "$REPO_DIR/main.py" start --daemon --no-voice
EOF
chmod +x "$BOOT_DIR/start-zia"

echo "Created $BOOT_DIR/start-zia"
echo "IMPORTANT: Open Termux:Boot once from Android app launcher and disable battery optimization."
echo "Note: Voice loop starts with --no-voice by default after reboot because Android 11+ may block mic before unlock."
