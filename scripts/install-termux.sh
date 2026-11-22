#!/data/data/com.termux/files/usr/bin/sh
# Install ZIA dependencies on Termux (Android ARM64 / Sony Xperia 1 III)
set -eu

pkg update && pkg upgrade -y
pkg install -y python python-pip sqlite libsqlite clang make cmake git \
    pulseaudio termux-api openssh espeak-ng jq

python -m pip install --upgrade pip
pip install -r requirements.txt

# Create zia wrapper in Termux PATH
BIN_DIR="${PREFIX:-/data/data/com.termux/files/usr/local}/bin"
mkdir -p "$BIN_DIR"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cat > "$BIN_DIR/zia" <<EOF
#!/data/data/com.termux/files/usr/bin/sh
exec python "$REPO_DIR/main.py" "\$@"
EOF
chmod +x "$BIN_DIR/zia"

# Prepare secure secrets template outside repo
mkdir -p "$HOME/.config/zia"
chmod 700 "$HOME/.config/zia"
if [ ! -f "$HOME/.config/zia/secrets.env" ]; then
    cat > "$HOME/.config/zia/secrets.env" <<'EOF'
# Add your free-tier API keys here and keep chmod 600
GROQ_API_KEY=
GEMINI_API_KEY=
EOF
    chmod 600 "$HOME/.config/zia/secrets.env"
fi

echo "Install complete. Edit ~/.config/zia/secrets.env and run: zia doctor"
