# ZIA Setup Guide (Windows Dev + Sony Xperia 1 III Termux)

## 1. Windows Development Environment
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -v
python main.py doctor
python main.py chat
```

Store API keys outside the repo in `%USERPROFILE%\.config\zia\secrets.env` or environment variables:
```powershell
$env:GROQ_API_KEY="gsk_..."
$env:GEMINI_API_KEY="AIza..."
```

## 2. Sony Xperia 1 III Preparation (Broken Display Options)
Since the Xperia 1 III display is physically broken, complete initial setup via one of:
1. **USB-C to HDMI adapter + USB mouse/keyboard** (Xperia 1 III supports USB-C DisplayPort Alt Mode), OR
2. **ADB + `scrcpy`** over USB if USB Debugging was already authorized.

Install matching builds from **F-Droid** (never mix Google Play and F-Droid builds):
- `Termux`
- `Termux:API`
- `Termux:Boot`

## 3. Termux Installation
```sh
git clone <repo-url> ~/zia
cd ~/zia
bash scripts/install-termux.sh
```

Configure SSH key authentication for remote maintenance (port `8022` by default):
```sh
mkdir -p ~/.ssh && chmod 700 ~/.ssh
# Append your Windows public key to ~/.ssh/authorized_keys (chmod 600)
sshd
```
Disable password auth in `$PREFIX/etc/ssh/sshd_config` (`PasswordAuthentication no`) once key auth is verified.

## 4. Android Permissions & Power Management
On the Xperia 1 III:
1. Open **Termux:API** and **Termux:Boot** once manually.
2. **Settings > Apps > Termux:API > Permissions > Microphone** -> *Allow while using the app*; disable *Remove permissions if app is unused*.
3. **Settings > Apps > Termux / Termux:API / Termux:Boot > Battery** -> **Unrestricted**.
4. **Settings > Battery > STAMINA Mode** -> **Off**; **Adaptive Battery** -> **Off**.
5. From Windows via ADB (Android 12/13 Phantom Process Killer mitigation):
   ```sh
   adb shell settings put global activity_manager_constants max_phantom_processes=2147483647
   adb shell /system/bin/device_config put android.background_process_freezer upm_enabled false
   ```
   *(Verify persistence after reboot; re-apply via ADB if `device_config` resets.)*

## 5. Run Audio Feasibility Probe
```sh
bash scripts/probe-audio.sh
```
Only enable `wakeword_backend` other than `"manual"` in `config/settings.toml` after confirming `parec` and screen-off recording work reliably on the Xperia 1 III.
