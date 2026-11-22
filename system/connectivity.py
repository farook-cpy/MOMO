"""Network connectivity and Termux Wi-Fi diagnostics."""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
from typing import Any


def get_termux_wifi_info() -> dict[str, Any] | None:
    """Query termux-wifi-connectioninfo if available."""
    if not shutil.which("termux-wifi-connectioninfo"):
        return None
    try:
        proc = subprocess.run(
            ["termux-wifi-connectioninfo"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return dict(json.loads(proc.stdout))
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
    return None


def check_internet_connectivity(host: str = "1.1.1.1", port: int = 443, timeout: float = 3.0) -> bool:
    """Verify outbound TCP connectivity without relying on ICMP ping."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def get_network_status() -> dict[str, Any]:
    """Return structured connectivity + Wi-Fi metadata."""
    online = check_internet_connectivity()
    wifi = get_termux_wifi_info()
    status: dict[str, Any] = {
        "online": online,
        "interface": "wifi" if wifi else "unknown",
    }
    if wifi:
        status.update(
            {
                "ssid": wifi.get("ssid"),
                "ip": wifi.get("ip"),
                "rssi": wifi.get("rssi"),
                "link_speed_mbps": wifi.get("link_speed_mbps"),
                "supplicant_state": wifi.get("supplicant_state"),
            }
        )
    return status
