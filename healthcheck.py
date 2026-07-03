"""Container HEALTHCHECK probe.

Prefers the running HTTP server's ``/healthz``; falls back to inspecting the
heartbeat file if the server is disabled or unreachable. Exits 0 (healthy) or 1.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

DATA_DIR = os.getenv("DATA_DIR", "/data")
STATE_FILE = os.path.join(DATA_DIR, "state", "last_run.json")


def _http_health(port: int) -> bool | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=5) as resp:
            return resp.status == 200
    except Exception:  # noqa: BLE001 — server down/unreachable → fall back to file
        return None


def _file_health() -> bool:
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return True  # no run yet — healthy during the start-period
    if data.get("auth_error"):
        return False
    try:
        interval_h = float(os.getenv("SYNC_INTERVAL_HOURS", "12") or 12)
    except ValueError:
        interval_h = 12.0
    age = time.time() - float(data.get("last_run_ts") or 0)
    return age <= interval_h * 3600 * 2


def main() -> int:
    try:
        port = int(os.getenv("METRICS_PORT", "9100") or 9100)
    except ValueError:
        port = 9100
    if port > 0:
        result = _http_health(port)
        if result is not None:
            return 0 if result else 1
    return 0 if _file_health() else 1


if __name__ == "__main__":
    sys.exit(main())
