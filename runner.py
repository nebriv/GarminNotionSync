"""Long-running scheduler/entrypoint for the Garmin → Notion sync.

Runs :func:`sync.sync_once` on an interval (default every 12h), sends notifications,
maintains a heartbeat file for health checks, and exposes a small HTTP server with
``/healthz``, ``/status`` and Prometheus ``/metrics``.

Environment:
  SYNC_INTERVAL_HOURS  how often to sync (default 12)
  SYNC_DAYS            rolling window passed to the sync (default 2)
  RUN_ON_START         run once immediately on startup (default true)
  METRICS_PORT         HTTP port for /healthz /status /metrics (default 9100; <=0 disables)
  TOKEN_WARN_DAYS      warn when the Garmin refresh token has fewer days left (default 21)
  DATA_DIR             base dir for state/heartbeat (default /data)
  plus everything read by sync.py / notify.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import threading
import time
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from dotenv import load_dotenv

import notify
from sync import setup_logging, sync_once

log = logging.getLogger("garminnotionsync.runner")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


DATA_DIR = os.getenv("DATA_DIR", "/data")
STATE_DIR = os.path.join(DATA_DIR, "state")
STATE_FILE = os.path.join(STATE_DIR, "last_run.json")
TOKEN_WARN_REPEAT_DAYS = 7


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest

    _PROM = True
    M_LAST_RUN = Gauge("gns_last_run_timestamp_seconds", "Unix time of the last sync run")
    M_LAST_SUCCESS = Gauge("gns_last_success_timestamp_seconds", "Unix time of the last successful run")
    M_CREATED = Gauge("gns_last_run_created", "Activities created in the last run")
    M_SKIPPED = Gauge("gns_last_run_skipped", "Activities skipped in the last run")
    M_FAILED = Gauge("gns_last_run_failed", "Activities failed in the last run")
    M_ATTACHED = Gauge("gns_last_run_attached", "Activity files (FIT/GPX) attached in the last run")
    M_DURATION = Gauge("gns_last_run_duration_seconds", "Duration of the last run")
    M_TOKEN_DAYS = Gauge("gns_garmin_token_days_remaining", "Estimated days until the Garmin refresh token expires")
    M_RUNS = Counter("gns_runs_total", "Total sync runs attempted")
    M_RUN_FAILURES = Counter("gns_run_failures_total", "Total sync runs that ended in failure/auth error")
    M_CREATED_TOTAL = Counter("gns_created_total", "Total Notion pages created")
    M_ATTACHED_TOTAL = Counter("gns_attached_total", "Total activity files attached to Notion")
except ImportError:  # pragma: no cover
    _PROM = False


def _update_metrics(result) -> None:
    if not _PROM:
        return
    now = time.time()
    M_LAST_RUN.set(now)
    M_CREATED.set(result.created)
    M_SKIPPED.set(result.skipped)
    M_FAILED.set(result.failed)
    M_ATTACHED.set(result.attached)
    M_DURATION.set(result.duration_s)
    M_CREATED_TOTAL.inc(result.created)
    M_ATTACHED_TOTAL.inc(result.attached)
    if result.token_days_remaining is not None:
        M_TOKEN_DAYS.set(result.token_days_remaining)
    if result.ok:
        M_LAST_SUCCESS.set(now)


# --------------------------------------------------------------------------- #
# State / heartbeat
# --------------------------------------------------------------------------- #


def _read_json(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _write_json_atomic(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _write_heartbeat(result) -> None:
    now = time.time()
    prev = _read_json(STATE_FILE) or {}
    payload = result.to_dict()
    payload["last_run_ts"] = now
    payload["last_success_ts"] = now if result.ok else prev.get("last_success_ts")
    _write_json_atomic(STATE_FILE, payload)


# --------------------------------------------------------------------------- #
# Token-expiry warning (throttled, persisted)
# --------------------------------------------------------------------------- #

_TOKEN_WARN_FILE = os.path.join(STATE_DIR, "token_warn.json")


def _maybe_warn_token(result) -> None:
    days = result.token_days_remaining
    warn_below = _env_int("TOKEN_WARN_DAYS", 21)
    if days is None or days > warn_below:
        return
    today = date.today()
    last = _read_json(_TOKEN_WARN_FILE) or {}
    last_day = last.get("warned_on")
    if last_day:
        try:
            if (today - date.fromisoformat(last_day)).days < TOKEN_WARN_REPEAT_DAYS:
                return
        except ValueError:
            pass
    notify.notify_warning(
        "Garmin token expiring soon",
        f"The Garmin refresh token has ~{days:.0f} days left. Re-run the one-time login "
        "bootstrap before it expires, or the sync will start failing:\n"
        "`docker compose run --rm garmin-notion-sync python sync.py --login`",
    )
    _write_json_atomic(_TOKEN_WARN_FILE, {"warned_on": today.isoformat(), "days_left": days})


# --------------------------------------------------------------------------- #
# The job
# --------------------------------------------------------------------------- #


def job() -> None:
    sync_days = _env_int("SYNC_DAYS", 2)
    if _PROM:
        M_RUNS.inc()
    try:
        result = sync_once(sync_days, dry_run=False)
    except Exception as e:  # noqa: BLE001  — never let the scheduler die
        log.exception("Unexpected error during sync run")
        if _PROM:
            M_RUN_FAILURES.inc()
        notify.notify_failure("❌ Garmin→Notion: unexpected error", str(e))
        return

    _update_metrics(result)
    _write_heartbeat(result)

    if result.ok:
        _maybe_warn_token(result)
        notify.notify_success(result)
    else:
        if _PROM:
            M_RUN_FAILURES.inc()
        notify.notify_failure(result)


# --------------------------------------------------------------------------- #
# HTTP server: /healthz /status /metrics
# --------------------------------------------------------------------------- #


def _health() -> tuple[bool, str]:
    data = _read_json(STATE_FILE)
    if data is None:
        return True, "starting"  # no run yet — healthy during start-period
    if data.get("auth_error"):
        return False, "auth_error"
    interval_h = _env_float("SYNC_INTERVAL_HOURS", 12)
    age = time.time() - float(data.get("last_run_ts") or 0)
    if age > interval_h * 3600 * 2:
        return False, f"stale ({age / 3600:.1f}h since last run)"
    return True, "degraded" if data.get("failed", 0) else "ok"


class _Handler(BaseHTTPRequestHandler):
    server_version = "garminnotionsync/1.0"

    def _respond(self, code: int, body: bytes, content_type: str = "text/plain; charset=utf-8") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path in ("/", "/healthz", "/health"):
            healthy, status = _health()
            self._respond(200 if healthy else 503, f"{status}\n".encode())
        elif path == "/status":
            data = _read_json(STATE_FILE) or {"status": "starting"}
            self._respond(200, json.dumps(data, indent=2).encode(), "application/json")
        elif path == "/metrics":
            if not _PROM:
                self._respond(503, b"prometheus_client not installed\n")
                return
            self._respond(200, generate_latest(), CONTENT_TYPE_LATEST)
        else:
            self._respond(404, b"not found\n")

    do_HEAD = do_GET  # noqa: N815

    def log_message(self, fmt: str, *args) -> None:  # quiet; route to debug
        log.debug("http %s - %s", self.address_string(), fmt % args)


def _start_http_server() -> ThreadingHTTPServer | None:
    port = _env_int("METRICS_PORT", 9100)
    if port <= 0:
        log.info("Metrics/health HTTP server disabled (METRICS_PORT<=0).")
        return None
    httpd = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    threading.Thread(target=httpd.serve_forever, name="http", daemon=True).start()
    log.info("Serving /healthz /status /metrics on :%d", port)
    return httpd


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Run a single sync and exit (no scheduler).")
    parser.add_argument("--test-notify", action="store_true", help="Send a test notification and exit.")
    args = parser.parse_args()

    load_dotenv()
    setup_logging()

    if args.test_notify:
        return 0 if notify.notify_test() else 1

    interval_h = _env_float("SYNC_INTERVAL_HOURS", 12)
    run_on_start = _env_bool("RUN_ON_START", True)

    if args.once:
        job()
        return 0

    _start_http_server()

    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.interval import IntervalTrigger

    scheduler = BlockingScheduler()
    scheduler.add_job(
        job,
        trigger=IntervalTrigger(hours=interval_h),
        id="sync",
        max_instances=1,   # never overlap a slow run with the next trigger
        coalesce=True,     # collapse missed runs into one
        misfire_grace_time=3600,
    )

    def _shutdown(signum, _frame):
        log.info("Received signal %s — shutting down.", signum)
        scheduler.shutdown(wait=False)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    log.info(
        "Scheduler started: every %sh, window=%sd, run_on_start=%s.",
        interval_h, _env_int("SYNC_DAYS", 2), run_on_start,
    )
    if run_on_start:
        job()

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):  # pragma: no cover
        pass
    log.info("Scheduler stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
