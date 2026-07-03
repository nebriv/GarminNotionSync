"""Notifications for the Garmin → Notion sync, via Apprise.

Set ``APPRISE_URLS`` to one or more Apprise URLs (comma/space/newline separated) to
enable notifications — e.g. a Discord webhook (``discord://id/token``) and/or email
(``mailto://user:pass@host``). See https://github.com/caronc/apprise/wiki for the full
list of supported targets. When unset, all notify calls are no-ops.

Success notifications honor ``NOTIFY_ON_SUCCESS``:
  * ``new``    (default) — only when the run created new Notion rows
  * ``always`` — every successful run
  * ``never``  — never (failures still notify)
"""

from __future__ import annotations

import logging
import os
import re
import socket
from datetime import datetime, timezone

try:  # apprise is optional at import time so tests/CLI work without it
    import apprise
except ImportError:  # pragma: no cover
    apprise = None  # type: ignore[assignment]

log = logging.getLogger("garminnotionsync.notify")

_HOST = socket.gethostname()
_apprise_cache: "apprise.Apprise | None" = None
_warned_missing = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _urls() -> list[str]:
    raw = os.getenv("APPRISE_URLS", "")
    return [p for p in re.split(r"[\s,]+", raw.strip()) if p]


def _redact(url: str) -> str:
    """Hide credentials/tokens when logging an Apprise URL."""
    return re.sub(r"//[^@/]+@", "//***@", re.sub(r"/[A-Za-z0-9_-]{12,}", "/***", url))


def _client() -> "apprise.Apprise | None":
    global _apprise_cache, _warned_missing
    if _apprise_cache is not None:
        return _apprise_cache

    urls = _urls()
    if not urls:
        if not _warned_missing:
            log.info("APPRISE_URLS not set — notifications disabled.")
            _warned_missing = True
        return None
    if apprise is None:
        if not _warned_missing:
            log.warning("APPRISE_URLS set but 'apprise' is not installed — notifications disabled.")
            _warned_missing = True
        return None

    ap = apprise.Apprise()
    for u in urls:
        if not ap.add(u):
            log.warning("Ignoring invalid Apprise URL: %s", _redact(u))
    _apprise_cache = ap
    return ap


def _notify_type(kind: str):
    if apprise is None:
        return None
    return {
        "success": apprise.NotifyType.SUCCESS,
        "failure": apprise.NotifyType.FAILURE,
        "warning": apprise.NotifyType.WARNING,
        "info": apprise.NotifyType.INFO,
    }.get(kind, apprise.NotifyType.INFO)


def _send(title: str, body: str, kind: str) -> bool:
    ap = _client()
    if ap is None:
        return False
    try:
        ok = ap.notify(title=title, body=body, notify_type=_notify_type(kind))
        if not ok:
            log.warning("Apprise reported a delivery failure for: %s", title)
        return bool(ok)
    except Exception:  # noqa: BLE001
        log.exception("Failed to send notification: %s", title)
        return False


def _footer(result) -> str:
    bits = [f"host: {_HOST}", f"at: {_now()}"]
    if result is not None and result.token_days_remaining is not None:
        bits.append(f"garmin token: ~{result.token_days_remaining:.0f}d left")
    return "\n".join(bits)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def notify_success(result) -> bool:
    """Notify about a successful run, subject to ``NOTIFY_ON_SUCCESS`` policy."""
    policy = os.getenv("NOTIFY_ON_SUCCESS", "new").lower()
    if policy == "never":
        return False
    if policy == "new" and result.created == 0:
        log.debug("No new activities — skipping success notification (NOTIFY_ON_SUCCESS=new).")
        return False

    noun = "activity" if result.created == 1 else "activities"
    title = f"✅ Garmin→Notion: {result.created} new {noun}"

    counts = f"**{result.created}** new · {result.skipped} skipped · {result.failed} failed"
    if getattr(result, "attached", 0):
        counts += f" · {result.attached} file(s) attached"
    if getattr(result, "metrics", 0):
        counts += f" · {result.metrics} with metrics"
    lines = [
        counts,
        f"window: {result.window_start} → {result.window_end}  ({result.duration_s:.0f}s)",
    ]
    if result.created_labels:
        lines.append("")
        shown = result.created_labels[:15]
        lines += [f"• {lbl}" for lbl in shown]
        if len(result.created_labels) > len(shown):
            lines.append(f"…and {len(result.created_labels) - len(shown)} more")
    lines += ["", _footer(result)]
    return _send(title, "\n".join(lines), "success")


def notify_failure(result_or_title, body: str | None = None) -> bool:
    """Notify about a failure.

    Accepts either a :class:`SyncResult` (formats counts/failures automatically) or
    a ``(title, body)`` pair for arbitrary errors.
    """
    if body is not None or isinstance(result_or_title, str):
        title = str(result_or_title)
        return _send(title, (body or "") + "\n\n" + _footer(None), "failure")

    result = result_or_title
    if result.auth_error:
        title = "❌ Garmin→Notion: authentication error"
        detail = result.auth_error
    else:
        title = f"❌ Garmin→Notion: {result.failed} failure(s)"
        detail = "\n".join(f"• {f}" for f in result.failures[:15]) or "(no detail)"
        if len(result.failures) > 15:
            detail += f"\n…and {len(result.failures) - 15} more"

    lines = [
        f"{result.created} new · {result.skipped} skipped · **{result.failed} failed**",
        f"window: {result.window_start} → {result.window_end}  ({result.duration_s:.0f}s)",
        "",
        detail,
        "",
        _footer(result),
    ]
    return _send(title, "\n".join(lines), "failure")


def notify_warning(title: str, body: str) -> bool:
    return _send(f"⚠️ {title}", body + "\n\n" + _footer(None), "warning")


def notify_test() -> bool:
    """Send a test message to verify configuration. Returns True if delivered."""
    if _client() is None:
        log.error("No Apprise targets configured (set APPRISE_URLS) — nothing to test.")
        return False
    ok = _send(
        "🔔 Garmin→Notion test notification",
        "If you can read this, notifications are configured correctly.\n\n" + _footer(None),
        "info",
    )
    log.info("Test notification %s.", "sent" if ok else "FAILED")
    return ok
