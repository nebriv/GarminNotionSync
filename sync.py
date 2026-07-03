"""Sync Garmin Connect hikes and runs to the Notion Exercise Log database.

Can run as a one-shot CLI (``python sync.py``) or be driven on a schedule by
``runner.py``. The reusable engine is :func:`sync_once`, which returns a
:class:`SyncResult` instead of printing, so callers can log, notify, and export
metrics from a single run.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import re
import sys
import time
import zipfile
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from garminconnect import (
    Garmin,
    GarminConnectAuthenticationError,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
)
from notion_client import Client as Notion
from notion_client.errors import APIResponseError

log = logging.getLogger("garminnotionsync")

HIKE_TYPES = {"hiking"}
RUN_TYPES = {"running", "trail_running"}
DEFAULT_TRACKED_TYPES = HIKE_TYPES | RUN_TYPES

FEEL_LABELS = {0: "Very Weak", 25: "Weak", 50: "Normal", 75: "Strong", 100: "Very Strong"}

TRAINING_EFFECT_LABELS = {
    "AEROBIC_BASE": "Aerobic Base",
    "TEMPO": "Tempo",
    "LACTATE_THRESHOLD": "Lactate Threshold",
    "VO2MAX": "VO2 Max",
    "ANAEROBIC_CAPACITY": "Anaerobic Capacity",
    "SPEED": "Speed",
    "RECOVERY": "Recovery",
    "NONE": "No Benefit",
}

# Garmin token store + tokens are loaded/persisted here. Override in containers so the
# tokens land on a mounted volume and survive restarts.
DEFAULT_TOKEN_DIR = "~/.garminconnect"

# Garmin issues a refresh token good for ~1 year; warn before it lapses.
GARMIN_REFRESH_TOKEN_LIFETIME_DAYS = 365

# --- Activity file attachments (FIT + GPX) -------------------------------------
# Notion "Files & media" columns the raw activity files are attached to. Notion's
# upload allowlist rejects .fit/.gpx extensions, so each is stored as a
# Notion-accepted type:
#   FIT  -> decoded JSON  (.json, application/json) — viewable in Notion
#   GPX  -> raw XML       (.xml,  application/xml)  — GPX is valid XML; rename to
#                                                     .gpx to open in GPS apps
DEFAULT_FIT_PROPERTY = "FIT File"
DEFAULT_GPX_PROPERTY = "GPX File"

# Notion single-part uploads top out at 20 MiB (paid) / 5 MiB (free). Anything
# larger is skipped rather than sent on a guaranteed-to-fail request.
DEFAULT_UPLOAD_MAX_MB = 20


class ConfigError(Exception):
    """A required environment variable is missing or invalid."""


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #


def require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise ConfigError(f"Missing env var: {name}")
    return val


def env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on", "y"}


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def token_dir_path() -> str:
    return os.path.expanduser(os.getenv("GARMIN_TOKEN_DIR", DEFAULT_TOKEN_DIR))


def tracked_types() -> set[str]:
    """Garmin ``typeKey``s to sync. Configurable via the ``TRACKED_TYPES`` env."""
    raw = os.getenv("TRACKED_TYPES", "").strip()
    if not raw:
        return set(DEFAULT_TRACKED_TYPES)
    return {t.strip().lower() for t in raw.split(",") if t.strip()}


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging() -> None:
    """Configure root logging from ``LOG_LEVEL`` and ``LOG_FORMAT`` (text|json)."""
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    fmt = os.getenv("LOG_FORMAT", "text").lower()

    # Keep emoji/arrows in messages safe on non-UTF-8 consoles (e.g. Windows cp1252).
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

    handler = logging.StreamHandler(sys.stdout)
    if fmt == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    if level != "DEBUG":
        for noisy in ("garth", "garminconnect", "urllib3", "httpx", "httpcore", "apscheduler"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


# --------------------------------------------------------------------------- #
# Unit conversions / Notion property builders
# --------------------------------------------------------------------------- #


def m_to_mi(v): return None if v is None else round(v * 0.000621371, 2)
def m_to_ft(v): return None if v is None else round(v * 3.28084)
def mps_to_mph(v): return None if v is None else round(v * 2.23694, 2)
def c_to_f(v): return None if v is None else round(v * 9 / 5 + 32, 1)
def sec_to_hr(v): return None if v is None else round(v / 3600, 2)
def sec_to_min(v): return None if v is None else round(v / 60, 1)


def _round(v, ndigits):
    return None if v is None else round(v, ndigits)


def session_type(type_key: str) -> str | None:
    if type_key in HIKE_TYPES:
        return "Hike"
    if type_key in RUN_TYPES:
        return "Run"
    # Configurable extra types fall back to a humanized label (Notion auto-creates
    # the select option on write).
    if type_key:
        return type_key.replace("_", " ").title()
    return None


def _num(v):
    return {"number": v if v is not None else None}


def _title(text: str):
    return {"title": [{"text": {"content": text or ""}}]}


def _text(text: str):
    return {"rich_text": [{"text": {"content": text}}]} if text else {"rich_text": []}


def _select(name: str):
    return {"select": {"name": name}} if name else {"select": None}


def _checkbox(v: bool):
    return {"checkbox": bool(v)}


def _date(iso: str):
    # Garmin's startTimeLocal looks like "2024-03-15 09:30:00" — Notion wants ISO 8601.
    return {"date": {"start": iso.replace(" ", "T")}}


def zone_minutes(zones_payload: Any) -> list[float]:
    """Return [z1, z2, z3, z4, z5] minutes; 0.0 if missing."""
    out = [0.0] * 5
    if not zones_payload:
        return out
    for z in zones_payload:
        idx = z.get("zoneNumber")
        secs = z.get("secsInZone")
        if idx is None or secs is None:
            continue
        if 1 <= idx <= 5:
            out[idx - 1] = round(secs / 60, 1)
    return out


def sleep_hours_for_date(garmin: Garmin, iso_date: str, cache: dict[str, float | None]) -> float | None:
    """Hours slept the night ending on ``iso_date`` (Garmin's sleep-day convention)."""
    if iso_date in cache:
        return cache[iso_date]
    try:
        payload = garmin.get_sleep_data(iso_date)
    except Exception as e:  # noqa: BLE001
        log.warning("sleep unavailable for %s: %s", iso_date, e)
        cache[iso_date] = None
        return None
    secs = ((payload or {}).get("dailySleepDTO") or {}).get("sleepTimeSeconds")
    hours = sec_to_hr(secs)
    cache[iso_date] = hours
    return hours


def build_properties(
    activity: dict,
    zones: list[float],
    detail: dict | None,
    sleep_hours: float | None,
) -> dict:
    type_key = activity.get("activityType", {}).get("typeKey", "")
    name = activity.get("activityName") or type_key.title()
    duration_s = activity.get("movingDuration") or activity.get("duration")
    summary = (detail or {}).get("summaryDTO", {}) if detail else {}

    # Avg temperature only lives in the detail summaryDTO, not the list entry.
    avg_temp_c = summary.get("averageTemperature") if summary else activity.get("averageTemperature")

    te_code = (summary.get("trainingEffectLabel") or activity.get("trainingEffectLabel") or "").upper()
    te_label = TRAINING_EFFECT_LABELS.get(te_code)

    props: dict = {
        "Name": _title(name),
        "Date": _date(activity["startTimeLocal"]),
        "Session Type": _select(session_type(type_key)),
        "Garmin Activity ID": _text(str(activity["activityId"])),
        "Location": _text(activity.get("locationName") or ""),
        "Distance (mi)": _num(m_to_mi(activity.get("distance"))),
        "Duration (hrs)": _num(sec_to_hr(duration_s)),
        "Elevation Gain (ft)": _num(m_to_ft(activity.get("elevationGain"))),
        "Elevation Loss (ft)": _num(m_to_ft(activity.get("elevationLoss"))),
        "Steps": _num(activity.get("steps")),
        "Calories": _num(_round(summary.get("calories") or activity.get("calories"), 0)),
        "Avg HR": _num(_round(activity.get("averageHR"), 0)),
        "Max HR": _num(_round(activity.get("maxHR"), 0)),
        "Avg Moving Speed (mph)": _num(mps_to_mph(activity.get("averageSpeed"))),
        "Avg Cadence (spm)": _num(_round(activity.get("averageRunningCadenceInStepsPerMinute"), 0)),
        "Avg Temp (F)": _num(c_to_f(avg_temp_c)),
        "Aerobic TE": _num(_round(activity.get("aerobicTrainingEffect"), 1)),
        "Anaerobic TE": _num(_round(activity.get("anaerobicTrainingEffect"), 1)),
        "Training Effect": _select(te_label),
        "Training Load": _num(_round(activity.get("activityTrainingLoad") or summary.get("activityTrainingLoad"), 1)),
        "Personal Record": _checkbox(activity.get("pr", False)),
        "HR Zone 1 (min)": _num(zones[0]),
        "HR Zone 2 (min)": _num(zones[1]),
        "HR Zone 3 (min)": _num(zones[2]),
        "HR Zone 4 (min)": _num(zones[3]),
        "HR Zone 5 (min)": _num(zones[4]),
        "Sleep (hrs)": _num(sleep_hours),
    }

    if summary:
        # Garmin stores RPE as 1–10 scaled ×10 (e.g. 60 = RPE 6).
        rpe_raw = summary.get("directWorkoutRpe")
        if rpe_raw:
            props["RPE"] = _num(round(rpe_raw / 10, 1))
        # Feel is bucketed at 0/25/50/75/100; snap to nearest bucket if needed.
        feel_raw = summary.get("directWorkoutFeel")
        if feel_raw is not None:
            bucket = min(FEEL_LABELS, key=lambda k: abs(k - feel_raw))
            props["Feel"] = _select(FEEL_LABELS[bucket])

    if detail:
        notes = detail.get("description") or (detail.get("metadataDTO") or {}).get("description")
        if notes:
            props["Notes"] = _text(notes)

    return props


# --------------------------------------------------------------------------- #
# Notion helpers
# --------------------------------------------------------------------------- #


def resolve_data_source_id(notion: Notion, database_id: str) -> str:
    """Return the single data source ID for a database (new Notion API)."""
    db = notion.databases.retrieve(database_id=database_id)
    sources = db.get("data_sources") or []
    if not sources:
        raise RuntimeError(
            f"Database {database_id} has no data sources — unexpected for a standard DB."
        )
    if len(sources) > 1:
        names = ", ".join(s.get("name", "?") for s in sources)
        raise RuntimeError(
            f"Database {database_id} has multiple data sources ({names}); "
            "pick one and hardcode its ID."
        )
    return sources[0]["id"]


def already_synced(notion: Notion, data_source_id: str, activity_id: str) -> bool:
    resp = notion.data_sources.query(
        data_source_id=data_source_id,
        filter={
            "property": "Garmin Activity ID",
            "rich_text": {"equals": activity_id},
        },
        page_size=1,
    )
    return bool(resp.get("results"))


# --------------------------------------------------------------------------- #
# Retry / rate-limit handling
# --------------------------------------------------------------------------- #

# Transient Garmin errors worth retrying with backoff.
_RETRYABLE_GARMIN = (GarminConnectConnectionError, GarminConnectTooManyRequestsError)


def _retry_after_seconds(err: APIResponseError) -> float | None:
    try:
        val = err.headers.get("Retry-After")
        return float(val) if val else None
    except Exception:  # noqa: BLE001
        return None


def _retry(fn: Callable[[], Any], *, what: str, attempts: int = 4, base: float = 1.5) -> Any:
    """Call ``fn`` with exponential backoff on transient Garmin/Notion errors.

    Notion ``429`` responses honor the ``Retry-After`` header. Non-transient
    errors (validation, auth, not-found) are re-raised immediately.
    """
    last: Exception | None = None
    for i in range(1, attempts + 1):
        try:
            return fn()
        except APIResponseError as e:
            last = e
            transient = e.status == 429 or (e.status is not None and e.status >= 500)
            if not transient or i == attempts:
                raise
            delay = _retry_after_seconds(e) or base * (2 ** (i - 1))
            log.warning("%s: Notion %s — retry %d/%d in %.1fs", what, e.status, i, attempts, delay)
            time.sleep(delay)
        except _RETRYABLE_GARMIN as e:
            last = e
            if i == attempts:
                raise
            delay = base * (2 ** (i - 1))
            log.warning("%s: %s — retry %d/%d in %.1fs", what, type(e).__name__, i, attempts, delay)
            time.sleep(delay)
    raise last  # pragma: no cover  (loop always returns or raises above)


# --------------------------------------------------------------------------- #
# Garmin auth
# --------------------------------------------------------------------------- #


_MFA_BOOTSTRAP_HINT = (
    "Garmin MFA required but no interactive input is available. Run the one-time login "
    "bootstrap: `docker compose run --rm garmin-notion-sync python sync.py --login`"
)


def _mfa_prompt() -> str:
    if sys.stdin is None or not sys.stdin.isatty():
        raise GarminConnectAuthenticationError(_MFA_BOOTSTRAP_HINT)
    try:
        return input("Garmin MFA code: ").strip()
    except EOFError as e:  # stdin is a TTY but closed/empty (e.g. detached container)
        raise GarminConnectAuthenticationError(_MFA_BOOTSTRAP_HINT) from e


def garmin_client() -> Garmin:
    return Garmin(
        require_env("GARMIN_EMAIL"),
        require_env("GARMIN_PASSWORD"),
        prompt_mfa=_mfa_prompt,
    )


def _token_dir_has_tokens(token_dir: str) -> bool:
    return any(
        os.path.exists(os.path.join(token_dir, f))
        for f in ("oauth1_token.json", "oauth2_token.json")
    )


def login_garmin(garmin: Garmin) -> None:
    """Log in using the persisted token store, seeding from base64 on first run.

    If ``GARMIN_TOKENS_BASE64`` is set and the token dir is empty, the tokens are
    loaded from that string and then dumped to the token dir so subsequent runs
    restore from disk without MFA.
    """
    token_dir = token_dir_path()
    os.makedirs(token_dir, exist_ok=True)

    b64 = os.getenv("GARMIN_TOKENS_BASE64", "").strip()
    if b64 and not _token_dir_has_tokens(token_dir):
        log.info("Seeding Garmin tokens from GARMIN_TOKENS_BASE64")
        garmin.login(b64)  # >512 chars → garth loads the token blob directly
        with contextlib.suppress(Exception):
            garmin.client.dump(token_dir)
        return

    garmin.login(token_dir)


def _garth_client(garmin: Garmin | None):
    if garmin is None:
        return None
    return getattr(garmin, "garth", None) or getattr(garmin, "client", None)


def garmin_token_days_remaining(garmin: Garmin | None = None) -> float | None:
    """Best-effort days until the Garmin refresh token expires.

    Prefers an explicit ``refresh_token_expires_at`` (from the in-memory garth
    client or the token JSON on disk); falls back to the token file mtime plus the
    ~1-year refresh-token lifetime. Returns ``None`` if nothing is available.
    """
    now = time.time()

    client = _garth_client(garmin)
    tok = getattr(client, "oauth2_token", None)
    exp = getattr(tok, "refresh_token_expires_at", None)
    if isinstance(exp, (int, float)) and exp > 0:
        return round((exp - now) / 86400, 1)

    token_dir = token_dir_path()
    try:
        json_files = [f for f in os.listdir(token_dir) if f.endswith(".json")]
    except (FileNotFoundError, NotADirectoryError):
        return None

    for fn in json_files:
        try:
            data = json.loads(Path(token_dir, fn).read_text())
        except Exception:  # noqa: BLE001
            continue
        v = data.get("refresh_token_expires_at") if isinstance(data, dict) else None
        if isinstance(v, (int, float)) and v > 0:
            return round((v - now) / 86400, 1)

    mtimes = []
    for fn in json_files:
        with contextlib.suppress(OSError):
            mtimes.append(Path(token_dir, fn).stat().st_mtime)
    if mtimes:
        expiry = max(mtimes) + GARMIN_REFRESH_TOKEN_LIFETIME_DAYS * 86400
        return round((expiry - now) / 86400, 1)
    return None


# --------------------------------------------------------------------------- #
# Activity file attachments (FIT -> JSON, GPX -> XML)
# --------------------------------------------------------------------------- #

_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(text: str, fallback: str = "activity") -> str:
    """Filesystem/URL-safe base name for an uploaded file (max 80 chars)."""
    s = _SLUG_RE.sub("_", (text or "").strip()).strip("._-")
    return s[:80] or fallback


def _fit_bytes_from_download(raw: bytes | None) -> bytes | None:
    """Unwrap the .fit from Garmin's ORIGINAL export (a zip); fall back to treating
    ``raw`` as an already-unwrapped .fit. Returns ``None`` if there's nothing usable."""
    if not raw:
        return None
    if raw[:4] == b"PK\x03\x04":  # zip local-file-header magic
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            names = zf.namelist()
            fits = [n for n in names if n.lower().endswith(".fit")]
            name = fits[0] if fits else (names[0] if names else None)
            return zf.read(name) if name else None
    return raw


def fit_to_json(fit_bytes: bytes) -> bytes:
    """Decode a FIT activity file into indented JSON bytes (all message streams)."""
    from garmin_fit_sdk import Decoder, Stream  # lazy: only needed for FIT attachments

    stream = Stream.from_byte_array(fit_bytes)
    messages, _errors = Decoder(stream).read()
    return json.dumps(messages, indent=2, default=str).encode("utf-8")


def _files_property(upload_id: str, filename: str) -> dict:
    """A Notion "files" property value referencing an uploaded file."""
    return {"files": [{"type": "file_upload", "file_upload": {"id": upload_id}, "name": filename}]}


def notion_upload(notion: Notion, filename: str, data: bytes, content_type: str) -> str:
    """Upload ``data`` to Notion via the single-part File Upload API; return the id."""
    up = _retry(
        lambda: notion.file_uploads.create(
            mode="single_part", filename=filename, content_type=content_type
        ),
        what="file_upload create",
    )
    upload_id = up["id"]
    _retry(
        lambda: notion.file_uploads.send(
            file_upload_id=upload_id, file=(filename, data, content_type)
        ),
        what="file_upload send",
    )
    return upload_id


def _prepare_fit(garmin: Garmin, activity_id: str, base: str) -> tuple[str, bytes, str] | None:
    raw = garmin.download_activity(activity_id, Garmin.ActivityDownloadFormat.ORIGINAL)
    fit = _fit_bytes_from_download(raw)
    if not fit:
        log.warning("No FIT data for %s — skipping FIT attachment.", activity_id)
        return None
    return f"{base}.json", fit_to_json(fit), "application/json"


def _prepare_gpx(garmin: Garmin, activity_id: str, base: str) -> tuple[str, bytes, str] | None:
    raw = garmin.download_activity(activity_id, Garmin.ActivityDownloadFormat.GPX)
    if not raw:
        log.warning("No GPX data for %s — skipping GPX attachment.", activity_id)
        return None
    return f"{base}.xml", raw, "application/xml"


# kind -> (download+convert builder). Order also determines column-fill order.
_ATTACHMENT_BUILDERS: dict[str, Callable[[Garmin, str, str], tuple[str, bytes, str] | None]] = {
    "fit": _prepare_fit,
    "gpx": _prepare_gpx,
}


def attachment_targets(notion: Notion, data_source_id: str) -> dict[str, str]:
    """Map ``{kind: property_name}`` for the FIT/GPX columns that actually exist as
    "files" properties. Empty when ``SYNC_ATTACHMENTS`` is off or the columns are
    absent — so deployments without those columns are unaffected."""
    if not env_bool("SYNC_ATTACHMENTS", True):
        log.info("Activity file attachments disabled (SYNC_ATTACHMENTS=false).")
        return {}

    wanted = {
        "fit": os.getenv("NOTION_FIT_PROPERTY", DEFAULT_FIT_PROPERTY),
        "gpx": os.getenv("NOTION_GPX_PROPERTY", DEFAULT_GPX_PROPERTY),
    }
    try:
        ds = _retry(
            lambda: notion.data_sources.retrieve(data_source_id=data_source_id),
            what="data source schema",
        )
    except Exception as e:  # noqa: BLE001
        log.warning("Could not read Notion schema; skipping attachments: %s", e)
        return {}

    schema = ds.get("properties") or {}
    targets: dict[str, str] = {}
    for kind, name in wanted.items():
        prop = schema.get(name)
        if prop is None:
            log.info("No %r column in Notion DB — skipping %s attachments.", name, kind.upper())
        elif prop.get("type") != "files":
            log.warning(
                "Column %r is type %r, not 'files' — skipping %s attachments.",
                name, prop.get("type"), kind.upper(),
            )
        else:
            targets[kind] = name
    if targets:
        log.info(
            "Attaching activity files to Notion for new activities: %s.",
            ", ".join(f"{k.upper()}→{v!r}" for k, v in targets.items()),
        )
    return targets


def build_attachment_props(
    garmin: Garmin, notion: Notion, activity: dict, targets: dict[str, str]
) -> tuple[dict, int]:
    """Download, convert and upload the FIT/GPX files for one activity, returning the
    Notion "files" properties to merge into the page plus a count attached. Each
    attachment is isolated: a failure is logged and skipped, never failing the activity."""
    activity_id = str(activity["activityId"])
    base = _slug(f"{activity.get('activityName') or ''}-{activity_id}", fallback=activity_id)
    max_bytes = env_int("NOTION_UPLOAD_MAX_MB", DEFAULT_UPLOAD_MAX_MB) * 1024 * 1024

    props: dict = {}
    attached = 0
    for kind, prop_name in targets.items():
        builder = _ATTACHMENT_BUILDERS[kind]
        try:
            prepared = _retry(lambda: builder(garmin, activity_id, base), what=f"{kind} download")
            if prepared is None:
                continue
            filename, data, content_type = prepared
            if len(data) > max_bytes:
                log.warning(
                    "%s for %s is %.1f MB (over the %d MB upload limit) — skipping.",
                    kind.upper(), activity_id, len(data) / 1_000_000, max_bytes // (1024 * 1024),
                )
                continue
            upload_id = notion_upload(notion, filename, data, content_type)
            props[prop_name] = _files_property(upload_id, filename)
            attached += 1
            log.info("attach %s %s (%.0f KB)", kind.upper(), filename, len(data) / 1024)
        except Exception as e:  # noqa: BLE001
            log.warning("%s attachment failed for %s: %s", kind.upper(), activity_id, e)
    return props, attached


# --------------------------------------------------------------------------- #
# Sync engine
# --------------------------------------------------------------------------- #


@dataclass
class SyncResult:
    window_start: str
    window_end: str
    created: int = 0
    skipped: int = 0
    failed: int = 0
    attached: int = 0
    created_labels: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    auth_error: str | None = None
    duration_s: float = 0.0
    token_days_remaining: float | None = None

    @property
    def ok(self) -> bool:
        return self.auth_error is None and self.failed == 0

    def summary(self) -> str:
        return (
            f"created={self.created} skipped={self.skipped} failed={self.failed} "
            f"attached={self.attached} "
            f"window={self.window_start}→{self.window_end} duration={self.duration_s}s"
        )

    def to_dict(self) -> dict:
        return {
            "window_start": self.window_start,
            "window_end": self.window_end,
            "created": self.created,
            "skipped": self.skipped,
            "failed": self.failed,
            "attached": self.attached,
            "created_labels": self.created_labels,
            "failures": self.failures,
            "auth_error": self.auth_error,
            "duration_s": self.duration_s,
            "token_days_remaining": self.token_days_remaining,
            "ok": self.ok,
        }


def sync_once(days: int, dry_run: bool = False) -> SyncResult:
    """Run one sync of the last ``days`` days. Never raises — failures are captured
    on the returned :class:`SyncResult` so a scheduler can keep running."""
    start_t = time.monotonic()
    end = date.today()
    start = end - timedelta(days=days)
    result = SyncResult(window_start=start.isoformat(), window_end=end.isoformat())
    log.info("Syncing Garmin activities %s → %s (dry_run=%s)", result.window_start, result.window_end, dry_run)

    # --- credentials + Garmin login (auth failures are non-fatal for the loop) ---
    try:
        notion_token = require_env("NOTION_TOKEN")
        database_id = require_env("NOTION_DATABASE_ID")
        garmin = garmin_client()
        login_garmin(garmin)
    except (ConfigError, GarminConnectAuthenticationError) as e:
        result.auth_error = str(e)
        log.error("Authentication/config error: %s", e)
        result.duration_s = round(time.monotonic() - start_t, 1)
        return result
    except Exception as e:  # noqa: BLE001
        result.auth_error = f"Garmin login failed: {e}"
        log.exception("Garmin login failed")
        result.duration_s = round(time.monotonic() - start_t, 1)
        return result

    result.token_days_remaining = garmin_token_days_remaining(garmin)

    # --- fetch + filter activities ---
    try:
        activities = _retry(
            lambda: garmin.get_activities_by_date(result.window_start, result.window_end),
            what="get_activities_by_date",
        )
    except Exception as e:  # noqa: BLE001
        log.exception("Failed to list activities")
        result.failed += 1
        result.failures.append(f"list activities: {e}")
        result.duration_s = round(time.monotonic() - start_t, 1)
        return result

    wanted = tracked_types()
    tracked = [a for a in activities if a.get("activityType", {}).get("typeKey") in wanted]
    log.info(
        "Found %d activities, %d tracked (%s).",
        len(activities), len(tracked), ", ".join(sorted(wanted)),
    )

    # --- Notion setup ---
    notion = Notion(auth=notion_token)
    try:
        data_source_id = _retry(
            lambda: resolve_data_source_id(notion, database_id), what="resolve_data_source"
        )
    except Exception as e:  # noqa: BLE001
        log.exception("Failed to resolve Notion data source")
        result.failed += 1
        result.failures.append(f"resolve data source: {e}")
        result.duration_s = round(time.monotonic() - start_t, 1)
        return result

    targets = attachment_targets(notion, data_source_id)

    pacing = max(0.0, float(os.getenv("NOTION_PACING_MS", "350")) / 1000.0)
    sleep_cache: dict[str, float | None] = {}

    for a in tracked:
        activity_id = str(a["activityId"])
        label = f"{a.get('activityName', '?')} [{activity_id}] ({a['activityType']['typeKey']})"
        try:
            if _retry(lambda: already_synced(notion, data_source_id, activity_id), what="dedupe"):
                log.info("skip  %s", label)
                result.skipped += 1
                continue

            try:
                zones_payload = _retry(
                    lambda: garmin.get_activity_hr_in_timezones(activity_id), what="hr zones"
                )
            except Exception as e:  # noqa: BLE001
                log.warning("hr zones unavailable for %s: %s", activity_id, e)
                zones_payload = None
            zones = zone_minutes(zones_payload)

            try:
                detail = _retry(lambda: garmin.get_activity(activity_id), what="detail")
            except Exception as e:  # noqa: BLE001
                log.warning("detail unavailable for %s: %s", activity_id, e)
                detail = None

            activity_date = a["startTimeLocal"][:10]
            sleep_hrs = sleep_hours_for_date(garmin, activity_date, sleep_cache)

            props = build_properties(a, zones, detail, sleep_hrs)

            if dry_run:
                log.info("DRY   %s", label)
                result.created += 1
                result.created_labels.append(label)
                continue

            if targets:
                att_props, n_attached = build_attachment_props(garmin, notion, a, targets)
                props.update(att_props)
                result.attached += n_attached

            _retry(
                lambda: notion.pages.create(
                    parent={"type": "data_source_id", "data_source_id": data_source_id},
                    properties=props,
                ),
                what="create page",
            )
            log.info("new   %s", label)
            result.created += 1
            result.created_labels.append(label)
            if pacing:
                time.sleep(pacing)
        except Exception as e:  # noqa: BLE001
            log.error("FAIL  %s: %s", label, e)
            result.failed += 1
            result.failures.append(f"{label}: {e}")

    result.duration_s = round(time.monotonic() - start_t, 1)
    log.info("Done. %s", result.summary())
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _do_login() -> int:
    """Interactive Garmin login that seeds/refreshes the token store, then exits."""
    garmin = garmin_client()
    token_dir = token_dir_path()
    os.makedirs(token_dir, exist_ok=True)
    log.info("Logging in to Garmin (you'll be prompted for an MFA code if required)…")
    garmin.login(token_dir)
    with contextlib.suppress(Exception):
        garmin.client.dump(token_dir)
    days = garmin_token_days_remaining(garmin)
    log.info("Garmin login OK. Tokens stored in %s (refresh token ~%s days remaining).", token_dir, days)
    return 0


def _debug_dump(days: int, target: str) -> int:
    garmin = garmin_client()
    login_garmin(garmin)
    end = date.today()
    start = end - timedelta(days=days)
    activities = garmin.get_activities_by_date(start.isoformat(), end.isoformat())
    wanted = tracked_types()
    tracked = [a for a in activities if a.get("activityType", {}).get("typeKey") in wanted]
    targets = tracked if target == "all" else [a for a in tracked if str(a["activityId"]) == target]
    if not targets:
        log.error("No tracked activity matches --debug %s.", target)
        return 1
    for a in targets:
        aid = str(a["activityId"])
        print(f"\n{'=' * 80}\n=== ACTIVITY {aid}: {a.get('activityName')} ({a['activityType']['typeKey']})\n{'=' * 80}")
        print("\n--- list entry (get_activities_by_date) ---")
        print(json.dumps(a, indent=2, default=str))
        try:
            print("\n--- get_activity(aid) ---")
            print(json.dumps(garmin.get_activity(aid), indent=2, default=str))
        except Exception as e:  # noqa: BLE001
            print(f"(get_activity failed: {e})")
        try:
            print("\n--- get_activity_details(aid) ---")
            print(json.dumps(garmin.get_activity_details(aid), indent=2, default=str))
        except Exception as e:  # noqa: BLE001
            print(f"(get_activity_details failed: {e})")
        try:
            print("\n--- get_activity_hr_in_timezones(aid) ---")
            print(json.dumps(garmin.get_activity_hr_in_timezones(aid), indent=2, default=str))
        except Exception as e:  # noqa: BLE001
            print(f"(hr zones failed: {e})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=7, help="Sync the last N days (default 7).")
    parser.add_argument("--dry-run", action="store_true", help="Log what would be created, write nothing.")
    parser.add_argument(
        "--login",
        action="store_true",
        help="Interactive Garmin login (handles MFA) to seed/refresh the token store, then exit.",
    )
    parser.add_argument(
        "--debug",
        nargs="?",
        const="all",
        metavar="ACTIVITY_ID",
        help="Dump raw Garmin JSON (summary + details + HR zones) for each matched activity "
        "and exit without writing to Notion. Pass an activity ID to dump just one.",
    )
    args = parser.parse_args()

    load_dotenv()
    setup_logging()

    try:
        if args.login:
            return _do_login()
        if args.debug:
            return _debug_dump(args.days, args.debug)
        result = sync_once(args.days, dry_run=args.dry_run)
    except ConfigError as e:
        log.error("%s", e)
        return 2

    if result.auth_error:
        return 1
    return 0 if result.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
