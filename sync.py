"""Sync Garmin Connect hikes and runs to the Notion Exercise Log database.

Can run as a one-shot CLI (``python sync.py``) or be driven on a schedule by
``runner.py``. The reusable engine is :func:`sync_once`, which returns a
:class:`SyncResult` instead of printing, so callers can log, notify, and export
metrics from a single run.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
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

# Notion single-part uploads top out at 5 MiB on free workspaces (20 MiB on paid).
# Attachments over this are gzipped to fit; only skipped if still too big after that.
# Default to the free-tier limit; paid users can raise it to keep files uncompressed.
DEFAULT_UPLOAD_MAX_MB = 5

# --- Physiology metrics --------------------------------------------------------
# The full JSON metrics record is attached to this Files column; a handful of
# headline scalars are promoted to their own columns for filtering/rollups.
DEFAULT_METRICS_PROPERTY = "Metrics JSON"
DEFAULT_METRICS_MIN_TRACKPOINTS = 30

# Headline scalar metrics promoted to Notion columns (name -> property type). These
# are the physiology-specific numbers not already populated by build_properties.
METRIC_PROPERTY_SPECS: dict[str, str] = {
    "Aerobic Decoupling (%)": "number",
    "HR Response Lag (s)": "number",
    "HRR @60s (bpm)": "number",
    "HRR Full (bpm)": "number",
    "Moving Time (hrs)": "number",
    "Stopped Time (min)": "number",
    "Ascent VAM (m/h)": "number",
    "Descent Speed (kmh)": "number",
    "Cadence Flat": "number",
    "Cadence Steep": "number",
    "Cadence Coverage (%)": "number",
    "Ascent Source": "rich_text",
}


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


def decode_fit(fit_bytes: bytes) -> dict:
    """Decode a FIT activity file into its message-stream dict."""
    from garmin_fit_sdk import Decoder, Stream  # lazy: only needed for FIT handling

    messages, _errors = Decoder(Stream.from_byte_array(fit_bytes)).read()
    return messages


def fit_messages_to_json(messages: dict) -> bytes:
    # Compact separators — decoded FIT is large (multi-MB); indentation only inflates it.
    return json.dumps(messages, separators=(",", ":"), default=str).encode("utf-8")


def fit_to_json(fit_bytes: bytes) -> bytes:
    """Decode a FIT activity file into indented JSON bytes (all message streams)."""
    return fit_messages_to_json(decode_fit(fit_bytes))


def fit_summary(messages: dict | None) -> dict:
    """Device-summary fields GPS can't reproduce, from the FIT ``session`` message.
    Barometric ascent/descent (in feet) override the GPS-derived values downstream."""
    sessions = (messages or {}).get("session_mesgs") or []
    if not sessions:
        return {}
    s = sessions[0]
    out: dict = {}
    if s.get("total_ascent") is not None:
        out["ascent_ft"] = round(s["total_ascent"] * 3.28084)
    if s.get("total_descent") is not None:
        out["descent_ft"] = round(s["total_descent"] * 3.28084)
    for src, dst in (
        ("total_calories", "calories"),
        ("total_training_effect", "aerobic_te"),
        ("total_anaerobic_training_effect", "anaerobic_te"),
        ("sweat_loss", "sweat_loss_ml"),
    ):
        if s.get(src) is not None:
            with contextlib.suppress(TypeError, ValueError):
                out[dst] = float(s[src])
    return out


# GPX/Garmin XML namespaces vary by prefix; match on the element's local name instead.
def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_gpx_time(text: str) -> Any:
    from datetime import datetime as _dt

    return _dt.fromisoformat(text.strip().replace("Z", "+00:00"))


def parse_gpx_track(gpx_bytes: bytes) -> list[dict]:
    """Parse Garmin GPX bytes into the metrics track contract: one dict per trackpoint
    with lat/lon/ele/t and optional hr/temp/cad (from ``TrackPointExtension``)."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(gpx_bytes)
    track: list[dict] = []
    for el in root.iter():
        if _localname(el.tag) != "trkpt":
            continue
        try:
            lat = float(el.attrib["lat"])
            lon = float(el.attrib["lon"])
        except (KeyError, ValueError):
            continue
        pt: dict = {"lat": lat, "lon": lon, "ele": 0.0, "t": None,
                    "hr": None, "temp": None, "cad": None}
        for child in el.iter():
            name = _localname(child.tag)
            txt = (child.text or "").strip()
            if not txt and name not in ("trkpt",):
                continue
            if name == "ele":
                with contextlib.suppress(ValueError):
                    pt["ele"] = float(txt)
            elif name == "time":
                with contextlib.suppress(ValueError):
                    pt["t"] = _parse_gpx_time(txt)
            elif name == "hr":
                with contextlib.suppress(ValueError):
                    pt["hr"] = int(float(txt))
            elif name == "atemp":
                with contextlib.suppress(ValueError):
                    pt["temp"] = float(txt)
            elif name == "cad":
                with contextlib.suppress(ValueError):
                    pt["cad"] = int(float(txt))
        if pt["t"] is not None:
            track.append(pt)
    return track


@dataclass
class ActivityMedia:
    """One activity's raw files, downloaded once and shared by attachments + metrics."""
    fit: bytes | None = None          # unwrapped .fit bytes
    fit_messages: dict | None = None  # decoded FIT (lazily, on first access)
    gpx: bytes | None = None          # raw GPX bytes


def download_activity_media(
    garmin: Garmin, activity_id: str, *, want_fit: bool, want_gpx: bool
) -> ActivityMedia:
    """Download an activity's FIT (original) and/or GPX exactly once. Best-effort: a
    failed download is logged and left as ``None`` rather than raising."""
    media = ActivityMedia()
    if want_fit:
        try:
            raw = _retry(
                lambda: garmin.download_activity(activity_id, Garmin.ActivityDownloadFormat.ORIGINAL),
                what="fit download",
            )
            media.fit = _fit_bytes_from_download(raw)
        except Exception as e:  # noqa: BLE001
            log.warning("FIT download failed for %s: %s", activity_id, e)
    if want_gpx:
        try:
            media.gpx = _retry(
                lambda: garmin.download_activity(activity_id, Garmin.ActivityDownloadFormat.GPX),
                what="gpx download",
            ) or None
        except Exception as e:  # noqa: BLE001
            log.warning("GPX download failed for %s: %s", activity_id, e)
    return media


def _upload_files_prop(
    notion: Notion, data: bytes | None, filename: str, max_bytes: int, what: str
) -> dict | None:
    """Gzip and upload ``data``, returning a Notion files-property value (or None).

    All attachments are stored gzipped (``.gz``, an accepted Notion type) for
    consistency and to stay well under Notion's single-part upload limit — the
    verbose FIT/GPX/JSON payloads compress by roughly 5–10×. Only skipped if it's
    still over the limit after compression."""
    if not data:
        return None
    gz = gzip.compress(data)
    filename = filename + ".gz"
    if len(gz) > max_bytes:
        log.warning(
            "%s is %.1f MB gzipped (over the %d MB upload limit) — skipping.",
            what, len(gz) / 1_000_000, max_bytes // (1024 * 1024),
        )
        return None
    upload_id = notion_upload(notion, filename, gz, "application/gzip")
    log.info("attach %s (%.0f KB, from %.0f KB)", filename, len(gz) / 1024, len(data) / 1024)
    return _files_property(upload_id, filename)


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


@dataclass
class MetricsTarget:
    """Where physiology metrics are written: the JSON files column, the headline scalar
    columns that exist, and the compute config."""
    json_prop: str
    headline_cols: set
    cfg: Any


def metrics_target(notion: Notion, data_source_id: str) -> "MetricsTarget | None":
    """Resolve where to write physiology metrics, creating the "Metrics JSON" files
    column and the headline scalar columns when missing (``NOTION_CREATE_COLUMNS``).
    Returns ``None`` when ``SYNC_METRICS`` is off or the metrics column is unavailable."""
    if not env_bool("SYNC_METRICS", True):
        log.info("Physiology metrics disabled (SYNC_METRICS=false).")
        return None

    json_prop = os.getenv("NOTION_METRICS_PROPERTY", DEFAULT_METRICS_PROPERTY)
    try:
        ds = _retry(
            lambda: notion.data_sources.retrieve(data_source_id=data_source_id),
            what="data source schema",
        )
    except Exception as e:  # noqa: BLE001
        log.warning("Could not read Notion schema; skipping metrics: %s", e)
        return None
    schema = ds.get("properties") or {}

    if json_prop in schema and schema[json_prop].get("type") != "files":
        log.warning(
            "Column %r is type %r, not 'files' — metrics disabled.",
            json_prop, schema[json_prop].get("type"),
        )
        return None

    to_create: dict = {}
    if json_prop not in schema:
        to_create[json_prop] = {"files": {}}
    for name, ptype in METRIC_PROPERTY_SPECS.items():
        if name not in schema:
            to_create[name] = {"number": {"format": "number"}} if ptype == "number" else {ptype: {}}

    created: set = set()
    if to_create and env_bool("NOTION_CREATE_COLUMNS", True):
        try:
            _retry(
                lambda: notion.data_sources.update(
                    data_source_id=data_source_id, properties=to_create
                ),
                what="create metric columns",
            )
            created = set(to_create)
            log.info("Created %d Notion metric column(s): %s", len(created), ", ".join(sorted(created)))
        except Exception as e:  # noqa: BLE001
            log.warning("Failed to create metric columns: %s", e)
    elif to_create:
        log.warning(
            "Metrics columns missing and NOTION_CREATE_COLUMNS=false — only existing "
            "columns will be written: %s", ", ".join(sorted(to_create)),
        )

    present = set(schema) | created
    if json_prop not in present:
        log.warning("Metrics column %r unavailable — metrics disabled.", json_prop)
        return None
    headline_cols = {n for n in METRIC_PROPERTY_SPECS if n in present}
    log.info("Computing physiology metrics into %r.", json_prop)
    return MetricsTarget(json_prop=json_prop, headline_cols=headline_cols, cfg=metrics_config())


def _page_files_present(page: dict, name: str) -> bool:
    return bool(((page.get("properties") or {}).get(name) or {}).get("files"))


def metrics_config():
    """Build an ``activity_metrics.Config`` from the environment (only MAX_HR today)."""
    import activity_metrics as am

    return am.Config(max_hr=env_int("MAX_HR", am.Config().max_hr))


def _metrics_json(record: dict) -> bytes:
    return json.dumps(record, separators=(",", ":"), default=str).encode("utf-8")


def metrics_headline_props(record: dict) -> dict:
    """The handful of scalar physiology metrics promoted to Notion database columns
    (see ``METRIC_PROPERTY_SPECS``). Everything else lives in the JSON record."""
    s = record.get("summary") or {}
    phases = record.get("phases") or {}
    asc, desc = phases.get("ascent") or {}, phases.get("descent") or {}
    cad = record.get("cadence") or {}
    dec = record.get("decoupling") or {}
    lag = record.get("hr_lag") or {}
    hardest = (record.get("hr_recovery") or {}).get("hardest") or {}

    def _min(v, div):
        return None if v is None else round(v / div, 2)

    props = {
        "Aerobic Decoupling (%)": _num(dec.get("decoupling_pct")),
        "HR Response Lag (s)": _num(lag.get("lag_s")),
        "HRR @60s (bpm)": _num(hardest.get("hrr60_bpm")),
        "HRR Full (bpm)": _num(hardest.get("hrr_full_bpm")),
        "Moving Time (hrs)": _num(_min(s.get("moving_s"), 3600)),
        "Stopped Time (min)": _num(_min(s.get("stopped_s"), 60)),
        "Ascent VAM (m/h)": _num(asc.get("vam_m_per_h")),
        "Descent Speed (kmh)": _num(desc.get("avg_moving_speed_kmh")),
        "Cadence Flat": _num(cad.get("flat")),
        "Cadence Steep": _num(cad.get("steep")),
        "Cadence Coverage (%)": _num(round(cad["coverage"] * 100, 1) if cad.get("coverage") is not None else None),
        "Ascent Source": _text(s.get("ascent_source") or ""),
    }
    return props


def build_media_props(
    garmin: Garmin,
    notion: Notion,
    activity: dict,
    *,
    fit_prop: str | None,
    gpx_prop: str | None,
    metrics: "MetricsTarget | None",
    dry_run: bool = False,
) -> tuple[dict, dict]:
    """Download an activity's files once and produce every Notion property they feed:
    the FIT-JSON / GPX-XML attachments, and (when ``metrics`` is set) the computed
    metrics record as a JSON attachment plus its headline scalar columns.

    Returns ``(props, stats)`` where ``stats = {"attached": int, "metrics": bool}``. Each
    piece is isolated — a download/parse/upload failure is logged and skipped, never
    failing the activity."""
    activity_id = str(activity["activityId"])
    base = _slug(f"{activity.get('activityName') or ''}-{activity_id}", fallback=activity_id)
    max_bytes = env_int("NOTION_UPLOAD_MAX_MB", DEFAULT_UPLOAD_MAX_MB) * 1024 * 1024
    min_points = env_int("METRICS_MIN_TRACKPOINTS", DEFAULT_METRICS_MIN_TRACKPOINTS)

    want_metrics = metrics is not None
    media = download_activity_media(
        garmin, activity_id,
        want_fit=fit_prop is not None or want_metrics,
        want_gpx=gpx_prop is not None or want_metrics,
    )

    # Decode the FIT once; reused by both the JSON attachment and the metrics summary.
    if media.fit and (fit_prop is not None or want_metrics):
        try:
            media.fit_messages = decode_fit(media.fit)
        except Exception as e:  # noqa: BLE001
            log.warning("FIT decode failed for %s: %s", activity_id, e)

    props: dict = {}
    attached = 0

    def _attach(prop_name, data, filename, what):
        nonlocal attached
        try:
            if dry_run:
                if data:
                    log.info("DRY   would attach %s.gz", filename)
                    attached += 1
                return
            fp = _upload_files_prop(notion, data, filename, max_bytes, what)
            if fp is not None:
                props[prop_name] = fp
                attached += 1
        except Exception as e:  # noqa: BLE001
            log.warning("%s attachment failed for %s: %s", what, activity_id, e)

    if fit_prop is not None:
        fit_json = fit_messages_to_json(media.fit_messages) if media.fit_messages else None
        _attach(fit_prop, fit_json, f"{base}.json", "FIT")
    if gpx_prop is not None:
        _attach(gpx_prop, media.gpx, f"{base}.xml", "GPX")

    metrics_done = False
    if want_metrics and media.gpx:
        try:
            import activity_metrics as am

            track = parse_gpx_track(media.gpx)
            if len(track) < min_points:
                log.info("metrics skipped for %s: only %d trackpoints", activity_id, len(track))
            else:
                record = am.compute_all(
                    track,
                    fit=fit_summary(media.fit_messages),
                    cfg=metrics.cfg,
                    activity_meta={"id": activity_id, "name": activity.get("activityName")},
                )
                # Only write headline columns that exist (auto-create may be off/failed).
                headline = {
                    k: v for k, v in metrics_headline_props(record).items()
                    if k in metrics.headline_cols
                }
                if dry_run:
                    log.info("DRY   would compute metrics for %s (%d pts)", activity_id, len(track))
                    metrics_done = True
                else:
                    _attach(metrics.json_prop, _metrics_json(record),
                            f"{base}.metrics.json", "metrics")
                    props.update(headline)
                    metrics_done = True
                    log.info("metrics %s: decouple=%s lag=%ss", activity_id,
                             (record["decoupling"] or {}).get("decoupling_pct"),
                             (record["hr_lag"] or {}).get("lag_s"))
        except Exception as e:  # noqa: BLE001
            log.warning("metrics failed for %s: %s", activity_id, e)

    return props, {"attached": attached, "metrics": metrics_done}


def _rich_text_value(prop: dict | None) -> str:
    """Flatten a Notion title/rich_text property value to plain text."""
    if not prop:
        return ""
    segs = prop.get("rich_text") or prop.get("title") or []
    return "".join(s.get("plain_text") or (s.get("text") or {}).get("content") or "" for s in segs)


def _page_activity_id(page: dict) -> str | None:
    """The ``Garmin Activity ID`` stored on a Notion page, if any."""
    val = _rich_text_value((page.get("properties") or {}).get("Garmin Activity ID")).strip()
    return val or None


def _page_title(page: dict) -> str:
    """The page's title-property text (used to name the uploaded files)."""
    for prop in (page.get("properties") or {}).values():
        if prop.get("type") == "title" or "title" in prop:
            return _rich_text_value(prop)
    return ""


def _page_missing_attachment_kinds(page: dict, targets: dict[str, str], force: bool) -> dict[str, str]:
    """Subset of ``targets`` whose column is empty on this page (or all, if ``force``)."""
    props = page.get("properties") or {}
    missing: dict[str, str] = {}
    for kind, name in targets.items():
        if force or not ((props.get(name) or {}).get("files") or []):
            missing[kind] = name
    return missing


def iter_data_source_pages(notion: Notion, data_source_id: str, page_size: int = 100):
    """Yield every page in a Notion data source, following pagination."""
    cursor: str | None = None
    while True:
        kwargs = {"data_source_id": data_source_id, "page_size": page_size}
        if cursor:
            kwargs["start_cursor"] = cursor
        resp = _retry(lambda: notion.data_sources.query(**kwargs), what="query pages")
        yield from resp.get("results", [])
        if not resp.get("has_more"):
            return
        cursor = resp.get("next_cursor")


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
    metrics: int = 0
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
            f"attached={self.attached} metrics={self.metrics} "
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
            "metrics": self.metrics,
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
    metrics = metrics_target(notion, data_source_id)

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

            if targets or metrics:
                media_props, stats = build_media_props(
                    garmin, notion, a,
                    fit_prop=targets.get("fit"),
                    gpx_prop=targets.get("gpx"),
                    metrics=metrics,
                )
                props.update(media_props)
                result.attached += stats["attached"]
                result.metrics += 1 if stats["metrics"] else 0

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
# Backfill: attach files to activities already in Notion
# --------------------------------------------------------------------------- #


def _backfill_pages(
    garmin: Garmin,
    notion: Notion,
    data_source_id: str,
    targets: dict[str, str],
    metrics: "MetricsTarget | None",
    result: SyncResult,
    *,
    force: bool,
    dry_run: bool,
    pacing: float,
) -> None:
    """Walk existing Notion pages and fill in any missing FIT/GPX attachments and/or
    metrics. Counts land on ``result``: ``created`` = pages filled, ``skipped`` =
    already complete / no data, ``attached`` = files added, ``metrics`` = pages with
    metrics computed, ``failed`` = update errors."""
    try:
        pages = iter_data_source_pages(notion, data_source_id)
        for page in pages:
            _backfill_page(garmin, notion, page, targets, metrics, result,
                           force=force, dry_run=dry_run, pacing=pacing)
    except Exception as e:  # noqa: BLE001 — a page-query failure shouldn't crash the run
        log.exception("Backfill stopped early")
        result.failed += 1
        result.failures.append(f"backfill loop: {e}")


def _backfill_page(
    garmin: Garmin,
    notion: Notion,
    page: dict,
    targets: dict[str, str],
    metrics: "MetricsTarget | None",
    result: SyncResult,
    *,
    force: bool,
    dry_run: bool,
    pacing: float,
) -> None:
    activity_id = _page_activity_id(page)
    title = _page_title(page) or (activity_id or "?")
    label = f"{title} [{activity_id}]"

    if not activity_id:
        log.debug("skip  %s: no Garmin Activity ID", label)
        result.skipped += 1
        return

    missing = _page_missing_attachment_kinds(page, targets, force)
    want_metrics = metrics is not None and (force or not _page_files_present(page, metrics.json_prop))
    if not missing and not want_metrics:
        result.skipped += 1
        return

    wanted = sorted([*missing, *(["metrics"] if want_metrics else [])])
    try:
        if dry_run:
            log.info("DRY   backfill %s: would fill %s", label, ", ".join(wanted))
            result.created += 1
            result.attached += len(missing)
            result.metrics += 1 if want_metrics else 0
            result.created_labels.append(label)
            return

        activity = {"activityId": activity_id, "activityName": title}
        page_props, stats = build_media_props(
            garmin, notion, activity,
            fit_prop=missing.get("fit"),
            gpx_prop=missing.get("gpx"),
            metrics=metrics if want_metrics else None,
        )
        if not page_props:
            log.warning("skip  %s: nothing could be produced from Garmin", label)
            result.skipped += 1
            return

        _retry(
            lambda: notion.pages.update(page_id=page["id"], properties=page_props),
            what="update page",
        )
        log.info("fill  %s (files+%d%s)", label, stats["attached"],
                 ", metrics" if stats["metrics"] else "")
        result.created += 1
        result.attached += stats["attached"]
        result.metrics += 1 if stats["metrics"] else 0
        result.created_labels.append(label)
        if pacing:
            time.sleep(pacing)
    except Exception as e:  # noqa: BLE001
        log.error("FAIL  %s: %s", label, e)
        result.failed += 1
        result.failures.append(f"{label}: {e}")


def backfill_once(force: bool = False, dry_run: bool = False) -> SyncResult:
    """One-time pass that attaches FIT/GPX files and computes metrics for activities
    already in Notion that are missing them. Idempotent — pages that already have the
    file/metrics are skipped, so a run interrupted by rate limits can simply be re-run.
    Never raises."""
    start_t = time.monotonic()
    result = SyncResult(window_start="backfill", window_end="backfill")
    log.info("Backfilling activity files + metrics (force=%s, dry_run=%s)", force, dry_run)

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
    metrics = metrics_target(notion, data_source_id)
    if not targets and metrics is None:
        log.warning("No FIT/GPX file columns and metrics disabled — nothing to backfill.")
        result.duration_s = round(time.monotonic() - start_t, 1)
        return result

    pacing = max(0.0, float(os.getenv("NOTION_PACING_MS", "350")) / 1000.0)
    _backfill_pages(
        garmin, notion, data_source_id, targets, metrics, result,
        force=force, dry_run=dry_run, pacing=pacing,
    )

    result.duration_s = round(time.monotonic() - start_t, 1)
    log.info(
        "Backfill done. filled=%d skipped=%d failed=%d attached=%d metrics=%d duration=%ss",
        result.created, result.skipped, result.failed, result.attached, result.metrics,
        result.duration_s,
    )
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
        "--backfill",
        action="store_true",
        help="One-time: attach FIT/GPX files to activities already in Notion that are missing "
        "them, then exit. Idempotent and resumable; combine with --dry-run to preview.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="With --backfill, re-download and overwrite attachments even if already present.",
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
        if args.backfill:
            result = backfill_once(force=args.force, dry_run=args.dry_run)
        else:
            result = sync_once(args.days, dry_run=args.dry_run)
    except ConfigError as e:
        log.error("%s", e)
        return 2

    if result.auth_error:
        return 1
    return 0 if result.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
