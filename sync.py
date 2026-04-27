"""Sync Garmin Connect hikes and runs to the Notion Exercise Log database."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, timedelta
from typing import Any

from dotenv import load_dotenv
from garminconnect import Garmin
from notion_client import Client as Notion

HIKE_TYPES = {"hiking"}
RUN_TYPES = {"running", "trail_running"}
TRACKED_TYPES = HIKE_TYPES | RUN_TYPES

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

TOKEN_DIR = os.path.expanduser("~/.garminconnect")


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
        print(f"    (sleep unavailable for {iso_date}: {e})")
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


def require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        sys.exit(f"Missing env var: {name}")
    return val


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=7, help="Sync the last N days (default 7).")
    parser.add_argument("--dry-run", action="store_true", help="Log what would be created, write nothing.")
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
    garmin_email = require_env("GARMIN_EMAIL")
    garmin_password = require_env("GARMIN_PASSWORD")
    notion_token = require_env("NOTION_TOKEN")
    database_id = require_env("NOTION_DATABASE_ID")

    end = date.today()
    start = end - timedelta(days=args.days)
    print(f"Syncing Garmin activities {start.isoformat()} → {end.isoformat()}")

    garmin = Garmin(
        garmin_email,
        garmin_password,
        prompt_mfa=lambda: input("Garmin MFA code: ").strip(),
    )
    garmin.login(TOKEN_DIR)

    activities = garmin.get_activities_by_date(start.isoformat(), end.isoformat())
    tracked = [
        a for a in activities
        if a.get("activityType", {}).get("typeKey") in TRACKED_TYPES
    ]
    print(f"Found {len(activities)} total activities, {len(tracked)} hikes/runs.")

    if args.debug:
        targets = tracked if args.debug == "all" else [a for a in tracked if str(a["activityId"]) == args.debug]
        if not targets:
            print(f"No tracked activity matches --debug {args.debug}.")
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

    notion = Notion(auth=notion_token)
    data_source_id = resolve_data_source_id(notion, database_id)

    sleep_cache: dict[str, float | None] = {}
    created = skipped = failed = 0
    for a in tracked:
        activity_id = str(a["activityId"])
        label = f"{a.get('activityName', '?')} [{activity_id}] ({a['activityType']['typeKey']})"

        try:
            if already_synced(notion, data_source_id, activity_id):
                print(f"  skip  {label}")
                skipped += 1
                continue

            try:
                zones_payload = garmin.get_activity_hr_in_timezones(activity_id)
            except Exception as e:  # noqa: BLE001
                print(f"    (hr zones unavailable: {e})")
                zones_payload = None
            zones = zone_minutes(zones_payload)

            try:
                detail = garmin.get_activity(activity_id)
            except Exception as e:  # noqa: BLE001
                print(f"    (detail unavailable: {e})")
                detail = None

            activity_date = a["startTimeLocal"][:10]
            sleep_hrs = sleep_hours_for_date(garmin, activity_date, sleep_cache)

            props = build_properties(a, zones, detail, sleep_hrs)

            if args.dry_run:
                print(f"  DRY   {label}")
                created += 1
                continue

            notion.pages.create(
                parent={"type": "data_source_id", "data_source_id": data_source_id},
                properties=props,
            )
            print(f"  new   {label}")
            created += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {label}: {e}")
            failed += 1

    verb = "would create" if args.dry_run else "created"
    print(f"\nDone. {verb}={created} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
