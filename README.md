# Garmin → Notion Exercise Log Sync

Pulls hikes and runs from Garmin Connect into a Notion database. Idempotent — safe to
re-run on a schedule.

## What it does

Fetches Garmin activities in a rolling window, filters to `hiking`, `running`, and
`trail_running`, and creates a Notion page per activity. Duplicates are prevented via
the `Garmin Activity ID` property, so re-running only writes new rows.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell / cmd)
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
cp .env.example .env            # then edit .env
```

Fill in `.env`:

```
GARMIN_EMAIL=you@example.com
GARMIN_PASSWORD=...
NOTION_TOKEN=secret_...
NOTION_DATABASE_ID=1e470776fe814d8c9a848ac1f24606b3
```

The Notion token comes from a **Notion internal integration**
(<https://www.notion.so/profile/integrations>). After creating it, open the Exercise
Log database in Notion → `...` menu → **Connections** → add your integration.

## Usage

```bash
python sync.py                     # sync last 7 days (default)
python sync.py --days 30           # sync last 30 days
python sync.py --dry-run --days 7  # preview without writing
python sync.py --debug             # dump raw Garmin JSON for matched activities
python sync.py --debug 22540601083 # debug a single activity
```

On first run you'll be prompted for a Garmin MFA code once. Tokens are cached to
`~/.garminconnect/garmin_tokens.json` and reused silently on later runs until the
refresh token expires (~1 year).

## What gets populated

Every Notion property below is auto-filled from Garmin:

| Property | Source |
|---|---|
| Name | Garmin activity name |
| Date | Start time (local) |
| Session Type | `hiking` → Hike, `running`/`trail_running` → Run |
| Garmin Activity ID | Used for dedupe — don't edit |
| Location | `locationName` |
| Distance (mi), Duration (hrs), Elevation Gain/Loss (ft), Steps, Calories | converted from metric |
| Avg HR, Max HR, Avg Moving Speed (mph), Avg Cadence (spm), Avg Temp (F) | summary |
| Aerobic TE, Anaerobic TE, Training Effect, Training Load | Garmin training metrics |
| HR Zone 1–5 (min) | `get_activity_hr_in_timezones` |
| RPE | `directWorkoutRpe ÷ 10` (Garmin stores it ×10) |
| Feel | `directWorkoutFeel` snapped to Very Weak / Weak / Normal / Strong / Very Strong |
| Personal Record | `pr` flag |
| Notes | Activity description, if you've set one in Garmin |

These stay **manual** (not touched by the script): Conditions, Focus, Exercises,
Progression Notes, Pack Weight (lbs), Trainer Led.

## Scheduling

Once it's working, run it automatically via Windows Task Scheduler (or cron on Unix):

```
python Z:\Documents\Projects\GarminNotionSync\sync.py --days 2
```

A 2-day window catches activities that sync late from the watch without being wasteful.
Re-runs are free since dedupe skips existing rows.

## Troubleshooting

**`429` during Garmin login** — Garmin rate-limits login attempts. The library retries
over a different transport automatically; if MFA still prompts, just enter it. The
token cache means this only happens on first setup.

**`DatabasesEndpoint object has no attribute 'query'`** — you're on a stale
`notion-client`. `pip install -U notion-client` (needs ≥ 2.5).

**`object_not_found` from Notion** — the integration hasn't been shared with the
database. In Notion: open the Exercise Log → `...` → Connections → add your
integration.

**An activity is missing** — check `python sync.py --debug --days N`. If the activity
type isn't `hiking` / `running` / `trail_running`, add it to `TRACKED_TYPES` in
`sync.py`.

## Files

- `sync.py` — the script
- `.env` — your credentials (gitignored)
- `.env.example` — template
- `requirements.txt` — `garminconnect`, `notion-client`, `python-dotenv`
