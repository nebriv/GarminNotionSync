"""Physiology metrics from a parsed Garmin activity track.

Pure compute — no Notion, no Garmin, numpy-only. ``compute_all()`` turns a list of
trackpoints into a structured record: summary, HR zones, ascent/descent phases, aerobic
decoupling, HR response lag, HR recovery, cadence bands, and the scatter/correlation
payloads needed to redraw the response plots without re-parsing the track.

Implemented from the "Activity Metrics" handoff. Every parameter that moves a result
lives in :class:`Config` and is written into the record under ``config`` so activities
stay comparable even if the parameters are retuned later.

Input contract — ``track`` is a list of dicts, one per trackpoint::

    {"lat": float, "lon": float, "ele": float,   # deg, deg, metres
     "t": datetime,                              # UTC-ish, naive is fine
     "hr": int | None, "temp": float | None, "cad": int | None}

``fit`` is an optional dict of device-summary fields GPS can't reproduce (calories,
training effect, barometric ascent/descent, …). When ``fit["ascent_ft"]`` is present it
overrides the GPS-derived ascent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

EARTH_RADIUS_M = 6_371_000.0
_M_TO_FT = 3.28084


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class Config:
    """Every knob that changes a number. Travels with the record for comparability."""

    moving_gate_mps: float = 0.15          # speed above which a sample counts as moving
    smoothing_window: int = 9              # centred rolling-mean window for elevation
    grade_clip_pct: float = 60.0           # clip grade to kill divide-by-tiny spikes
    max_hr: int = 190                      # for percent-of-max HR zones
    zone_fracs: tuple = (0.60, 0.70, 0.80, 0.90)  # lower bounds of zones 2..5
    rest_min_s: float = 90.0               # min stop length for HR recovery
    resample_s: float = 1.0                # uniform grid for HR-lag cross-correlation
    lag_max_s: float = 120.0               # max offset searched for HR lag
    vam_clip: float = 2000.0               # clip vertical rate (m/h) for lag/plots
    decouple_climb_grade: float = 2.0      # grade above which a sample is "climbing"
    scatter_points: int = 400              # downsample size for stored scatter clouds
    grade_bin_pct: float = 5.0             # width of grade bins for the response curves
    grade_bin_range: tuple = (-40.0, 40.0)  # binned-curve grade extent
    flat_grade_band: tuple = (-5.0, 5.0)   # cadence "flat" band
    steep_grade_band: tuple = (15.0, 60.0)  # cadence "steep" band


# --------------------------------------------------------------------------- #
# Derived per-sample series
# --------------------------------------------------------------------------- #


@dataclass
class Derived:
    n: int
    secs: np.ndarray
    dt: np.ndarray            # per-sample represented duration (weights)
    cumdist: np.ndarray       # cumulative distance, m
    speed: np.ndarray         # per-sample instantaneous speed, m/s
    ele: np.ndarray
    ele_s: np.ndarray         # smoothed elevation, m
    de: np.ndarray            # diff(ele_s), length n-1
    grade: np.ndarray         # percent, clipped
    vam: np.ndarray           # vertical speed, m/h
    hr: np.ndarray            # bpm, NaN where missing
    temp: np.ndarray
    cad: np.ndarray
    moving: np.ndarray
    hr_valid: np.ndarray
    cad_valid: np.ndarray


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Edge-safe centred rolling mean: divides by the true sample count at each index so
    the endpoints aren't dragged toward zero. Removing the edge handling sags the first
    and last few hundred metres and corrupts ascent/descent and the summit index."""
    x = np.asarray(x, float)
    if window <= 1 or x.size == 0:
        return x
    k = np.ones(window)
    counts = np.convolve(np.ones_like(x), k, mode="same")
    sums = np.convolve(x, k, mode="same")
    return sums / counts


def _haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _col(track, key):
    return np.array([np.nan if p.get(key) is None else float(p[key]) for p in track], float)


def derive(track: list[dict], cfg: Config) -> Derived:
    n = len(track)
    lat = np.array([p["lat"] for p in track], float)
    lon = np.array([p["lon"] for p in track], float)
    ele = np.array([p["ele"] for p in track], float)
    t0 = track[0]["t"]
    secs = np.array([(p["t"] - t0).total_seconds() for p in track], float)
    hr, temp, cad = _col(track, "hr"), _col(track, "temp"), _col(track, "cad")

    dt = np.gradient(secs) if n > 1 else np.ones(n)
    dt = np.where(dt > 0, dt, 0.0)

    if n > 1:
        step_d = _haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])
        step_dt = np.diff(secs)
        cumdist = np.concatenate([[0.0], np.cumsum(step_d)])
        ele_s = rolling_mean(ele, cfg.smoothing_window)
        de = np.diff(ele_s)
        with np.errstate(divide="ignore", invalid="ignore"):
            speed = np.concatenate([[0.0], np.where(step_dt > 0, step_d / step_dt, 0.0)])
            grade = np.concatenate([[0.0], np.where(step_d > 0, de / step_d * 100.0, 0.0)])
            vam = np.concatenate([[0.0], np.where(step_dt > 0, de / step_dt * 3600.0, 0.0)])
        grade = np.clip(np.nan_to_num(grade), -cfg.grade_clip_pct, cfg.grade_clip_pct)
        vam = np.nan_to_num(vam)
        speed = np.nan_to_num(speed)
    else:
        cumdist = np.zeros(n)
        ele_s = ele.copy()
        de = np.array([])
        speed = grade = vam = np.zeros(n)

    return Derived(
        n=n, secs=secs, dt=dt, cumdist=cumdist, speed=speed, ele=ele, ele_s=ele_s, de=de,
        grade=grade, vam=vam, hr=hr, temp=temp, cad=cad,
        moving=speed > cfg.moving_gate_mps, hr_valid=~np.isnan(hr), cad_valid=~np.isnan(cad),
    )


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #


def _wmean(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    if x.size == 0:
        return None
    if w.sum() <= 0:
        return float(np.mean(x))
    return float(np.average(x, weights=w))


def _round(v, ndigits=2):
    return None if v is None else round(float(v), ndigits)


def _hr_filled(d: Derived) -> np.ndarray | None:
    """HR with NaN gaps linearly interpolated over time; None if no valid HR."""
    if not d.hr_valid.any():
        return None
    return np.interp(d.secs, d.secs[d.hr_valid], d.hr[d.hr_valid])


# --------------------------------------------------------------------------- #
# Metric groups
# --------------------------------------------------------------------------- #


def summary(d: Derived, cfg: Config, fit: dict | None = None) -> dict:
    elapsed_s = float(d.secs[-1]) if d.n else 0.0
    moving_s = float(d.dt[d.moving].sum())
    stopped_s = max(0.0, elapsed_s - moving_s)
    dist_km = float(d.cumdist[-1] / 1000.0) if d.n else 0.0

    ascent_m = float(d.de[d.de > 0].sum()) if d.de.size else 0.0
    descent_m = float(-d.de[d.de < 0].sum()) if d.de.size else 0.0
    ascent_ft, descent_ft = ascent_m * _M_TO_FT, descent_m * _M_TO_FT
    baro = False
    if fit and fit.get("ascent_ft") is not None:
        ascent_ft = float(fit["ascent_ft"])
        ascent_m = ascent_ft / _M_TO_FT
        baro = True
    if fit and fit.get("descent_ft") is not None:
        descent_ft = float(fit["descent_ft"])
        descent_m = descent_ft / _M_TO_FT

    moving_h = moving_s / 3600.0
    return {
        "distance_km": _round(dist_km, 3),
        "distance_mi": _round(dist_km * 0.621371, 3),
        "elapsed_s": round(elapsed_s, 1),
        "moving_s": round(moving_s, 1),
        "stopped_s": round(stopped_s, 1),
        "ascent_m": _round(ascent_m, 1),
        "ascent_ft": _round(ascent_ft, 0),
        "descent_m": _round(descent_m, 1),
        "descent_ft": _round(descent_ft, 0),
        "ascent_source": "barometric" if baro else "gps",
        "avg_moving_speed_kmh": _round(dist_km / moving_h, 2) if moving_h > 0 else None,
        "avg_moving_pace_min_km": _round(moving_s / 60.0 / dist_km, 2) if dist_km > 0 else None,
    }


def heart_rate(d: Derived, cfg: Config) -> dict:
    model = {"type": "percent_of_max", "max_hr": cfg.max_hr, "zone_fracs": list(cfg.zone_fracs)}
    if not d.hr_valid.any():
        return {"avg_hr": None, "max_hr": None, "zones_min": [0.0] * 5, "model": model}

    avg = _wmean(d.hr[d.hr_valid], d.dt[d.hr_valid])
    edges = [-np.inf] + [f * cfg.max_hr for f in cfg.zone_fracs] + [np.inf]
    zones = []
    for i in range(5):
        band = d.hr_valid & (d.hr >= edges[i]) & (d.hr < edges[i + 1])
        zones.append(round(float(d.dt[band].sum()) / 60.0, 1))
    return {
        "avg_hr": _round(avg, 0),
        "max_hr": _round(np.nanmax(d.hr), 0),
        "zones_min": zones,
        "model": model,
    }


def _leg(d: Derived, lo: int, hi: int) -> dict:
    """Metrics for a contiguous slice [lo, hi) of samples."""
    hi = max(hi, lo + 1)
    sl = slice(lo, hi)
    dist_km = float(d.cumdist[hi - 1] - d.cumdist[lo]) / 1000.0
    moving = d.moving[sl]
    moving_s = float(d.dt[sl][moving].sum())
    hr_mask = d.hr_valid[sl]
    avg_hr = _wmean(d.hr[sl][hr_mask], d.dt[sl][hr_mask]) if hr_mask.any() else None
    net_gain = float(d.ele_s[hi - 1] - d.ele_s[lo])
    moving_h = moving_s / 3600.0
    return {
        "distance_km": _round(dist_km, 3),
        "moving_s": round(moving_s, 1),
        "avg_hr": _round(avg_hr, 0),
        "avg_moving_speed_kmh": _round(dist_km / moving_h, 2) if moving_h > 0 else None,
        "net_gain_m": _round(net_gain, 1),
        "vam_m_per_h": _round(net_gain / moving_h, 0) if moving_h > 0 else None,
    }


def phases(d: Derived, cfg: Config) -> dict:
    """Split at the elevation summit into ascent and descent legs. Single-summit
    assumption; multi-summit routes want peak detection instead."""
    if d.n < 3:
        return {"summit_index": 0, "ascent": None, "descent": None}
    summit = int(np.argmax(d.ele_s))
    summit = min(max(summit, 1), d.n - 2)
    return {
        "summit_index": summit,
        "summit_ele_m": _round(d.ele_s[summit], 1),
        "ascent": _leg(d, 0, summit + 1),
        "descent": _leg(d, summit, d.n),
    }


def aerobic_decoupling(d: Derived, cfg: Config) -> dict:
    """Vertical-metres-per-heartbeat drift between the first and second half of the climb.
    Positive = the engine drifted (same output cost more beats late); near zero = durable."""
    climb = d.moving & d.hr_valid & (d.grade > cfg.decouple_climb_grade)
    if climb.sum() < 20:
        return {"decoupling_pct": None, "note": "insufficient climbing samples"}
    tmid = float(np.median(d.secs[climb]))
    first = climb & (d.secs <= tmid)
    second = climb & (d.secs > tmid)

    def eff(mask):
        if mask.sum() < 5:
            return None
        hr_avg = _wmean(d.hr[mask], d.dt[mask])
        vam_avg = _wmean(d.vam[mask], d.dt[mask])
        return None if not hr_avg else vam_avg / hr_avg

    e1, e2 = eff(first), eff(second)
    if not e1 or e2 is None:
        return {"decoupling_pct": None, "note": "insufficient data"}
    return {
        "decoupling_pct": _round(100.0 * (e1 - e2) / e1, 2),
        "eff_first": _round(e1, 3),
        "eff_second": _round(e2, 3),
    }


def hr_response_lag(d: Derived, cfg: Config) -> dict:
    """Seconds HR trails effort (VAM), by cross-correlation on a uniform time grid."""
    hr_filled = _hr_filled(d)
    if hr_filled is None or d.n < 10 or d.secs[-1] < 3 * cfg.resample_s:
        return {"lag_s": None, "r_at_lag": None, "r_at_zero": None}
    grid = np.arange(0.0, float(d.secs[-1]), cfg.resample_s)
    if grid.size < 10:
        return {"lag_s": None, "r_at_lag": None, "r_at_zero": None}
    hr_u = np.interp(grid, d.secs, hr_filled)
    ef_u = np.interp(grid, d.secs, np.clip(d.vam, -cfg.vam_clip, cfg.vam_clip))
    hr_c, ef_c = hr_u - hr_u.mean(), ef_u - ef_u.mean()

    max_lag = int(cfg.lag_max_s / cfg.resample_s)
    best_lag, best_r, r0 = 0, -2.0, 0.0
    for lag in range(max_lag + 1):
        a = hr_c[lag:]
        b = ef_c[: ef_c.size - lag] if lag else ef_c
        if a.size < 10:
            break
        denom = a.std() * b.std()
        r = float((a * b).mean() / denom) if denom > 0 else 0.0
        if lag == 0:
            r0 = r
        if r > best_r:
            best_r, best_lag = r, lag
    return {
        "lag_s": round(best_lag * cfg.resample_s, 1),
        "r_at_lag": _round(best_r, 3),
        "r_at_zero": _round(r0, 3),
    }


def hr_recovery(d: Derived, cfg: Config) -> dict:
    """Heart-rate drop during rest stops. Reports every stop >= rest_min_s plus the
    hardest one (highest HR at onset), the most informative recovery."""
    hr_filled = _hr_filled(d)
    if hr_filled is None or d.n < 3:
        return {"count": 0, "stops": [], "hardest": None}
    stationary = d.speed < 0.8 * cfg.moving_gate_mps
    stops = []
    i, n = 0, d.n
    while i < n:
        if not stationary[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and stationary[j + 1]:
            j += 1
        dur = float(d.secs[j] - d.secs[i])
        if dur >= cfg.rest_min_s:
            onset = float(np.interp(d.secs[i], d.secs, hr_filled))
            at60 = float(np.interp(min(d.secs[i] + 60.0, d.secs[j]), d.secs, hr_filled))
            at_end = float(np.interp(d.secs[j], d.secs, hr_filled))
            stops.append({
                "t_start_h": _round(d.secs[i] / 3600.0, 2),
                "duration_s": round(dur, 1),
                "hr_onset": _round(onset, 0),
                "hrr60_bpm": _round(onset - at60, 0),
                "hrr_full_bpm": _round(onset - at_end, 0),
            })
        i = j + 1
    hardest = max(stops, key=lambda s: (s["hr_onset"] or 0)) if stops else None
    return {"count": len(stops), "stops": stops, "hardest": hardest}


def cadence_bands(d: Derived, cfg: Config) -> dict:
    """Weighted-mean cadence on a flat vs steep grade band. Directional only — cadence
    capture is sparse and the values are the watch's own units, not verified spm."""
    mv = d.moving
    n_moving = int(mv.sum())
    coverage = float((mv & d.cad_valid).sum()) / n_moving if n_moving else 0.0

    def band(lo, hi):
        m = mv & d.cad_valid & (d.grade >= lo) & (d.grade < hi)
        return _round(_wmean(d.cad[m], d.dt[m]), 1) if m.sum() >= 8 else None

    return {
        "flat": band(*cfg.flat_grade_band),
        "steep": band(*cfg.steep_grade_band),
        "coverage": round(coverage, 3),
    }


# --------------------------------------------------------------------------- #
# Scatter + correlation payloads (so plots redraw without re-parsing the track)
# --------------------------------------------------------------------------- #


def binned_response(x, y, weights, mask, edges) -> list[dict]:
    out = []
    for i in range(len(edges) - 1):
        b = mask & (x >= edges[i]) & (x < edges[i + 1])
        if int(b.sum()) >= 8:
            out.append({
                "x": round(float((edges[i] + edges[i + 1]) / 2), 2),
                "y": round(float(np.average(y[b], weights=weights[b])), 2),
                "n": int(b.sum()),
            })
    return out


def _downsample(x, y, mask, k) -> list[list[float]]:
    xs, ys = x[mask], y[mask]
    if xs.size == 0:
        return []
    step = max(1, xs.size // max(1, k))
    return [[round(float(a), 3), round(float(b), 3)] for a, b in zip(xs[::step], ys[::step])]


def _corr(x, y, mask):
    if int(mask.sum()) < 8:
        return None
    a, b = x[mask], y[mask]
    if a.std() == 0 or b.std() == 0:
        return None
    return round(float(np.corrcoef(a, b)[0, 1]), 3)


def _relationship(d, cfg, x, y, mask):
    edges = np.arange(cfg.grade_bin_range[0], cfg.grade_bin_range[1] + cfg.grade_bin_pct, cfg.grade_bin_pct)
    return {
        "points": _downsample(x, y, mask, cfg.scatter_points),
        "curve": binned_response(x, y, d.dt, mask, edges),
        "r": _corr(x, y, mask),
    }


def scatter_block(d: Derived, cfg: Config) -> dict:
    speed_kmh = d.speed * 3.6
    mv_hr = d.moving & d.hr_valid
    return {
        "hr_vs_grade": _relationship(d, cfg, d.grade, d.hr, mv_hr),
        "speed_vs_grade": _relationship(d, cfg, d.grade, speed_kmh, d.moving),
        "cadence_vs_grade": _relationship(d, cfg, d.grade, d.cad, d.moving & d.cad_valid),
    }


def correlations(d: Derived, cfg: Config) -> dict:
    """Pairwise correlations on the moving+valid-HR mask, with the two classic confounds
    flagged: HR~elevation is effort in an altitude costume; grade~temp is time-of-day."""
    mask = d.moving & d.hr_valid
    series = {
        "hr": d.hr, "grade": d.grade, "elevation": d.ele_s,
        "temp": d.temp, "vam": d.vam, "speed_kmh": d.speed * 3.6, "cadence": d.cad,
    }
    keys = list(series)
    matrix = {}
    for a in keys:
        for b in keys:
            if a < b:
                m = mask & ~np.isnan(series[a]) & ~np.isnan(series[b])
                matrix[f"{a}~{b}"] = _corr(series[a], series[b], m)
    return {
        "matrix": matrix,
        "confounds": {
            "hr~elevation": "climb effort rises with elevation on an out-and-back; "
            "this reads as altitude but is effort.",
            "grade~temp": "negative only because the climb was in the cool morning and "
            "the descent in afternoon heat; no physical link.",
        },
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def compute_all(
    track: list[dict],
    fit: dict | None = None,
    cfg: Config | None = None,
    activity_meta: dict | None = None,
) -> dict:
    """Turn a parsed track into the full metrics record. Assumes at least a few dozen
    points — guard with a length check before calling (see the sync integration)."""
    cfg = cfg or Config()
    d = derive(track, cfg)
    record = {
        "activity": activity_meta or {},
        "config": asdict(cfg),
        "summary": summary(d, cfg, fit),
        "heart_rate": heart_rate(d, cfg),
        "phases": phases(d, cfg),
        "decoupling": aerobic_decoupling(d, cfg),
        "hr_lag": hr_response_lag(d, cfg),
        "hr_recovery": hr_recovery(d, cfg),
        "cadence": cadence_bands(d, cfg),
        "scatter": scatter_block(d, cfg),
        "correlations": correlations(d, cfg),
    }
    if fit:
        record["device"] = fit
    return record


def tobler_kmh(grade_pct: float) -> float:
    """Tobler's hiking function (km/h) — the theoretical reference curve for
    speed-vs-grade. Not stored (it's a fixed function); add at render time."""
    s = grade_pct / 100.0
    return 6.0 * float(np.exp(-3.5 * abs(s + 0.05)))
