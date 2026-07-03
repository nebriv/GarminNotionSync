"""Tests for activity_metrics.py against a synthetic hike (no real data / no network)."""

from __future__ import annotations

import datetime
import json

import numpy as np
import pytest

import activity_metrics as am


# --------------------------------------------------------------------------- #
# Synthetic track: flat → steep climb → rest stop at the top → descent → flat.
# HR trails grade by ~44 s; temp rises over time; cadence rises with grade (sparse).
# --------------------------------------------------------------------------- #

N = 900
DT = 4  # seconds per sample
LAG_SAMPLES = 11  # ~44 s


def _synth_track():
    rng = np.arange(N)
    t0 = datetime.datetime(2026, 6, 20, 6, 0, 0)

    ele = np.empty(N)
    ele[:150] = 100.0                              # flat approach
    ele[150:450] = np.linspace(100, 400, 300)      # steep climb (~20%)
    ele[450:500] = 400.0                           # rest at the top
    ele[500:800] = np.linspace(400, 120, 300)      # descent
    ele[800:] = 120.0                              # flat finish
    ele = ele + 2.0 * np.sin(rng * 0.5)            # a few metres of GPS noise

    # Freeze movement during the rest stop so speed drops below the gate.
    step = np.full(N, 6e-5)                         # ~4.8 m/sample at lat 44°
    step[450:500] = 0.0
    lon = -73.9 + np.cumsum(step)
    lat = np.full(N, 44.1)
    secs = rng * DT

    grade = np.gradient(ele) / 4.8 * 100.0
    g_lag = np.concatenate([np.zeros(LAG_SAMPLES), grade[:-LAG_SAMPLES]])
    hr = 110.0 + 1.0 * np.clip(g_lag, -25, 35)
    hr[450:500] = np.linspace(165, 128, 50)        # recovery during the stop
    hr = np.clip(hr, 90, 185)

    temp = 8.0 + 12.0 * (rng / N)                  # cool morning → warm afternoon
    cad = 50.0 + 0.35 * np.clip(grade, 0, 45)

    track = []
    for i in range(N):
        track.append({
            "lat": float(lat[i]), "lon": float(lon[i]), "ele": float(ele[i]),
            "t": t0 + datetime.timedelta(seconds=int(secs[i])),
            "hr": int(hr[i]),
            "temp": float(temp[i]),
            "cad": int(cad[i]) if i % 3 == 0 else None,  # ~1/3 coverage
        })
    return track


@pytest.fixture(scope="module")
def record():
    return am.compute_all(
        _synth_track(),
        fit={"ascent_ft": 1000.0, "calories": 3200},
        activity_meta={"id": "synth-1", "name": "Synthetic Hike"},
    )


# --------------------------------------------------------------------------- #
# rolling_mean edge safety
# --------------------------------------------------------------------------- #


def test_rolling_mean_preserves_constant_and_edges():
    x = np.full(20, 5.0)
    assert np.allclose(am.rolling_mean(x, 9), 5.0)  # constant stays constant at edges
    ramp = np.arange(50, dtype=float)
    sm = am.rolling_mean(ramp, 9)
    # Endpoints must not be dragged toward zero (naive conv would sag them).
    assert sm[0] > 1.0 and abs(sm[0] - ramp[0]) < 3.0
    assert abs(sm[-1] - ramp[-1]) < 3.0


def test_tobler_curve_peaks_at_slight_downhill():
    assert am.tobler_kmh(-5) == pytest.approx(6.0, abs=1e-6)   # peak at -5% grade
    assert am.tobler_kmh(0) < am.tobler_kmh(-5)
    assert am.tobler_kmh(20) < am.tobler_kmh(0)                # steep climb slower


# --------------------------------------------------------------------------- #
# Whole-record structure + JSON safety
# --------------------------------------------------------------------------- #


def test_record_is_json_serializable(record):
    blob = json.dumps(record)          # raises if any numpy scalar leaked through
    assert len(blob) > 500
    assert set(record) >= {
        "activity", "config", "summary", "heart_rate", "phases", "decoupling",
        "hr_lag", "hr_recovery", "cadence", "scatter", "correlations", "device",
    }


def test_config_travels_with_record(record):
    assert record["config"]["smoothing_window"] == 9
    assert record["config"]["moving_gate_mps"] == 0.15


def test_summary(record):
    s = record["summary"]
    assert s["distance_km"] > 2.0
    assert s["moving_s"] > 0 and s["stopped_s"] > 100        # the rest stop counts
    assert s["ascent_ft"] == 1000                            # FIT barometric override
    assert s["ascent_source"] == "barometric"


def test_phases_split_at_summit(record):
    p = record["phases"]
    assert 150 < p["summit_index"] < 550
    assert p["ascent"]["net_gain_m"] > 200                   # climbed
    assert p["descent"]["net_gain_m"] < 0                    # descended


def test_heart_rate_zones(record):
    hr = record["heart_rate"]
    assert hr["avg_hr"] is not None
    assert len(hr["zones_min"]) == 5
    assert sum(hr["zones_min"]) > 0
    assert hr["model"]["type"] == "percent_of_max"


def test_hr_lag_detected(record):
    lag = record["hr_lag"]
    assert lag["lag_s"] is not None and lag["lag_s"] > 0     # HR trails effort
    assert lag["r_at_lag"] >= lag["r_at_zero"]               # lagged fit is no worse


def test_hr_recovery_finds_stop(record):
    rec = record["hr_recovery"]
    assert rec["count"] >= 1
    assert rec["hardest"] is not None
    assert rec["hardest"]["hrr60_bpm"] > 0                   # HR dropped during the stop


def test_cadence_bands_rise_with_grade(record):
    cad = record["cadence"]
    assert cad["flat"] is not None and cad["steep"] is not None
    assert cad["steep"] > cad["flat"]
    assert 0.15 < cad["coverage"] < 0.5                      # ~1/3 sampled


def test_scatter_payload(record):
    hr_g = record["scatter"]["hr_vs_grade"]
    assert len(hr_g["points"]) > 10
    assert len(hr_g["curve"]) > 2
    assert hr_g["r"] is not None and hr_g["r"] > 0           # HR rises with grade
    # speed-vs-grade is populated even though there's no HR dependency
    assert len(record["scatter"]["speed_vs_grade"]["points"]) > 10


def test_correlations_matrix(record):
    m = record["correlations"]["matrix"]
    assert m["grade~hr"] is not None and m["grade~hr"] > 0
    assert "hr~elevation" in record["correlations"]["confounds"]


# --------------------------------------------------------------------------- #
# Degenerate inputs
# --------------------------------------------------------------------------- #


def test_compute_all_handles_no_hr():
    track = _synth_track()
    for p in track:
        p["hr"] = None
    rec = am.compute_all(track)
    json.dumps(rec)
    assert rec["heart_rate"]["avg_hr"] is None
    assert rec["hr_lag"]["lag_s"] is None
    assert rec["summary"]["distance_km"] > 2.0               # non-HR metrics still work
