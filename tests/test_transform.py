"""Unit tests for the pure transform/util functions in sync.py (no network)."""

from __future__ import annotations

import pytest
from garminconnect import GarminConnectAuthenticationError, GarminConnectConnectionError

import sync


# --------------------------------------------------------------------------- #
# Unit conversions
# --------------------------------------------------------------------------- #


def test_conversions_happy_path():
    assert sync.m_to_mi(1609.344) == 1.0
    assert sync.m_to_ft(100) == 328
    assert sync.mps_to_mph(1) == 2.24
    assert sync.c_to_f(0) == 32.0
    assert sync.c_to_f(20) == 68.0
    assert sync.sec_to_hr(3600) == 1.0
    assert sync.sec_to_min(60) == 1.0


@pytest.mark.parametrize("fn", [sync.m_to_mi, sync.m_to_ft, sync.mps_to_mph, sync.c_to_f, sync.sec_to_hr, sync.sec_to_min])
def test_conversions_none_passthrough(fn):
    assert fn(None) is None


# --------------------------------------------------------------------------- #
# zone_minutes
# --------------------------------------------------------------------------- #


def test_zone_minutes_empty():
    assert sync.zone_minutes(None) == [0.0] * 5
    assert sync.zone_minutes([]) == [0.0] * 5


def test_zone_minutes_partial_and_full():
    payload = [
        {"zoneNumber": 1, "secsInZone": 600},   # 10 min
        {"zoneNumber": 3, "secsInZone": 90},     # 1.5 min
        {"zoneNumber": 9, "secsInZone": 999},    # out of range → ignored
        {"zoneNumber": 2},                        # missing secs → ignored
    ]
    assert sync.zone_minutes(payload) == [10.0, 0.0, 1.5, 0.0, 0.0]


# --------------------------------------------------------------------------- #
# session_type / tracked_types
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "type_key,expected",
    [
        ("hiking", "Hike"),
        ("running", "Run"),
        ("trail_running", "Run"),
        ("cycling", "Cycling"),
        ("lap_swimming", "Lap Swimming"),
        ("", None),
    ],
)
def test_session_type(type_key, expected):
    assert sync.session_type(type_key) == expected


def test_tracked_types_default():
    assert sync.tracked_types() == {"hiking", "running", "trail_running"}


def test_tracked_types_env_override(monkeypatch):
    monkeypatch.setenv("TRACKED_TYPES", "cycling, Lap_Swimming ,hiking")
    assert sync.tracked_types() == {"cycling", "lap_swimming", "hiking"}


# --------------------------------------------------------------------------- #
# build_properties
# --------------------------------------------------------------------------- #


SAMPLE_ACTIVITY = {
    "activityId": 123456789,
    "activityName": "Morning Run",
    "activityType": {"typeKey": "running"},
    "startTimeLocal": "2026-06-20 06:30:00",
    "distance": 8046.72,          # ~5 mi
    "duration": 1800,
    "movingDuration": 1700,
    "elevationGain": 100,
    "elevationLoss": 90,
    "steps": 5000,
    "calories": 400,
    "averageHR": 150.4,
    "maxHR": 175.6,
    "averageSpeed": 2.5,
    "averageRunningCadenceInStepsPerMinute": 170.2,
    "aerobicTrainingEffect": 3.24,
    "anaerobicTrainingEffect": 1.13,
    "activityTrainingLoad": 120.4,
    "pr": True,
    "trainingEffectLabel": "TEMPO",
    "locationName": "Trailhead",
}

SAMPLE_DETAIL = {
    "summaryDTO": {
        "directWorkoutRpe": 60,        # → RPE 6.0
        "directWorkoutFeel": 75,       # → "Strong"
        "calories": 410,
        "averageTemperature": 20,      # → 68 F
    },
    "description": "felt good",
}


def test_build_properties_full():
    props = sync.build_properties(SAMPLE_ACTIVITY, [10.0, 5.0, 3.0, 1.0, 0.0], SAMPLE_DETAIL, 7.5)

    assert props["Name"]["title"][0]["text"]["content"] == "Morning Run"
    assert props["Session Type"]["select"]["name"] == "Run"
    assert props["Date"]["date"]["start"] == "2026-06-20T06:30:00"
    assert props["Garmin Activity ID"]["rich_text"][0]["text"]["content"] == "123456789"
    assert props["Distance (mi)"]["number"] == 5.0
    assert props["Duration (hrs)"]["number"] == sync.sec_to_hr(1700)
    assert props["Avg HR"]["number"] == 150
    assert props["Max HR"]["number"] == 176
    assert props["Avg Temp (F)"]["number"] == 68.0
    assert props["Training Effect"]["select"]["name"] == "Tempo"
    assert props["Personal Record"]["checkbox"] is True
    assert props["HR Zone 1 (min)"]["number"] == 10.0
    assert props["Sleep (hrs)"]["number"] == 7.5
    assert props["RPE"]["number"] == 6.0
    assert props["Feel"]["select"]["name"] == "Strong"
    assert props["Notes"]["rich_text"][0]["text"]["content"] == "felt good"
    # Calories prefers the detail summary value.
    assert props["Calories"]["number"] == 410


def test_build_properties_minimal_no_detail():
    props = sync.build_properties(SAMPLE_ACTIVITY, [0.0] * 5, None, None)
    assert props["Sleep (hrs)"]["number"] is None
    assert "RPE" not in props        # no summaryDTO → no RPE/Feel
    assert "Feel" not in props
    assert "Notes" not in props
    assert props["Calories"]["number"] == 400   # falls back to list-entry calories


def test_feel_snaps_to_nearest_bucket():
    detail = {"summaryDTO": {"directWorkoutFeel": 60}}   # nearest bucket is 50 → "Normal"
    props = sync.build_properties(SAMPLE_ACTIVITY, [0.0] * 5, detail, None)
    assert props["Feel"]["select"]["name"] == "Normal"


# --------------------------------------------------------------------------- #
# _retry
# --------------------------------------------------------------------------- #


def test_retry_succeeds_after_transient(monkeypatch):
    monkeypatch.setattr(sync.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise GarminConnectConnectionError("temporary")
        return "ok"

    assert sync._retry(flaky, what="test", attempts=4, base=0.01) == "ok"
    assert calls["n"] == 3


def test_retry_reraises_non_transient(monkeypatch):
    monkeypatch.setattr(sync.time, "sleep", lambda *_: None)

    def boom():
        raise ValueError("not retryable")

    with pytest.raises(ValueError):
        sync._retry(boom, what="test", attempts=4, base=0.01)


def test_retry_gives_up_after_attempts(monkeypatch):
    monkeypatch.setattr(sync.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def always_fail():
        calls["n"] += 1
        raise GarminConnectConnectionError("down")

    with pytest.raises(GarminConnectConnectionError):
        sync._retry(always_fail, what="test", attempts=3, base=0.01)
    assert calls["n"] == 3


# --------------------------------------------------------------------------- #
# _mfa_prompt — headless safety (no network)
# --------------------------------------------------------------------------- #


class _FakeStdin:
    def __init__(self, tty: bool):
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def test_mfa_prompt_raises_without_tty(monkeypatch):
    monkeypatch.setattr(sync.sys, "stdin", _FakeStdin(tty=False))
    with pytest.raises(GarminConnectAuthenticationError):
        sync._mfa_prompt()


def test_mfa_prompt_raises_on_eof(monkeypatch):
    monkeypatch.setattr(sync.sys, "stdin", _FakeStdin(tty=True))

    def _eof(_prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", _eof)
    with pytest.raises(GarminConnectAuthenticationError):
        sync._mfa_prompt()
