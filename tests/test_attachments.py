"""Unit tests for the FIT/GPX attachment helpers in sync.py (no network).

Garmin and Notion are replaced with in-memory fakes; the only real dependency
exercised is ``garmin-fit-sdk`` (for the FIT→JSON decode), guarded with
``importorskip`` so the rest of the suite runs without it.
"""

from __future__ import annotations

import datetime
import io
import json
import zipfile

import pytest

import sync


# --------------------------------------------------------------------------- #
# env helpers
# --------------------------------------------------------------------------- #


def test_env_bool(monkeypatch):
    monkeypatch.delenv("FLAG", raising=False)
    assert sync.env_bool("FLAG", True) is True
    assert sync.env_bool("FLAG", False) is False
    for truthy in ("1", "true", "YES", "on", "y"):
        monkeypatch.setenv("FLAG", truthy)
        assert sync.env_bool("FLAG", False) is True
    for falsy in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("FLAG", falsy)
        assert sync.env_bool("FLAG", True) is False


def test_env_int(monkeypatch):
    monkeypatch.delenv("N", raising=False)
    assert sync.env_int("N", 20) == 20
    monkeypatch.setenv("N", "5")
    assert sync.env_int("N", 20) == 5
    monkeypatch.setenv("N", "not-a-number")
    assert sync.env_int("N", 20) == 20


# --------------------------------------------------------------------------- #
# _slug
# --------------------------------------------------------------------------- #


def test_slug_sanitizes_and_truncates():
    assert sync._slug("Morning Run-123") == "Morning_Run-123"
    assert sync._slug("Trail: 10mi / 2h!!") == "Trail_10mi_2h"
    assert sync._slug("") == "activity"
    assert sync._slug("///", fallback="x123") == "x123"
    assert len(sync._slug("z" * 200)) == 80


# --------------------------------------------------------------------------- #
# _fit_bytes_from_download
# --------------------------------------------------------------------------- #


def _zip_with(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_fit_bytes_unwraps_zip():
    raw = _zip_with({"99.fit": b"FITDATA", "readme.txt": b"nope"})
    assert sync._fit_bytes_from_download(raw) == b"FITDATA"


def test_fit_bytes_zip_without_fit_falls_back_to_first_entry():
    raw = _zip_with({"only.bin": b"BLOB"})
    assert sync._fit_bytes_from_download(raw) == b"BLOB"


def test_fit_bytes_raw_passthrough_and_empty():
    assert sync._fit_bytes_from_download(b"\x0e\x10rawfit") == b"\x0e\x10rawfit"
    assert sync._fit_bytes_from_download(b"") is None
    assert sync._fit_bytes_from_download(None) is None


# --------------------------------------------------------------------------- #
# fit_to_json (real garmin-fit-sdk round-trip)
# --------------------------------------------------------------------------- #


def _tiny_fit_bytes() -> bytes:
    """Encode a minimal but valid FIT activity file using the SDK encoder."""
    sdk = pytest.importorskip("garmin_fit_sdk")
    enc = sdk.Encoder()
    enc.on_mesg(
        mesg_num=sdk.Profile["mesg_num"]["FILE_ID"],
        mesg={
            "type": "activity",
            "manufacturer": "garmin",
            "time_created": datetime.datetime(2026, 6, 20, 6, 30, 0),
        },
    )
    enc.on_mesg(
        mesg_num=sdk.Profile["mesg_num"]["RECORD"],
        mesg={
            "timestamp": datetime.datetime(2026, 6, 20, 6, 30, 1),
            "heart_rate": 150,
            "altitude": 100.0,
        },
    )
    return enc.close()


def test_fit_to_json_decodes_streams():
    fit = _tiny_fit_bytes()
    blob = sync.fit_to_json(fit)
    parsed = json.loads(blob)
    assert "file_id_mesgs" in parsed
    assert "record_mesgs" in parsed
    assert parsed["record_mesgs"][0]["heart_rate"] == 150


# --------------------------------------------------------------------------- #
# _files_property
# --------------------------------------------------------------------------- #


def test_files_property_shape():
    prop = sync._files_property("up_1", "run.json")
    assert prop == {
        "files": [{"type": "file_upload", "file_upload": {"id": "up_1"}, "name": "run.json"}]
    }


# --------------------------------------------------------------------------- #
# Fakes for Garmin + Notion
# --------------------------------------------------------------------------- #


class FakeGarmin:
    ActivityDownloadFormat = sync.Garmin.ActivityDownloadFormat

    def __init__(self, fit: bytes | None = None, gpx: bytes | None = None, fail: bool = False):
        self._fit = fit
        self._gpx = gpx
        self._fail = fail

    def download_activity(self, activity_id, dl_fmt):
        if self._fail:
            raise ValueError("garmin download boom")
        if dl_fmt == self.ActivityDownloadFormat.ORIGINAL:
            return self._fit
        if dl_fmt == self.ActivityDownloadFormat.GPX:
            return self._gpx
        raise AssertionError(f"unexpected format {dl_fmt}")


class FakeFileUploads:
    def __init__(self):
        self.created: list[dict] = []
        self.sent: list[dict] = []

    def create(self, **kwargs):
        self.created.append(kwargs)
        return {"id": f"up_{len(self.created)}"}

    def send(self, **kwargs):
        self.sent.append(kwargs)
        return {"status": "uploaded"}


class FakeDataSources:
    def __init__(self, properties: dict, pages: list[dict] | None = None):
        self._properties = properties
        self._pages = pages or []

    def retrieve(self, data_source_id):
        return {"properties": self._properties}

    def query(self, **kwargs):
        # Single-page result; pagination is exercised separately.
        return {"results": self._pages, "has_more": False, "next_cursor": None}


class FakePages:
    def __init__(self):
        self.updated: list[tuple[str, dict]] = []

    def update(self, page_id, properties):
        self.updated.append((page_id, properties))
        return {"id": page_id}


class FakeNotion:
    def __init__(self, properties: dict | None = None, pages: list[dict] | None = None):
        self.file_uploads = FakeFileUploads()
        self.data_sources = FakeDataSources(properties or {}, pages)
        self.pages = FakePages()


# --------------------------------------------------------------------------- #
# notion_upload
# --------------------------------------------------------------------------- #


def test_notion_upload_creates_and_sends():
    notion = FakeNotion()
    upload_id = sync.notion_upload(notion, "run.xml", b"<gpx/>", "application/xml")
    assert upload_id == "up_1"
    assert notion.file_uploads.created[0]["filename"] == "run.xml"
    assert notion.file_uploads.created[0]["content_type"] == "application/xml"
    sent = notion.file_uploads.sent[0]
    assert sent["file_upload_id"] == "up_1"
    assert sent["file"] == ("run.xml", b"<gpx/>", "application/xml")


# --------------------------------------------------------------------------- #
# attachment_targets
# --------------------------------------------------------------------------- #


FILES_SCHEMA = {
    "FIT File": {"type": "files"},
    "GPX File": {"type": "files"},
    "Name": {"type": "title"},
}


def test_attachment_targets_detects_both(monkeypatch):
    monkeypatch.delenv("SYNC_ATTACHMENTS", raising=False)
    monkeypatch.delenv("NOTION_FIT_PROPERTY", raising=False)
    monkeypatch.delenv("NOTION_GPX_PROPERTY", raising=False)
    notion = FakeNotion(FILES_SCHEMA)
    assert sync.attachment_targets(notion, "ds") == {"fit": "FIT File", "gpx": "GPX File"}


def test_attachment_targets_disabled(monkeypatch):
    monkeypatch.setenv("SYNC_ATTACHMENTS", "false")
    notion = FakeNotion(FILES_SCHEMA)
    assert sync.attachment_targets(notion, "ds") == {}


def test_attachment_targets_skips_missing_and_wrong_type(monkeypatch):
    monkeypatch.delenv("SYNC_ATTACHMENTS", raising=False)
    monkeypatch.delenv("NOTION_FIT_PROPERTY", raising=False)
    monkeypatch.delenv("NOTION_GPX_PROPERTY", raising=False)
    schema = {"FIT File": {"type": "rich_text"}}  # wrong type; GPX File absent
    notion = FakeNotion(schema)
    assert sync.attachment_targets(notion, "ds") == {}


def test_attachment_targets_custom_property_names(monkeypatch):
    monkeypatch.delenv("SYNC_ATTACHMENTS", raising=False)
    monkeypatch.setenv("NOTION_FIT_PROPERTY", "Raw FIT")
    monkeypatch.setenv("NOTION_GPX_PROPERTY", "Track GPX")
    notion = FakeNotion({"Raw FIT": {"type": "files"}, "Track GPX": {"type": "files"}})
    assert sync.attachment_targets(notion, "ds") == {"fit": "Raw FIT", "gpx": "Track GPX"}


# --------------------------------------------------------------------------- #
# build_attachment_props
# --------------------------------------------------------------------------- #


ACTIVITY = {"activityId": 777, "activityName": "Morning Run"}


def test_build_attachment_props_uploads_both(monkeypatch):
    monkeypatch.delenv("NOTION_UPLOAD_MAX_MB", raising=False)
    fit = _tiny_fit_bytes()
    fit_zip = _zip_with({"777.fit": fit})
    garmin = FakeGarmin(fit=fit_zip, gpx=b"<gpx>track</gpx>")
    notion = FakeNotion()

    props, attached = sync.build_attachment_props(
        garmin, notion, ACTIVITY, {"fit": "FIT File", "gpx": "GPX File"}
    )

    assert attached == 2
    assert set(props) == {"FIT File", "GPX File"}
    # FIT went up as JSON, GPX as XML, both named from the activity slug.
    assert props["FIT File"]["files"][0]["name"] == "Morning_Run-777.json"
    assert props["GPX File"]["files"][0]["name"] == "Morning_Run-777.xml"
    assert notion.file_uploads.created[0]["content_type"] == "application/json"
    assert notion.file_uploads.created[1]["content_type"] == "application/xml"


def test_build_attachment_props_isolates_failures(monkeypatch):
    monkeypatch.delenv("NOTION_UPLOAD_MAX_MB", raising=False)
    garmin = FakeGarmin(fail=True)  # every download raises
    notion = FakeNotion()
    props, attached = sync.build_attachment_props(
        garmin, notion, ACTIVITY, {"fit": "FIT File", "gpx": "GPX File"}
    )
    assert attached == 0
    assert props == {}


def test_build_attachment_props_skips_missing_gpx(monkeypatch):
    monkeypatch.delenv("NOTION_UPLOAD_MAX_MB", raising=False)
    garmin = FakeGarmin(gpx=b"")  # empty GPX → skipped
    props, attached = sync.build_attachment_props(
        garmin, FakeNotion(), ACTIVITY, {"gpx": "GPX File"}
    )
    assert attached == 0
    assert props == {}


def test_build_attachment_props_size_guard(monkeypatch):
    monkeypatch.setenv("NOTION_UPLOAD_MAX_MB", "0")  # nothing fits under 0 MB
    garmin = FakeGarmin(gpx=b"<gpx>too big</gpx>")
    props, attached = sync.build_attachment_props(
        garmin, FakeNotion(), ACTIVITY, {"gpx": "GPX File"}
    )
    assert attached == 0
    assert props == {}


# --------------------------------------------------------------------------- #
# Page-reading helpers (backfill)
# --------------------------------------------------------------------------- #


def _rich_text(text: str) -> dict:
    return {"rich_text": [{"plain_text": text, "text": {"content": text}}]}


def _title(text: str) -> dict:
    return {"type": "title", "title": [{"plain_text": text, "text": {"content": text}}]}


def _files(*names: str) -> dict:
    return {"files": [{"name": n} for n in names]}


def test_page_activity_id_and_title():
    page = {
        "id": "pg1",
        "properties": {
            "Name": _title("Morning Run"),
            "Garmin Activity ID": _rich_text("777"),
        },
    }
    assert sync._page_activity_id(page) == "777"
    assert sync._page_title(page) == "Morning Run"


def test_page_activity_id_missing():
    assert sync._page_activity_id({"properties": {}}) is None
    assert sync._page_activity_id({"properties": {"Garmin Activity ID": _rich_text("  ")}}) is None


def test_page_missing_attachment_kinds():
    targets = {"fit": "FIT File", "gpx": "GPX File"}
    page = {"properties": {"FIT File": _files("run.json"), "GPX File": _files()}}
    # FIT already present, GPX empty → only GPX is missing.
    assert sync._page_missing_attachment_kinds(page, targets, force=False) == {"gpx": "GPX File"}
    # force re-attaches everything.
    assert sync._page_missing_attachment_kinds(page, targets, force=True) == targets


# --------------------------------------------------------------------------- #
# iter_data_source_pages
# --------------------------------------------------------------------------- #


def test_iter_data_source_pages_paginates():
    class Paginated:
        def __init__(self):
            self.calls = 0

        def query(self, **kwargs):
            self.calls += 1
            if kwargs.get("start_cursor") is None:
                return {"results": [{"id": "a"}], "has_more": True, "next_cursor": "c1"}
            return {"results": [{"id": "b"}], "has_more": False, "next_cursor": None}

    notion = FakeNotion()
    notion.data_sources = Paginated()
    ids = [p["id"] for p in sync.iter_data_source_pages(notion, "ds")]
    assert ids == ["a", "b"]
    assert notion.data_sources.calls == 2


# --------------------------------------------------------------------------- #
# _backfill_pages
# --------------------------------------------------------------------------- #


def _backfill_result() -> sync.SyncResult:
    return sync.SyncResult(window_start="backfill", window_end="backfill")


def test_backfill_fills_missing_and_skips_complete(monkeypatch):
    monkeypatch.delenv("NOTION_UPLOAD_MAX_MB", raising=False)
    fit = _tiny_fit_bytes()
    garmin = FakeGarmin(fit=_zip_with({"1.fit": fit}), gpx=b"<gpx/>")
    pages = [
        # Missing both → gets filled.
        {"id": "pg1", "properties": {
            "Name": _title("Run A"), "Garmin Activity ID": _rich_text("111"),
            "FIT File": _files(), "GPX File": _files()}},
        # Already complete → skipped, no download.
        {"id": "pg2", "properties": {
            "Name": _title("Run B"), "Garmin Activity ID": _rich_text("222"),
            "FIT File": _files("b.json"), "GPX File": _files("b.xml")}},
        # No activity id → skipped.
        {"id": "pg3", "properties": {"Name": _title("Manual"), "GPX File": _files()}},
    ]
    notion = FakeNotion({"FIT File": {"type": "files"}, "GPX File": {"type": "files"}}, pages)
    result = _backfill_result()

    sync._backfill_pages(
        garmin, notion, "ds", {"fit": "FIT File", "gpx": "GPX File"}, result,
        force=False, dry_run=False, pacing=0.0,
    )

    assert result.created == 1          # only pg1 filled
    assert result.attached == 2         # fit + gpx
    assert result.skipped == 2          # pg2 complete, pg3 no id
    assert result.failed == 0
    assert [pid for pid, _ in notion.pages.updated] == ["pg1"]
    updated_props = notion.pages.updated[0][1]
    assert set(updated_props) == {"FIT File", "GPX File"}


def test_backfill_dry_run_uploads_nothing(monkeypatch):
    monkeypatch.delenv("NOTION_UPLOAD_MAX_MB", raising=False)
    garmin = FakeGarmin(fit=b"x", gpx=b"y")
    pages = [{"id": "pg1", "properties": {
        "Name": _title("Run A"), "Garmin Activity ID": _rich_text("111"),
        "FIT File": _files(), "GPX File": _files()}}]
    notion = FakeNotion({"FIT File": {"type": "files"}, "GPX File": {"type": "files"}}, pages)
    result = _backfill_result()

    sync._backfill_pages(
        garmin, notion, "ds", {"fit": "FIT File", "gpx": "GPX File"}, result,
        force=False, dry_run=True, pacing=0.0,
    )

    assert result.created == 1
    assert result.attached == 2         # projected, not uploaded
    assert notion.pages.updated == []   # nothing written
    assert notion.file_uploads.created == []


def test_backfill_no_files_available_is_skipped(monkeypatch):
    monkeypatch.delenv("NOTION_UPLOAD_MAX_MB", raising=False)
    garmin = FakeGarmin(gpx=b"")        # empty GPX → nothing to attach
    pages = [{"id": "pg1", "properties": {
        "Name": _title("Treadmill"), "Garmin Activity ID": _rich_text("111"),
        "GPX File": _files()}}]
    notion = FakeNotion({"GPX File": {"type": "files"}}, pages)
    result = _backfill_result()

    sync._backfill_pages(
        garmin, notion, "ds", {"gpx": "GPX File"}, result,
        force=False, dry_run=False, pacing=0.0,
    )

    assert result.created == 0
    assert result.skipped == 1
    assert result.failed == 0
    assert notion.pages.updated == []
