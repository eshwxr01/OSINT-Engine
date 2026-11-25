import json
import os
import time

import pytest

import archive_lookup as al
from spiderfoot_engine import FetchError, FetchResult, MiniSpiderFoot

HEADER = ["timestamp", "original", "statuscode"]


def cdx(*rows):
    return json.dumps([HEADER, *rows]).encode()


def test_build_cdx_url_asks_for_oldest_successful_capture():
    url = al.build_cdx_url("https://example.com/a?x=1&y=2")
    assert url.startswith(al.CDX_ENDPOINT + "?")
    assert "limit=1" in url and "output=json" in url and "filter=statuscode%3A200" in url
    assert "url=https%3A%2F%2Fexample.com%2Fa%3Fx%3D1%26y%3D2" in url


def test_parse_found_capture():
    out = al.parse_cdx_response(cdx(["20230202080015", "https://example.com/a", "200"]))
    assert out == {
        "found": True,
        "first_capture_utc": "2023-02-02T08:00:15+00:00",
        "snapshot_url": "https://web.archive.org/web/20230202080015/https://example.com/a",
    }


@pytest.mark.parametrize("raw", [b"", b"[]", json.dumps([HEADER]).encode(), b"   "])
def test_parse_no_capture(raw):
    assert al.parse_cdx_response(raw) == {"found": False, "first_capture_utc": None, "snapshot_url": None}


@pytest.mark.parametrize("raw", [b"<html>rate limited</html>", b'{"a": 1}', b'[["abc", "x"]]', b'[["99999999999999", "x", "200"]]'])
def test_parse_malformed_raises_value_error(raw):
    with pytest.raises(ValueError):
        al.parse_cdx_response(raw)


def test_short_timestamps_are_padded():
    out = al.parse_cdx_response(cdx(["2019", "https://e.com/", "200"]))
    assert out["first_capture_utc"] == "2019-01-01T00:00:00+00:00"
    out = al.parse_cdx_response(cdx(["20190315", "https://e.com/", "200"]))
    assert out["first_capture_utc"] == "2019-03-15T00:00:00+00:00"


# ------------------------------------------------------------ engine wrapper
def fake_get(payload, calls):
    def _get(url, **kwargs):
        calls.append(url)
        if isinstance(payload, Exception):
            raise payload
        return FetchResult(url, "application/json", payload)
    return _get


def test_archive_lookup_found_and_memoised(monkeypatch):
    engine = MiniSpiderFoot(delay_scale=0)
    calls = []
    monkeypatch.setattr(engine, "_safe_get", fake_get(cdx(["20200101000000", "https://e.com/p", "200"]), calls))
    first = engine.archive_first_capture("https://e.com/p")
    assert first["found"] and first["first_capture_utc"] == "2020-01-01T00:00:00+00:00"
    assert engine.archive_first_capture("https://e.com/p") == first
    assert len(calls) == 1                                       # second call served from memory


def test_archive_lookup_nothing_captured_is_a_result_not_a_failure(monkeypatch):
    engine = MiniSpiderFoot(delay_scale=0)
    monkeypatch.setattr(engine, "_safe_get", fake_get(b"[]", []))
    assert engine.archive_first_capture("https://e.com/x") == {"found": False, "first_capture_utc": None, "snapshot_url": None}


@pytest.mark.parametrize("error", [FetchError("HTTP 429"), ValueError("boom")])
def test_archive_lookup_failure_returns_none_and_is_reported(monkeypatch, error):
    engine = MiniSpiderFoot(delay_scale=0)
    monkeypatch.setattr(engine, "_safe_get", fake_get(error, []))
    assert engine.archive_first_capture("https://e.com/x") is None
    assert any(d["phase"] == "archive" and "failed" in d["message"] for d in engine.diagnostics)


def test_malformed_archive_response_is_a_reported_failure(monkeypatch):
    engine = MiniSpiderFoot(delay_scale=0)
    monkeypatch.setattr(engine, "_safe_get", fake_get(b"<html>nope</html>", []))
    assert engine.archive_first_capture("https://e.com/x") is None
    assert any(d["phase"] == "archive" for d in engine.diagnostics)


def test_disk_cache_survives_between_engines_and_expires(monkeypatch, tmp_path):
    payload = cdx(["20210101000000", "https://e.com/p", "200"])
    first_calls = []
    one = MiniSpiderFoot(delay_scale=0, cache_dir=str(tmp_path))
    monkeypatch.setattr(one, "_safe_get", fake_get(payload, first_calls))
    result = one.archive_first_capture("https://e.com/p")
    assert len(first_calls) == 1 and len(list(tmp_path.glob("archive_*.json"))) == 1

    second_calls = []
    two = MiniSpiderFoot(delay_scale=0, cache_dir=str(tmp_path))          # a fresh engine, as after an app rerun
    monkeypatch.setattr(two, "_safe_get", fake_get(payload, second_calls))
    assert two.archive_first_capture("https://e.com/p") == result
    assert second_calls == []

    # age the cache entry past its TTL -> it must be refetched
    entry = next(tmp_path.glob("archive_*.json"))
    data = json.loads(entry.read_text())
    data["cached_at"] = time.time() - 8 * 24 * 3600
    entry.write_text(json.dumps(data))
    third_calls = []
    three = MiniSpiderFoot(delay_scale=0, cache_dir=str(tmp_path))
    monkeypatch.setattr(three, "_safe_get", fake_get(payload, third_calls))
    three.archive_first_capture("https://e.com/p")
    assert len(third_calls) == 1


def test_corrupt_cache_entry_is_ignored(monkeypatch, tmp_path):
    engine = MiniSpiderFoot(delay_scale=0, cache_dir=str(tmp_path))
    path = engine._archive_cache_path("https://e.com/p")
    os.makedirs(tmp_path, exist_ok=True)
    open(path, "w").write("{not json")
    calls = []
    monkeypatch.setattr(engine, "_safe_get", fake_get(b"[]", calls))
    assert engine.archive_first_capture("https://e.com/p")["found"] is False
    assert len(calls) == 1 and any("unreadable cache" in d["message"] for d in engine.diagnostics)
