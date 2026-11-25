"""Tests for the 2.1 features: retries, result-thumbnail comparison, archive-aware ranking,
rotated-face detection and report hygiene."""
import base64
import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import spiderfoot_engine as se
from helpers import (
    FakeBrowser,
    FakePage,
    QuietHandler,
    evidence_for,
    make_structured_image,
    noise_bytes,
    textured_image,
    to_bytes,
)
from spiderfoot_engine import FetchError, MiniSpiderFoot

FACE = Path(__file__).parent / "data" / "face_sample.jpg"
FACE_CENTRE = (224, 113)      # centre of the face in the 512x512 sample photo


def cmp_(strength, distance, inliers=None):
    return {"best_matching_slice": "FULL_IMAGE", "phash_distance": distance, "dhash_distance": distance,
            "mirrored": False, "strength": strength, "matched_by": "hash", "feature_inliers": inliers}


# ------------------------------------------------------------ retry / backoff
def test_transient_503_is_retried_and_recovers(engine, web):
    _, base = web
    result = engine._safe_get(f"{base}/flaky")
    assert b"recovered" in result.content
    assert QuietHandler.hits["/flaky"] == 2
    assert any("retry 1/2" in d["message"] for d in engine.diagnostics)


def test_retries_are_bounded(engine, web):
    _, base = web
    with pytest.raises(FetchError, match="HTTP 503"):
        engine._safe_get(f"{base}/always503")
    assert QuietHandler.hits["/always503"] == 3          # 1 try + 2 retries, then it stops


def test_retries_can_be_turned_off(engine, web):
    _, base = web
    with pytest.raises(FetchError, match="HTTP 503"):
        engine._safe_get(f"{base}/always503", retries=0)
    assert QuietHandler.hits["/always503"] == 1


def test_retry_wait_honours_retry_after_and_is_capped():
    class Resp:
        def __init__(self, retry_after=None):
            self.headers = {"Retry-After": retry_after} if retry_after is not None else {}

    live = MiniSpiderFoot(delay_scale=1.0)
    assert live._retry_wait(Resp("2"), 0) == 2.0
    assert live._retry_wait(Resp("120"), 0) == 5.0        # never waits more than 5s
    assert live._retry_wait(Resp(), 0) == pytest.approx(0.8)
    assert live._retry_wait(Resp(), 2) == pytest.approx(3.2)
    assert MiniSpiderFoot(delay_scale=0)._retry_wait(Resp("2"), 0) == 0


# --------------------------------------------------- result thumbnails (Yandex)
def test_search_slice_collects_thumbnail_references(engine):
    anchors = [
        {"href": "https://a.example/p1", "text": "Result one here", "title": "", "thumb": "https://img.example/t1.jpg"},
        {"href": "https://b.example/p2", "text": "Result two here", "title": "", "thumb": "data:image/png;base64,AAAA"},
        {"href": "https://c.example/p3", "text": "Result three", "title": "", "thumb": "javascript:alert(1)"},
        {"href": "https://d.example/p4", "text": "Result four", "title": ""},
    ]
    matches = engine._search_single_slice(FakeBrowser(FakePage(anchors)), {"name": "FULL_IMAGE", "bytes": b"0"}, set())
    assert matches[0]["thumbnail_url"] == "https://img.example/t1.jpg" and matches[0]["_thumb_data"] is None
    assert matches[1]["thumbnail_url"] is None and matches[1]["_thumb_data"].startswith("data:image/png")
    assert matches[2]["thumbnail_url"] is None and matches[2]["_thumb_data"] is None    # unsafe scheme dropped
    assert matches[3]["thumbnail_url"] is None


def test_compare_thumbnails_handles_http_and_data_uris_and_reports_failures(engine, web):
    folder, base = web
    img = textured_image(1)
    (folder / "thumb.jpg").write_bytes(to_bytes(img.resize((200, 150)), quality=70))
    (folder / "page.html").write_text("<html>not an image</html>")
    uri = "data:image/jpeg;base64," + base64.b64encode(to_bytes(img, quality=60)).decode()
    sources = [
        {"url": "https://a.example/1", "thumbnail_url": f"{base}/thumb.jpg", "_thumb_data": None},
        {"url": "https://b.example/2", "thumbnail_url": None, "_thumb_data": uri},
        {"url": "https://c.example/3", "thumbnail_url": f"{base}/missing.jpg", "_thumb_data": None},
        {"url": "https://d.example/4", "thumbnail_url": None, "_thumb_data": "data:text/html;base64,PGI+"},
        {"url": "https://e.example/5", "thumbnail_url": f"{base}/page.html", "_thumb_data": None},
        {"url": "https://f.example/6"},
    ]
    assert engine.compare_thumbnails(sources, evidence_for(img)) == 2
    assert sources[0]["thumbnail_comparison"]["strength"] in ("NEAR_DUPLICATE", "SIMILAR")
    assert sources[1]["thumbnail_comparison"]["strength"] == "NEAR_DUPLICATE"
    assert not any("thumbnail_comparison" in s for s in sources[2:])
    reasons = " | ".join(d["message"] for d in engine.diagnostics if d["phase"] == "thumbnail_compare")
    assert "HTTP 404" in reasons and "unsupported data URI" in reasons and "not an image" in reasons


def test_compare_thumbnails_respects_the_cap(engine):
    img = textured_image(1)
    uri = "data:image/jpeg;base64," + base64.b64encode(to_bytes(img)).decode()
    sources = [{"url": f"https://s{i}.example/", "_thumb_data": uri} for i in range(5)]
    assert engine.compare_thumbnails(sources, evidence_for(img), max_items=2) == 2
    assert sum("thumbnail_comparison" in s for s in sources) == 2


def test_oversized_data_uri_is_refused():
    with pytest.raises(ValueError, match="too large"):
        MiniSpiderFoot._decode_data_uri("data:image/png;base64," + "A" * 3_000_000)
    with pytest.raises(ValueError):
        MiniSpiderFoot._decode_data_uri("data:image/png,not-base64")


# ----------------------------------------------------- lead ranking + archives
def test_lead_uses_the_better_of_page_image_and_thumbnail(engine):
    weak_page, strong_thumb = cmp_("WEAK", 30), cmp_("NEAR_DUPLICATE", 2, 120)
    url = "https://a.example/p"
    meta = {"source_url": url, "meta_title": "A", "detected_timestamp_utc": None, "timestamp_kind": None,
            "image_comparison": weak_page}
    leads, _ = engine.rank_leads([url], [{"url": url, "page_title": "t", "thumbnail_comparison": strong_thumb}], [meta])
    assert leads[0]["compared_via"] == "result_thumbnail" and leads[0]["image_comparison"] is strong_thumb
    assert leads[0]["comparisons"] == {"page_image": weak_page, "result_thumbnail": strong_thumb}

    meta["image_comparison"] = strong_thumb
    leads, _ = engine.rank_leads([url], [{"url": url, "thumbnail_comparison": weak_page}], [meta])
    assert leads[0]["compared_via"] == "page_image"

    leads, _ = engine.rank_leads([url], [{"url": url, "thumbnail_comparison": strong_thumb}], [])
    assert leads[0]["compared_via"] == "result_thumbnail" and leads[0]["analyzed"] is False


def test_archive_lookups_only_for_strong_leads_and_are_capped(engine, monkeypatch):
    calls = []
    monkeypatch.setattr(engine, "archive_first_capture",
                        lambda url: calls.append(url) or {"found": True, "first_capture_utc": "2020-01-01T00:00:00+00:00",
                                                          "snapshot_url": "x"})
    sources = [{"url": f"https://strong{i}.example/", "thumbnail_comparison": cmp_("NEAR_DUPLICATE", i)} for i in range(6)]
    sources += [{"url": f"https://weak{i}.example/", "thumbnail_comparison": cmp_("WEAK", 40)} for i in range(2)]
    leads, _ = engine.rank_leads([s["url"] for s in sources], sources, [], check_archives=True, max_archive_lookups=3)
    assert len(calls) == 3 and all("strong" in c for c in calls)
    assert all(l["archive"] is None for l in leads if "weak" in l["source_url"])

    calls.clear()
    engine.rank_leads([s["url"] for s in sources], sources, [], check_archives=False)
    assert calls == []


def test_earliest_archived_lead_is_the_oldest_capture(engine, monkeypatch):
    captures = {
        "https://new.example/": {"found": True, "first_capture_utc": "2022-01-01T00:00:00+00:00", "snapshot_url": "a"},
        "https://old.example/": {"found": True, "first_capture_utc": "2019-06-01T00:00:00+00:00", "snapshot_url": "b"},
        "https://none.example/": {"found": False, "first_capture_utc": None, "snapshot_url": None},
        "https://failed.example/": None,
    }
    monkeypatch.setattr(engine, "archive_first_capture", captures.get)
    sources = [{"url": u, "thumbnail_comparison": cmp_("NEAR_DUPLICATE", 1)} for u in captures]
    _, summary = engine.rank_leads(list(captures), sources, [], check_archives=True, max_archive_lookups=10)
    assert summary["earliest_archived_lead"] == "https://old.example/"


# ------------------------------------------------------------------ faces
def test_real_photo_gets_a_face_based_subject_crop():
    pytest.importorskip("cv2")
    engine = MiniSpiderFoot(delay_scale=0)
    slices = engine.generate_image_slices(FACE.read_bytes())
    subject = next(s for s in slices if s["name"] == "SUBJECT_FOCUS_CROP")
    assert subject["method"] == "face_detection"
    left, top, right, bottom = subject["box"]
    assert left <= FACE_CENTRE[0] <= right and top <= FACE_CENTRE[1] <= bottom
    assert (right - left) * (bottom - top) < 0.9 * 512 * 512
    assert not any(d["level"] == "warning" for d in engine.diagnostics)


@pytest.mark.parametrize("tilt", [15, 25, 35, -35, 45])
def test_tilted_faces_are_found_and_mapped_back(tilt):
    pytest.importorskip("cv2")
    engine = MiniSpiderFoot(delay_scale=0)
    cascade = engine._get_face_cascade()
    assert cascade is not None
    photo = Image.open(FACE).convert("RGB")
    tilted = photo.rotate(tilt, expand=True, fillcolor=(127, 127, 127))

    theta = math.radians(tilt)
    px, py = FACE_CENTRE[0] - photo.width / 2, FACE_CENTRE[1] - photo.height / 2
    expected = (tilted.width / 2 + px * math.cos(theta) + py * math.sin(theta),
                tilted.height / 2 - px * math.sin(theta) + py * math.cos(theta))

    faces = engine._detect_faces(tilted, cascade)
    assert faces, f"no face found at {tilt} degrees"
    x0, y0, x1, y1 = faces[0]
    assert math.dist(((x0 + x1) / 2, (y0 + y1) / 2), expected) < 35      # face is ~96px wide


@pytest.mark.parametrize("angle", [20, -20, 40, -40])
def test_unrotate_box_inverts_pil_rotation(angle):
    base = Image.new("RGB", (240, 160), (0, 0, 0))
    base.paste((255, 255, 255), (150, 40, 170, 60))                     # marker centred on (160, 50)
    rotated = base.rotate(angle, expand=True)
    ys, xs = np.where(np.asarray(rotated.convert("L")) > 128)
    box = MiniSpiderFoot._unrotate_box((xs.mean() - 10, ys.mean() - 10, 20, 20), angle, base.size, rotated.size)
    assert math.dist(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), (160, 50)) < 2.5


def test_faceless_images_stay_faceless_even_with_rotation_retries():
    pytest.importorskip("cv2")
    engine = MiniSpiderFoot(delay_scale=0)
    cascade = engine._get_face_cascade()
    for image in (make_structured_image(), textured_image(3), Image.fromarray(
            np.random.default_rng(1).integers(0, 256, (300, 400, 3), dtype=np.uint8))):
        assert engine._detect_faces(image, cascade) == []


class ScoredCascade:
    """Finds nothing upright; reports one face with a fixed confidence in every rotated copy."""
    def __init__(self, weight):
        self.weight = weight

    def detectMultiScale(self, gray, **kwargs):
        return np.empty((0, 4), dtype=int)

    def detectMultiScale3(self, gray, outputRejectLevels=False, **kwargs):
        return np.array([[100, 80, 60, 60]]), np.array([0]), np.array([self.weight])


def test_weak_rotated_detections_are_discarded_but_confident_ones_kept():
    pytest.importorskip("cv2")
    engine = MiniSpiderFoot(delay_scale=0)
    work = make_structured_image((400, 300))
    assert engine._detect_faces(work, ScoredCascade(se.ROTATED_FACE_MIN_WEIGHT - 0.5)) == []
    kept = engine._detect_faces(work, ScoredCascade(se.ROTATED_FACE_MIN_WEIGHT + 1.5))
    assert len(kept) == 1


# ------------------------------------------------------ report hygiene
def _keys(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _keys(value)
    elif isinstance(node, list):
        for item in node:
            yield from _keys(item)


def test_report_has_no_private_keys_and_is_plain_json(full_report):
    assert not [k for k in _keys(full_report) if str(k).startswith("_")]
    assert json.loads(json.dumps(full_report)) == full_report
    assert full_report["engine_version"] == se.ENGINE_VERSION
    assert full_report["report_schema_version"] == se.REPORT_SCHEMA_VERSION


def test_pipeline_ranks_a_thumbnail_only_match_and_dates_it_from_the_archive(full_report):
    leads = {l["source_url"].rsplit("/", 1)[-1]: l for l in full_report["osint_findings"]["ranked_leads"]}
    assert leads["bare.html"]["compared_via"] == "result_thumbnail"         # page has no og:image
    assert leads["bare.html"]["image_comparison"]["strength"] == "NEAR_DUPLICATE"
    assert leads["orig.html"]["compared_via"] == "page_image"
    assert leads["bare.html"]["archive"]["first_capture_utc"] == "2021-05-05T00:00:00+00:00"
    summary = full_report["osint_findings"]["lead_summary"]
    assert summary["earliest_archived_lead"] is not None
    assert any("Internet Archive" in c for c in full_report["analysis_caveats"])


def test_archive_checks_can_be_switched_off(monkeypatch, engine, web, evidence_bytes):
    from helpers import prepare_pipeline
    prepare_pipeline(monkeypatch, engine, web, evidence_bytes)
    monkeypatch.setattr(engine, "archive_first_capture", lambda url: pytest.fail("archive must not be queried"))
    report = engine.run_full_recon(evidence_bytes, "e.jpg", check_archives=False)
    assert all(l["archive"] is None for l in report["osint_findings"]["ranked_leads"])
    assert report["osint_findings"]["lead_summary"]["earliest_archived_lead"] is None
    assert not any("Internet Archive" in c for c in report["analysis_caveats"])


def test_noise_is_not_a_match_for_the_evidence(engine, web, evidence_bytes):
    """A thumbnail of something unrelated must not be promoted to a lead."""
    folder, base = web
    (folder / "noise.jpg").write_bytes(noise_bytes())
    slices = engine.generate_image_slices(evidence_bytes)
    ev = [{"name": s["name"], "phash": s["phash"], "dhash": s["dhash"], "_features": s["_features"]} for s in slices]
    sources = [{"url": "https://x.example/", "thumbnail_url": f"{base}/noise.jpg", "_thumb_data": None}]
    engine.compare_thumbnails(sources, ev)
    assert sources[0]["thumbnail_comparison"]["strength"] == "WEAK"
