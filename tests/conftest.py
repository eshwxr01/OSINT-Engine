import pytest

from helpers import prepare_pipeline, start_server, structured_bytes
from spiderfoot_engine import MiniSpiderFoot


@pytest.fixture
def evidence_bytes():
    return structured_bytes()


@pytest.fixture
def engine(monkeypatch):
    """Engine with the SSRF guard relaxed (tests talk to 127.0.0.1) and no sleeping.

    The Internet Archive lookup is stubbed so the suite never touches the real network;
    archive tests build their own engine and fake `_safe_get`.
    """
    e = MiniSpiderFoot(allow_private_hosts=True, delay_scale=0)
    monkeypatch.setattr(e, "archive_first_capture", lambda url: None)
    return e


@pytest.fixture
def strict_engine():
    """Engine with production network safety on."""
    return MiniSpiderFoot(delay_scale=0)


@pytest.fixture
def web(tmp_path):
    server, base = start_server(tmp_path)
    yield tmp_path, base
    server.shutdown()
    server.server_close()


@pytest.fixture
def full_report(monkeypatch, engine, web, evidence_bytes):
    """A genuine run_full_recon() report: page-image match, thumbnail-only match, archive dates."""
    folder, base = web
    prepare_pipeline(monkeypatch, engine, web, evidence_bytes)
    (folder / "thumb.jpg").write_bytes(evidence_bytes)
    (folder / "bare.html").write_text("<html><head><title>Bare page</title></head><body></body></html>")

    def fake_visual(image_bytes, enable_crops=True, slices=None):
        def hit(url, title, thumb=None):
            return {"platform": "WEB", "url": url, "page_title": title, "search_slice": "FULL_IMAGE",
                    "match_type": "VISUAL MATCH (FULL_IMAGE)", "thumbnail_url": thumb, "_thumb_data": None}
        return [
            hit("https://www.instagram.com/first.hit/", "A post by first.hit"),
            hit(f"{base}/orig.html", "Original photo shoot"),
            hit(f"{base}/bare.html", "Bare page without og image", thumb=f"{base}/thumb.jpg"),
        ]

    monkeypatch.setattr(engine, "multi_crop_visual_reverse_search", fake_visual)
    monkeypatch.setattr(engine, "archive_first_capture", lambda url: {
        "found": True,
        "first_capture_utc": "2021-05-05T00:00:00+00:00",
        "snapshot_url": "https://web.archive.org/web/20210505000000/" + url,
    })
    return engine.run_full_recon(evidence_bytes, "evidence.jpg")
