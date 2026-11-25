"""Headless run of the Streamlit app (no browser) using Streamlit's AppTest."""
import io
from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).parent.parent / "app.py")


class FakeUpload(io.BytesIO):
    name = "evidence.jpg"

    @property
    def size(self):
        return len(self.getvalue())


@pytest.fixture
def app(monkeypatch, tmp_path, full_report, evidence_bytes):
    """The app with a stubbed uploader and an engine that returns a genuine report."""
    import streamlit as st

    import spiderfoot_engine

    seen = {}

    def fake_recon(self, image_bytes, filename, **kwargs):
        seen["calls"] = seen.get("calls", 0) + 1
        seen["kwargs"] = kwargs
        return full_report

    monkeypatch.chdir(tmp_path)                                     # app writes ./outputs
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: FakeUpload(evidence_bytes))
    monkeypatch.setattr(spiderfoot_engine.MiniSpiderFoot, "run_full_recon", fake_recon)
    at = AppTest.from_file(APP, default_timeout=60)
    at.seen = seen
    return at


def test_asks_for_an_upload_first(monkeypatch):
    import streamlit as st
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: None)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("Upload an image" in i.value for i in at.info)


def test_waits_for_the_run_button(app):
    at = app.run()
    assert not at.exception and any("click" in i.value.lower() for i in at.info)
    assert app.seen.get("calls") is None


def test_scan_renders_every_tab(app):
    at = app.run()
    at.button[0].click().run()
    assert not at.exception
    assert [t.label for t in at.tabs] == [
        "Ranked Leads", "Visual Sightings", "Profiles (Dorking)",
        "Page Metadata & Infrastructure", "Diagnostics", "Raw Forensic JSON",
    ]
    assert (Path("outputs")).is_dir() and list(Path("outputs").glob("maltego_graph_*.html"))
    assert list(Path("outputs").glob("spiderfoot_intel_*.json"))


def test_results_survive_a_rerun_like_a_download_click(app):
    at = app.run()
    at.button[0].click().run()
    at.run()                                                         # what clicking Download triggers
    assert not at.exception and len(at.tabs) == 6
    assert app.seen["calls"] == 1                                    # not re-scanned


def test_sidebar_options_reach_the_engine(app):
    at = app.run()
    at.text_input[0].set_value("some.user")
    at.checkbox[0].uncheck()        # multi-crop
    at.checkbox[1].uncheck()        # auto-pivot
    at.checkbox[2].uncheck()        # internet archive
    at.button[0].click().run()
    assert not at.exception
    assert app.seen["kwargs"] == {
        "seed_url": "", "seed_username": "some.user",
        "enable_crops": False, "auto_pivot": False, "check_archives": False,
    }


def test_default_options_enable_everything(app):
    at = app.run()
    at.button[0].click().run()
    kwargs = app.seen["kwargs"]
    assert kwargs["enable_crops"] and kwargs["auto_pivot"] and kwargs["check_archives"]


def test_engine_failure_is_shown_not_raised(app, monkeypatch):
    import spiderfoot_engine

    def boom(self, *a, **k):
        raise ValueError("Could not decode image: nope")

    monkeypatch.setattr(spiderfoot_engine.MiniSpiderFoot, "run_full_recon", boom)
    at = app.run()
    at.button[0].click().run()
    assert not at.exception
    assert any("Could not decode image" in e.value for e in at.error)
