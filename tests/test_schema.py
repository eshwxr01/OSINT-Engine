"""The JSON schema is the contract with the Stage 6 reporting channel."""
import copy
import json
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

SCHEMA = json.loads((Path(__file__).parent.parent / "report_schema.json").read_text(encoding="utf-8"))


def validator():
    return jsonschema.Draft202012Validator(SCHEMA, format_checker=jsonschema.FormatChecker())


def test_schema_is_itself_valid():
    jsonschema.Draft202012Validator.check_schema(SCHEMA)


def test_real_pipeline_report_conforms(full_report):
    validator().validate(full_report)


def test_report_with_no_findings_conforms(monkeypatch, engine, evidence_bytes):
    """Everything blocked or empty (no browser, no network) must still produce a valid report."""
    monkeypatch.setattr(engine, "multi_crop_visual_reverse_search", lambda *a, **k: [])
    report = engine.run_full_recon(evidence_bytes, "evidence.jpg", check_archives=False)
    validator().validate(report)
    assert report["osint_findings"]["ranked_leads"] == []


def _mutations():
    def drop(path):
        def apply(r):
            node = r
            for key in path[:-1]:
                node = node[key]
            del node[path[-1]]
        return apply

    def setter(path, value):
        def apply(r):
            node = r
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = value
        return apply

    return {
        "missing caveats": drop(["analysis_caveats"]),
        "empty caveats": setter(["analysis_caveats"], []),
        "missing diagnostics": drop(["engine_diagnostics"]),
        "wrong schema version": setter(["report_schema_version"], "2"),
        "uppercase sha256": setter(["evidence_profile", "sha256"], "AB" * 32),
        "short sha256": setter(["evidence_profile", "sha256"], "abc"),
        "bad resolution": setter(["evidence_profile", "resolution"], "big"),
        "unknown pivot origin": setter(["seed_inputs", "pivot_username_origin"], "guess"),
        "unknown slice method": setter(["search_slices", 0, "method"], "magic"),
        "bad crop box": setter(["search_slices", 0, "box"], [1, 2, 3]),
        "unknown strength": setter(["osint_findings", "ranked_leads", 0, "image_comparison", "strength"], "STRONG"),
        "hash distance out of range": setter(["osint_findings", "ranked_leads", 0, "image_comparison", "phash_distance"], 99),
        "bad matched_by": setter(["osint_findings", "ranked_leads", 0, "image_comparison", "matched_by"], "vibes"),
        "bad diagnostic level": setter(["engine_diagnostics"], [{"level": "fatal", "phase": "x", "message": "y"}]),
        "non-http source url": setter(["osint_findings", "other_web_sources", 0, "url"], "javascript:alert(1)"),
        "missing lead summary": drop(["osint_findings", "lead_summary"]),
    }


@pytest.mark.parametrize("name", list(_mutations()))
def test_broken_reports_are_rejected(full_report, name):
    broken = copy.deepcopy(full_report)
    _mutations()[name](broken)
    with pytest.raises(jsonschema.ValidationError):
        validator().validate(broken)
