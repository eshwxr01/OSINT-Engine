import pytest
from PIL import ImageOps

import image_matching as im
from helpers import evidence_for, jpeg_roundtrip, textured_image


@pytest.fixture(scope="module")
def base():
    return textured_image(1)


@pytest.fixture(scope="module")
def evidence(base):
    return evidence_for(base)


def test_identical_image_is_confirmed_by_both_signals(base, evidence):
    result = im.compare_to_evidence(base, evidence)
    assert result["strength"] == "NEAR_DUPLICATE" and result["matched_by"] == "both"
    assert result["phash_distance"] == 0 and result["feature_inliers"] >= 100
    assert result["mirrored"] is False


def test_cropped_copy_is_missed_by_hashes_but_caught_by_features(base, evidence):
    w, h = base.size
    cropped = base.crop((int(w * .2), int(h * .2), int(w * .75), int(h * .75))).resize((400, 300))
    result = im.compare_to_evidence(cropped, evidence)
    assert im.strength_for(result["phash_distance"]) == "WEAK"          # the hash alone would say "unrelated"
    assert result["strength"] == "SIMILAR" and result["matched_by"] == "features"
    assert result["feature_inliers"] >= im.FEATURE_INLIER_THRESHOLD


def test_rotated_copy_is_verified(base, evidence):
    rotated = base.rotate(20, expand=True, fillcolor=(127, 127, 127))
    result = im.compare_to_evidence(rotated, evidence)
    assert result["strength"] in ("SIMILAR", "NEAR_DUPLICATE")
    assert result["feature_inliers"] >= im.FEATURE_INLIER_THRESHOLD


def test_mirrored_repost_is_flagged_as_mirrored(base, evidence):
    result = im.compare_to_evidence(ImageOps.mirror(base), evidence)
    assert result["strength"] == "NEAR_DUPLICATE" and result["mirrored"] is True


def test_heavy_recompression_and_downscale_still_match(base, evidence):
    for variant in (jpeg_roundtrip(base, 30), base.resize((160, 120))):
        result = im.compare_to_evidence(variant, evidence)
        assert result["strength"] in ("SIMILAR", "NEAR_DUPLICATE"), result


@pytest.mark.parametrize("seed", range(2, 9))
def test_unrelated_images_are_not_matched(evidence, seed):
    result = im.compare_to_evidence(textured_image(seed), evidence)
    assert result["strength"] == "WEAK" and result["matched_by"] == "hash"
    assert result["feature_inliers"] < im.FEATURE_INLIER_THRESHOLD


def test_best_slice_is_reported(base):
    other = textured_image(9)
    ev = evidence_for(other, "SUBJECT_FOCUS_CROP") + evidence_for(base, "FULL_IMAGE")
    assert im.compare_to_evidence(base, ev)["best_matching_slice"] == "FULL_IMAGE"


def test_without_opencv_only_hashes_are_used(monkeypatch, base):
    monkeypatch.setattr(im, "cv2", None)
    ev = evidence_for(base)
    assert ev[0]["_features"] == {"orig": None, "mirror": None}
    result = im.compare_to_evidence(base, ev)
    assert result["feature_inliers"] is None and result["matched_by"] == "hash"
    assert result["strength"] == "NEAR_DUPLICATE"


def test_tiny_candidate_falls_back_to_hashes(base, evidence):
    tiny = base.resize((40, 30))
    result = im.compare_to_evidence(tiny, evidence)
    assert result["feature_inliers"] is None          # too few keypoints to verify geometry


def test_empty_evidence_is_an_error(base):
    with pytest.raises(ValueError):
        im.compare_to_evidence(base, [])


@pytest.mark.parametrize("distance,expected", [
    (0, "NEAR_DUPLICATE"), (6, "NEAR_DUPLICATE"), (7, "SIMILAR"), (12, "SIMILAR"), (13, "WEAK"), (64, "WEAK"),
])
def test_strength_thresholds(distance, expected):
    assert im.strength_for(distance) == expected


def test_better_comparison_prefers_stronger_then_closer_then_more_inliers():
    def c(strength, d, inliers=None):
        return {"strength": strength, "phash_distance": d, "feature_inliers": inliers}

    assert im.better_comparison(None, None) is None
    a = c("WEAK", 30)
    assert im.better_comparison(a, None) is a and im.better_comparison(None, a) is a
    strong, similar = c("NEAR_DUPLICATE", 3), c("SIMILAR", 8)
    assert im.better_comparison(similar, strong) is strong
    assert im.better_comparison(c("SIMILAR", 8), c("SIMILAR", 5)) ["phash_distance"] == 5
    assert im.better_comparison(c("SIMILAR", 8, 20), c("SIMILAR", 8, 90))["feature_inliers"] == 90
    first, second = c("SIMILAR", 8, 20), c("SIMILAR", 8, 20)
    assert im.better_comparison(first, second) is first        # ties keep the first argument
