"""
image_matching.py - compare a candidate image with the evidence slices.

Two independent signals are combined:

  * Perceptual hashes (pHash / dHash) - fast, good for near-identical copies,
    but they break on crops, rotations and heavy edits. The candidate is also
    compared in mirrored form, so flipped reposts are caught.
  * ORB feature matching + RANSAC (OpenCV, optional) - survives crops, scaling,
    rotation, mirroring and recompression. A copy is "verified" when enough
    keypoints agree on ONE consistent geometric transform.

Everything here is a heuristic that prioritises leads. It never proves that
one image is the original of another.
"""
from __future__ import annotations

import imagehash
from PIL import Image, ImageOps

try:  # optional: without OpenCV only the hash signal is used
    import cv2
    import numpy as np
except ImportError:  # tests simulate this by monkeypatching `cv2` to None
    cv2 = None
    np = None

# Perceptual-hash Hamming distances (64-bit hashes). Heuristics, not proof.
NEAR_DUPLICATE_MAX_DISTANCE = 6
SIMILAR_MAX_DISTANCE = 12

# Feature matching. Measured on synthetic and real photos: edited copies
# (crop / rotate / mirror / JPEG q35 / 1/3 downscale) gave 62-842 inliers,
# unrelated images gave 7 or fewer. 15 leaves a wide safety margin.
FEATURE_INLIER_THRESHOLD = 15
ORB_MAX_SIDE = 640
ORB_MIN_KEYPOINTS = 20
ORB_RATIO = 0.75
RANSAC_REPROJ_THRESHOLD = 4.0

STRENGTH_ORDER = {"NEAR_DUPLICATE": 0, "SIMILAR": 1, "WEAK": 2}


def strength_for(phash_distance: int) -> str:
    if phash_distance <= NEAR_DUPLICATE_MAX_DISTANCE:
        return "NEAR_DUPLICATE"
    if phash_distance <= SIMILAR_MAX_DISTANCE:
        return "SIMILAR"
    return "WEAK"


def comparison_rank(comparison):
    """Sort key: stronger, closer, more feature-verified comparisons sort first."""
    if not comparison:
        return (3, 64, 0)
    return (
        STRENGTH_ORDER.get(comparison.get("strength"), 3),
        comparison.get("phash_distance", 64),
        -(comparison.get("feature_inliers") or 0),
    )


def better_comparison(a, b):
    """The stronger of two comparisons (either may be None)."""
    if not a:
        return b
    if not b:
        return a
    return a if comparison_rank(a) <= comparison_rank(b) else b


# ------------------------------------------------------------------ features
def orb_features(img: Image.Image):
    """ORB keypoints + descriptors, or None (no OpenCV / too few keypoints)."""
    if cv2 is None:
        return None
    gray = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    scale = min(1.0, ORB_MAX_SIDE / max(h, w))
    if scale < 1.0:
        gray = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    keypoints, descriptors = cv2.ORB_create(nfeatures=1000).detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) < ORB_MIN_KEYPOINTS:
        return None
    return {"pts": np.float32([k.pt for k in keypoints]), "desc": descriptors}


def evidence_features(img: Image.Image):
    """Features of the evidence slice and of its mirror image (mirror reposts)."""
    return {"orig": orb_features(img), "mirror": orb_features(ImageOps.mirror(img))}


def count_inliers(query, candidate) -> int:
    """Keypoints that agree on one similarity transform (scale+rotation+shift)."""
    if cv2 is None or query is None or candidate is None:
        return 0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(query["desc"], candidate["desc"], k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < ORB_RATIO * p[1].distance]
    if len(good) < 6:
        return 0
    src = np.float32([query["pts"][m.queryIdx] for m in good]).reshape(-1, 1, 2)
    dst = np.float32([candidate["pts"][m.trainIdx] for m in good]).reshape(-1, 1, 2)
    _, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD)
    return int(mask.sum()) if mask is not None else 0


# ---------------------------------------------------------------- comparison
def compare_to_evidence(candidate: Image.Image, evidence):
    """Compare a candidate image with every evidence slice.

    `evidence` is a list of dicts: {"name", "phash", "dhash"} (hex strings) and,
    optionally, "_features" from evidence_features().
    """
    candidate = candidate.convert("RGB")
    variants = [(candidate, False), (ImageOps.mirror(candidate), True)]
    hashed = [(imagehash.phash(img), imagehash.dhash(img), flipped) for img, flipped in variants]

    best = None
    for ev in evidence:
        ev_p, ev_d = imagehash.hex_to_hash(ev["phash"]), imagehash.hex_to_hash(ev["dhash"])
        for cand_p, cand_d, flipped in hashed:
            p_dist, d_dist = int(cand_p - ev_p), int(cand_d - ev_d)
            if best is None or (p_dist, d_dist) < (best["phash_distance"], best["dhash_distance"]):
                best = {
                    "best_matching_slice": ev["name"],
                    "phash_distance": p_dist,
                    "dhash_distance": d_dist,
                    "mirrored": flipped,
                }
    if best is None:
        raise ValueError("no evidence slices to compare against")

    inliers, feature_slice, feature_mirrored = None, None, False
    candidate_features = orb_features(candidate)
    if candidate_features is not None and any(ev.get("_features") for ev in evidence):
        inliers = 0
        for ev in evidence:
            for orientation in ("orig", "mirror"):
                count = count_inliers((ev.get("_features") or {}).get(orientation), candidate_features)
                if count > inliers:
                    inliers, feature_slice, feature_mirrored = count, ev["name"], orientation == "mirror"

    strength = strength_for(best["phash_distance"])
    verified = inliers is not None and inliers >= FEATURE_INLIER_THRESHOLD
    if verified and strength == "WEAK":
        strength, matched_by = "SIMILAR", "features"
        best["best_matching_slice"], best["mirrored"] = feature_slice, feature_mirrored
    elif verified:
        matched_by = "both"
    else:
        matched_by = "hash"

    best.update(
        strength=strength,
        matched_by=matched_by,
        feature_inliers=inliers,
        note="Hash + feature similarity of the compared image; a lead, not proof of origin.",
    )
    return best
