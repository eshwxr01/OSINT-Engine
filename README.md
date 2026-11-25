# OSINT Attribution & Spread Mapping Engine
**Stage 5 Module — Deepfake Detection & Media Provenance Pipeline**

[![tests](https://github.com/eshwxr01/OSINT-Engine/actions/workflows/tests.yml/badge.svg)](https://github.com/eshwxr01/OSINT-Engine/actions/workflows/tests.yml)

## Overview
This repository contains the **OSINT, Scraping & Entity Attribution Module**: a standalone Python engine designed and developed independently as part of a multi-stage **Deepfake Detection & Original Source Identification** system.

Once upstream forensic stages flag a suspect image as manipulated or synthetic, this module **gathers leads** about where visually similar content appears online, which accounts and domains are associated with those appearances, and how the content spreads. It checks each lead's image against the evidence (perceptual hashes plus feature matching that survives crops, rotations and mirroring), cross-checks dates against the Internet Archive, and delivers everything as a schema-validated forensic JSON report plus an interactive link graph.

> **What the output is, and is not.** A reverse-image hit shows *where similar content appears*, not *who created, owns or posted it*. An associated account may be a victim, a repost, or the original photographer. Everything here is a **lead for a human analyst to verify**, and every report says so in an `analysis_caveats` section.

---

## End-to-End System Pipeline

```text
[1. Intake & Hashing]             ──▶ MD5 / SHA-256 chain-of-custody fingerprinting
         │
[2. Metadata & Payload]           ──▶ ExifTool & Binwalk static inspection
         │
[3. Pixel-Level Manipulation]     ──▶ ELA, noise & clone analysis, ML & frequency detection
         │
[4. Provenance & Signatures]      ──▶ C2PA manifests, YARA rules, GAN/Diffusion artifact checks
         │
╔══════════════════════════════════════════════════════════════════════════════════╗
║ [5. OSINT & Scraping (THIS REPO)] ──▶ Multi-crop lead search, social dorking,    ║
║                                       image verification & ranking, archive      ║
║                                       dating, WHOIS/IP recon & link-graphing     ║
╚══════════════════════════════════════════════════════════════════════════════════╝
         │
[6. Legal / Reporting Channel]    ──▶ Forensic JSON (report_schema.json) & graph handoff (External)
```

---

## What the Engine Does

1. **Fingerprint** the evidence: SHA-256, MD5, pHash, dHash and resolution (EXIF orientation is applied before hashing and slicing).
2. **Slice** the image into three views:
   - `FULL_IMAGE`
   - `SUBJECT_FOCUS_CROP`: located with **OpenCV face detection** (retried on rotated copies, so tilted heads up to roughly 40° either way are found), expanded to head, shoulders and torso. If no confident face is found, or OpenCV is missing, it falls back to a fixed region and the report records which method was used per slice.
   - `CONTEXT_BACKGROUND_CROP`: the full image with the subject region **blurred out**, so the reverse search matches the scenery rather than the person.
3. **Reverse-search** each slice on Yandex Images with Playwright, using one browser session. CAPTCHA pages are detected and reported. The preview thumbnail of every hit is captured too.
4. **Pivot on handles:** extracts usernames from social URLs in the hits, then runs randomized multi-platform dorks (ddgs) on Instagram, X, YouTube, Reddit, Facebook, LinkedIn, GitHub, Telegram and Pinterest. Each result records whether the handle actually appears in it (`handle_in_result`).
5. **Scrape page metadata:** OpenGraph tags, publication timestamps (classified as `published`, `updated` or `time_tag_unverified`) and outbound links (BeautifulSoup4).
6. **Verify images:** every hit's **result thumbnail** and every page's **primary image** (`og:image`) is compared with every evidence slice using two independent signals (see below).
7. **Rank leads:** best match first; `lead_summary` names the closest lead, the earliest *explicitly published* similar lead, and the earliest *archived* similar lead.
8. **Date against the Internet Archive:** the first Wayback Machine capture of the strongest leads (up to 5), cached on disk for 7 days. This is independent of what a page says about itself.
9. **Infrastructure recon:** server IPs and WHOIS registrar / earliest creation date (python-whois) on the registrable domain.
10. **Graph:** interactive physics-enabled entity network (NetworkX + PyVis) with degree-centrality pivot-hub highlighting, near-duplicate sightings emphasised, tooltips showing how each lead matched and when it was first archived, and handles labelled as *unverified associations*.

### How images are compared
| Signal | What it catches | What it misses |
|---|---|---|
| **Perceptual hashes** (pHash / dHash, also checked on the mirrored image) | Near-identical copies, recompression, mirrored reposts | Crops, rotation, heavy edits |
| **ORB feature matching + RANSAC** (OpenCV) | Crops, rescaling, rotation, mirroring, recompression, partial overlap | Smooth, low-texture images; very small images |

A lead is `NEAR_DUPLICATE` (pHash distance ≤ 6), `SIMILAR` (≤ 12) or `WEAK`. When the hash says `WEAK` but at least 15 keypoints agree on one geometric transform, the lead is upgraded to `SIMILAR` and marked `matched_by: "features"`. On the test images, edited copies produced 62–842 agreeing keypoints and unrelated images 7 or fewer. Thresholds live at the top of `image_matching.py`.

---

## Repository Structure
- `spiderfoot_engine.py` — core engine: hashing, face-aware slicing, Playwright reverse search, dorking, safe fetching with retries, ranking, WHOIS/DNS recon.
- `image_matching.py` — perceptual-hash and ORB feature comparison of a candidate image against the evidence.
- `archive_lookup.py` — Internet Archive (Wayback CDX) query building and parsing.
- `maltego_visualizer.py` — converts the forensic JSON into an interactive PyVis entity map.
- `osint_utils.py` — small pure helpers (domain matching, timestamp / WHOIS normalisation, text sanitising).
- `app.py` — Streamlit testbench: run scans, browse ranked leads, diagnostics, graph and raw JSON.
- `report_schema.json` — JSON Schema of the report, for the Stage 6 handoff.
- `tests/` — offline pytest suite (no browser or internet needed). `tests/data/face_sample.jpg` is the public-domain NASA astronaut photo from scikit-image's sample set.
- `requirements.txt` / `requirements-dev.txt` — runtime / development dependencies.
- `.github/workflows/tests.yml` — CI running the tests on Python 3.10 and 3.12 for every push.

---

## Local Setup & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/eshwxr01/OSINT-Engine.git
   cd OSINT-Engine
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies and a browser for Playwright:**
   ```bash
   pip install -r requirements.txt
   playwright install chromium
   ```
   The engine tries **Microsoft Edge, then Google Chrome, then Playwright's bundled Chromium**, so any one of them is enough. To force one, set `OSINT_BROWSER_CHANNEL` to `msedge`, `chrome` or `chromium`.

4. **Launch the Streamlit testbench:**
   ```bash
   streamlit run app.py
   ```

### Using the engine from Python
```python
from spiderfoot_engine import MiniSpiderFoot
from maltego_visualizer import MiniMaltego

with open("evidence.jpg", "rb") as f:
    image_bytes = f.read()

engine = MiniSpiderFoot(cache_dir="outputs/cache")   # cache_dir is optional (archive lookups)
report = engine.run_full_recon(
    image_bytes,
    "evidence.jpg",
    seed_username="",        # optional: a handle YOU supply
    seed_url="",             # optional: a known page URL
    enable_crops=True,
    auto_pivot=False,        # True = dork on the first handle found (unverified)
    check_archives=True,     # Internet Archive first-capture lookups for the strongest leads
)
html_path = MiniMaltego().build_graph_from_intel(report)   # writes outputs/maltego_graph_<sha>.html
```

### Validating a report (Stage 6 handoff)
```python
import json
import jsonschema   # pip install jsonschema

schema = json.load(open("report_schema.json", encoding="utf-8"))
jsonschema.validate(report, schema)
```

### Running the tests
```bash
pip install -r requirements-dev.txt
pytest
```
About 210 tests, roughly 40 seconds (most of it the headless Streamlit app checks). They use a local web server and fake browser objects, so they need no internet and no Playwright browser.

---

## Report Format
`run_full_recon()` returns a JSON-serialisable dict, described formally by `report_schema.json`:

| Key | Contents |
|---|---|
| `engine_version`, `report_schema_version` | Versions that produced the report |
| `evidence_profile` | Filename, SHA-256, MD5, pHash, dHash, resolution, scan time (UTC) |
| `seed_inputs` | `provided_url`, `pivot_username`, `pivot_username_origin` (`analyst_supplied` / `auto_first_visual_hit` / `null`), `auto_discovered_handles` |
| `search_slices` | Per slice: name, `method` (`whole_image` / `face_detection` / `fixed_heuristic`), crop box, hashes |
| `osint_findings` | `linked_social_accounts`, `other_web_sources`, `page_metadata_intel`, `infrastructure_intel`, `ranked_leads`, `lead_summary` |
| `ranked_leads[]` | Best `image_comparison` (strength, `matched_by`, hash distances, `feature_inliers`, `mirrored`), `compared_via` (`page_image` / `result_thumbnail`), `archive` first-capture info |
| `analysis_caveats` | Plain-language limits of the findings (always present) |
| `engine_diagnostics` | Every skipped page, failed query, CAPTCHA, retry, missing browser or fallback, as `{level, phase, message}` |

**Read `engine_diagnostics` before trusting an empty result.** An empty section can mean "blocked / rate-limited / offline / layout changed" rather than "nothing exists".

---

## Known Limitations
- **The live integrations are tested with fakes, not against the live services.** Yandex Images, DuckDuckGo-style search and the Wayback API are exercised in the test suite with fake browser pages and realistic response fixtures. Their real-world behaviour can differ, and the diagnostics exist to show you when it does.
- **Yandex is the only reverse-image engine.** Its page layout can change or CAPTCHA you, and thumbnail capture assumes each result is a link containing an image. Other engines can be added by writing another `_search_single_slice`-style method, but none is included because their pages cannot be verified here.
- **Search backends are scraped, not official APIs.** Use low volumes and respect each service's terms.
- **Face detection is Haar-cascade based** (classic OpenCV): it was validated on one real photo, handles tilts up to about 40° either way, and does not handle profile faces, small or occluded faces, or non-human subjects. Rotated detections below a confidence floor are discarded, and the labelled fixed-region fallback is used instead.
- **Feature matching needs texture.** Smooth, very small or heavily stylised images fall back to hash-only comparison. Neither signal proves one image is the original of another.
- **Archive dates are upper bounds** ("this page existed no later than...") and a missing capture proves nothing. Page-metadata timestamps can be absent, edited or wrong; only explicit published-time tags feed `earliest_published_lead`.
- **Outbound fetches are SSRF-guarded** (public hosts only, redirects re-checked, size-capped, bounded retries), but the check is not immune to DNS-rebinding races. Run it as a local analyst tool, not as a public service.
- OpenCV 5.x wheels no longer ship the Haar cascade file, so `requirements.txt` pins `opencv-python-headless<5`.

---

## What's New

### 2.1
- **Result-thumbnail verification:** the preview image Yandex returns for each hit is compared with the evidence, not just each page's `og:image`.
- **Feature matching (ORB + RANSAC)** catches crops, rotations, rescaling and mirrored reposts that perceptual hashes miss; mirrored copies are also detected by hash.
- **Internet Archive dating** of the strongest leads, with a 7-day on-disk cache.
- **Tilted faces** are now detected (rotated retries with a confidence floor).
- **Retries with backoff** for 429 / 5xx responses, honouring `Retry-After`.
- **`report_schema.json`** and a validation test for the Stage 6 handoff.
- Headless Streamlit app tests and Python 3.10 / 3.12 CI.

### 2.0
- Subject/background slices use real face detection; the background slice masks the subject instead of cropping an overlapping region.
- Handles are presented as **associated / unverified**, never as a "suspect" or "target"; auto-pivot is optional and flagged.
- Failures are surfaced in `engine_diagnostics` and the UI instead of being swallowed.
- Browser launch no longer hard-codes Microsoft Edge.
- Fixed: substring-based platform and domain matching (`box.com` was labelled X/Twitter; `olive.com` was blocked), subreddit and YouTube channel-ID names read as usernames, pages silently dropped on a missing `og:title` content attribute, list-valued WHOIS fields printed raw, WHOIS creation dates missing from graph tooltips, a misleading `direct_image_url` field, and results vanishing from the UI after clicking Download.
- Moved from the deprecated `st.components.v1.html` to `st.iframe`; renamed `test_app.py` to `app.py`; added a real test suite and CI.

---

## Disclaimer
Developed for academic research, digital media forensics, and authorized OSINT provenance verification. Results are investigative leads and must be independently verified before any attribution or legal action.
