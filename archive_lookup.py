"""
archive_lookup.py - earliest Internet Archive (Wayback Machine) capture of a URL.

A first-capture date is INDEPENDENT of what the page says about itself, which
makes it a useful cross-check on page-metadata timestamps. It is an upper
bound ("this page existed no later than..."), not a publication date, and a
missing capture proves nothing.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from urllib.parse import urlencode

CDX_ENDPOINT = "https://web.archive.org/cdx/search/cdx"
_HEADER = ["timestamp", "original", "statuscode"]


def build_cdx_url(url: str) -> str:
    """CDX query for the OLDEST successful capture (CDX sorts oldest-first)."""
    query = urlencode({
        "url": url,
        "output": "json",
        "limit": 1,
        "fl": "timestamp,original,statuscode",
        "filter": "statuscode:200",
    })
    return f"{CDX_ENDPOINT}?{query}"


def parse_cdx_response(raw: bytes) -> dict:
    """Parse a CDX JSON response into {'found': bool, ...}. Raises ValueError if malformed."""
    text = raw.decode("utf-8", "replace").strip()
    if not text:
        return {"found": False, "first_capture_utc": None, "snapshot_url": None}
    try:
        rows = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"CDX response is not JSON: {exc}") from exc
    if not isinstance(rows, list):
        raise ValueError("CDX response is not a list")

    data_rows = [r for r in rows if isinstance(r, list) and r != _HEADER]
    if not data_rows:
        return {"found": False, "first_capture_utc": None, "snapshot_url": None}

    row = data_rows[0]
    if len(row) < 2 or not str(row[0]).isdigit():
        raise ValueError("unexpected CDX row shape")
    stamp, original = str(row[0]), str(row[1])
    try:
        # Wayback stamps are normally 14 digits; if truncated, fill the rest with Jan 1st 00:00:00.
        padded = stamp[:14] + "00000101000000"[len(stamp[:14]):]
        captured = datetime.strptime(padded, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError(f"bad CDX timestamp {stamp!r}") from exc
    return {
        "found": True,
        "first_capture_utc": captured.isoformat(),
        "snapshot_url": f"https://web.archive.org/web/{stamp}/{original}",
    }
