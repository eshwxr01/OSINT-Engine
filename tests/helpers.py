"""Shared test helpers (synthetic images and a tiny local web server)."""
import functools
import random
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO

import imagehash
import numpy as np
from PIL import Image, ImageDraw

import image_matching


def make_structured_image(size=(400, 300)):
    """A gradient with shapes: has enough structure for perceptual hashing."""
    w, h = size
    img = Image.new("RGB", size, (30, 60, 120))
    d = ImageDraw.Draw(img)
    for i in range(h):
        d.line([(0, i), (w, i)], fill=(i * 255 // h, 80, 255 - i * 255 // h))
    d.ellipse((w * 0.3, h * 0.1, w * 0.7, h * 0.6), fill=(240, 200, 170))
    d.rectangle((w * 0.1, h * 0.65, w * 0.9, h * 0.95), fill=(20, 20, 20))
    return img


def textured_image(seed, size=(480, 360)):
    """Many random rectangles/ellipses/lines: rich in corners, so ORB finds plenty of keypoints.
    Different seeds give unrelated images."""
    rnd = random.Random(seed)
    img = Image.new("RGB", size, (rnd.randint(0, 255),) * 3)
    d = ImageDraw.Draw(img)
    for _ in range(140):
        x, y = rnd.randint(0, size[0]), rnd.randint(0, size[1])
        w, h = rnd.randint(8, 70), rnd.randint(8, 70)
        color = tuple(rnd.randint(0, 255) for _ in range(3))
        kind = rnd.random()
        if kind < 0.4:
            d.rectangle((x, y, x + w, y + h), fill=color)
        elif kind < 0.7:
            d.ellipse((x, y, x + w, y + h), fill=color)
        else:
            d.line((x, y, x + w, y + h), fill=color, width=rnd.randint(1, 4))
    return img


def jpeg_roundtrip(img, quality=35):
    return Image.open(BytesIO(to_bytes(img, quality=quality))).convert("RGB")


def to_bytes(img, fmt="JPEG", **kwargs):
    buf = BytesIO()
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def structured_bytes(size=(400, 300)):
    return to_bytes(make_structured_image(size), quality=92)


def noise_bytes(size=(400, 300), seed=7):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    return to_bytes(Image.fromarray(arr), quality=92)


class QuietHandler(SimpleHTTPRequestHandler):
    """Serves files from a directory; /redir -> 302 to /page.html; /loop -> redirect loop."""

    hits = {}

    def log_message(self, *args):
        pass

    def _reply(self, status, body=b"ok", ctype="text/html"):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/flaky", "/always503"):
            QuietHandler.hits[self.path] = QuietHandler.hits.get(self.path, 0) + 1
            if self.path == "/always503" or QuietHandler.hits[self.path] == 1:
                self._reply(503, b"busy")
            else:
                self._reply(200, b"<html><title>recovered</title></html>")
        elif self.path == "/redir":
            self.send_response(302)
            self.send_header("Location", "/page.html")
            self.end_headers()
        elif self.path == "/loop":
            self.send_response(302)
            self.send_header("Location", "/loop")
            self.end_headers()
        else:
            super().do_GET()


def start_server(directory):
    QuietHandler.hits = {}
    handler = functools.partial(QuietHandler, directory=str(directory))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


# ---------------------------------------------------------------- fakes
import spiderfoot_engine as se  # noqa: E402


class FakeLocator:
    def __init__(self, n=1):
        self._n = n
        self.first = self
        self.last = self

    def count(self):
        return self._n

    def set_input_files(self, *a, **k):
        pass

    def click(self, *a, **k):
        pass


class FakePage:
    def __init__(self, anchors, captcha=False):
        self.anchors = anchors
        self.captcha = captcha
        self.url = "https://yandex.com/images/"

    def goto(self, *a, **k):
        if self.captcha:
            self.url = "https://yandex.com/showcaptcha?retpath=x"

    def wait_for_timeout(self, *a):
        pass

    def locator(self, selector):
        return FakeLocator(1)

    def get_by_text(self, *a, **k):
        return FakeLocator(0)

    def eval_on_selector_all(self, selector, js):
        return self.anchors


class FakeContext:
    def __init__(self, page):
        self.page = page
        self.closed = False

    def new_page(self):
        return self.page

    def close(self):
        self.closed = True


class FakeBrowser:
    def __init__(self, page):
        self.page = page
        self.contexts = []
        self.closed = False

    def new_context(self, **kwargs):
        ctx = FakeContext(self.page)
        self.contexts.append(ctx)
        return ctx

    def close(self):
        self.closed = True


ANCHORS = [
    {"href": "https://www.instagram.com/someone/", "text": "Someone on Instagram", "title": ""},
    {"href": "https://www.instagram.com/someone/", "text": "duplicate", "title": ""},
    {"href": "https://www.google.com/intl/en/about", "text": "Google junk", "title": ""},
    {"href": "https://yandex.com/support", "text": "Yandex junk", "title": ""},
    {"href": "/relative/link", "text": "relative", "title": ""},
    {"href": "javascript:void(0)", "text": "js", "title": ""},
    {"href": "https://olive.example/photo", "text": "x", "title": "Photo title from attribute"},
    {"href": "https://news.example/story", "text": "", "title": ""},
]


class FakeDDGS:
    """Stand-in for ddgs.DDGS. `behaviour` maps a substring of the query to results or an Exception."""
    behaviour = {}
    default = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def text(self, query, max_results=3):
        for needle, outcome in self.behaviour.items():
            if needle in query:
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome
        return list(self.default)


def prepare_pipeline(monkeypatch, engine, web, evidence_bytes):
    folder, base = web
    (folder / "evidence.jpg").write_bytes(evidence_bytes)
    (folder / "noise.jpg").write_bytes(noise_bytes())
    (folder / "orig.html").write_text("""<html><head><title>Original photo shoot</title>
        <meta property="article:published_time" content="2023-02-02T08:00:00Z">
        <meta property="og:image" content="/evidence.jpg"></head><body></body></html>""")
    (folder / "other.html").write_text("""<html><head><title>Unrelated page</title>
        <meta property="og:image" content="/noise.jpg"></head><body></body></html>""")

    monkeypatch.setattr(engine, "multi_crop_visual_reverse_search", lambda image_bytes, enable_crops=True, slices=None: [
        {"platform": "INSTAGRAM", "url": "https://www.instagram.com/first.hit/", "page_title": "A post by first.hit",
         "search_slice": "FULL_IMAGE", "match_type": "VISUAL MATCH (FULL_IMAGE)"},
        {"platform": "WEB", "url": f"{base}/orig.html", "page_title": "Original photo shoot",
         "search_slice": "SUBJECT_FOCUS_CROP", "match_type": "VISUAL MATCH (SUBJECT_FOCUS_CROP)"},
        {"platform": "WEB", "url": f"{base}/other.html", "page_title": "Unrelated page content here",
         "search_slice": "CONTEXT_BACKGROUND_CROP", "match_type": "VISUAL MATCH (CONTEXT_BACKGROUND_CROP)"},
    ])
    FakeDDGS.behaviour = {}
    FakeDDGS.default = [{"href": "https://github.com/first.hit", "title": "first.hit", "body": "bio"}]
    monkeypatch.setattr(se, "DDGS", FakeDDGS)
    monkeypatch.setattr(se.whois, "whois", lambda d: type("W", (), {"registrar": "Reg", "creation_date": None})())
    monkeypatch.setattr(se.socket, "gethostbyname", lambda d: "93.184.216.34")
    return base


def evidence_for(img, name="FULL_IMAGE"):
    """Evidence-slice dict (hashes + ORB features) for an image, as run_full_recon builds them."""
    return [{
        "name": name,
        "phash": str(imagehash.phash(img)),
        "dhash": str(imagehash.dhash(img)),
        "_features": image_matching.evidence_features(img),
    }]
