"""Screenshot capture for the screenshot arms (S0/S1/S2); runs only inside the lab container.

The three arms differ ONLY in how much of the browser's own UI the image shows; the page pixels
are the same pixels:

  S1  the real browser window: an X11 grab (mss) of the window rectangle on the Xvfb display.
      The window sits at (0, 0) with the configured size, re-verified on every capture from
      Chromium's own metrics and from the X server, so the image is what a user sees: tab strip,
      omnibox with the real security state (the twin CA is NSS-trusted, no ignore-certificate
      flag) and the page.
  S0  the same grab with the top `chrome_px` rows (tab strip + toolbar = outerHeight -
      innerHeight, measured at the first capture) painted a uniform neutral gray: identical
      dimensions, identical page pixels, no browser UI.
  S2  = S1. The twin renders the fake address bar (browser-in-the-browser) into the page itself
      when site.bitb, so the real omnibox still shows the true host while the page claims the
      legitimate URL.

The Xvfb screen is one pixel wider and taller than the window (docker/entrypoint.sh): Chromium
shrinks a window that would exactly fill the screen by 1 px (its X11 anti-maximize rule), so the
grab region is the window rectangle, not the whole screen.

Images are a function of the page state, not of the capture time: the lab image disables the
caret blink (GTK `gtk-cursor-blink = false`, docker/lab.Dockerfile), so a focused field shows a
steady caret instead of one that is on or off depending on when the grab happens. Hence the same
state renders the same pixels in every arm, and `page_sha256` (the pixels below the band) is
equal across S0 and S1 at matched steps of a deterministic trajectory (checked by
tests/docker/lab_check.py on the always-fill smoke). The page region is bit-reproducible across
browser contexts, runs and private per-worker displays; the chrome band only up to anti-aliasing
noise (a few pixels of the toolbar's rounded corners and icons differ by 2-3 levels between
browser contexts; lab_check band_repeat). Stimulus identity across episodes is therefore
`page_sha256` plus the OCR'd omnibox, never `png_sha256`.

Per-capture manipulation checks (`meta`, persisted in the episode record): image size, chrome_px,
the tesseract OCR of the chrome band as sent, the page URL and how legibly its host appears in
the omnibox, punycode-vs-Unicode flags for IDN hosts, the OCR of the in-page fake bar (S2), the
CDP security state, the number of visible browser X windows (a bubble or popup is an extra
window), sha256 of the PNG sent and of the page region, and, on the first capture of each page,
the pixel alignment of the grab with the page's own viewport screenshot (proves the band/page
split is exact).

There is no fallback: outside the lab (ORIGIN_BLIND_IN_LAB != '1' or DISPLAY unset), with a
misplaced or resized window, or when the browser chrome changes height mid-run (an infobar),
capture raises RuntimeError and the episode is a harness error, never a silently different image.
"""

from __future__ import annotations

import base64
import difflib
import hashlib
import io
import os
import shutil
import ssl
import subprocess
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

ARMS = ("S0", "S1", "S2")
CAPTURE_MODE = "xvfb-window"
NEUTRAL_GRAY = (128, 128, 128)
NSS_NICKNAME = "ob-local-ca"
OCR_SCALE = 3  # tesseract reads 13 px UI text reliably only when upscaled
OCR_CONFIG = "--psm 6"  # one uniform block: the tab strip + toolbar rows
HOST_MATCH_MIN = 0.85  # OCR-tolerant similarity at which a string counts as shown
ALIGN_MAX_DIFF = 0.002  # max fraction of differing pixels between the grab and the viewport shot
# Browser-UI notices that would tell a VLM the browser is automated (the launch flags suppress
# them: Playwright's default --disable-infobars, --test-type, no --enable-automation) and the
# insecure-connection label (the NSS-trusted CA must prevent it). The bundled Chrome for Testing
# build shows its own "... is only for automated testing" infobar in a startup window whenever
# infobars are enabled; tests/docker/lab_check.py provokes it as a positive control.
AUTOMATION_NOTICES = ("controlled by automated test software", "unsupported command-line flag",
                      "only for automated testing")
INSECURE_NOTICE = "not secure"

_WINDOW_JS = """() => ({screen_x: window.screenX, screen_y: window.screenY,
  outer_w: window.outerWidth, outer_h: window.outerHeight,
  inner_w: window.innerWidth, inner_h: window.innerHeight, dpr: window.devicePixelRatio})"""
# two animation frames: the last DOM change has been painted and handed to the compositor
_PAINT_JS = "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(() => r(true))))"
_BAR_JS = """() => { const e = document.getElementById('ob-addressbar'); if (!e) return null;
  const r = e.getBoundingClientRect(); return {x: r.left, y: r.top, w: r.width, h: r.height}; }"""

_DASHES = str.maketrans({c: "-" for c in "‐‑‒–—―−"})
# Cyrillic/Greek homoglyphs -> the Latin letter an English OCR model reads for them
_SKELETON = str.maketrans({
    "а": "a", "е": "e", "о": "o", "і": "i", "с": "c", "р": "p", "у": "y", "х": "x", "ѕ": "s",
    "ј": "j", "ԁ": "d", "һ": "h", "ӏ": "l", "ο": "o", "α": "a", "ε": "e", "ι": "i", "κ": "k",
    "ν": "v", "ρ": "p", "τ": "t", "υ": "u",
})


def to_data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


# ---------------------------------------------------------------- lab guard

def require_lab(env: dict | None = None) -> str:
    """Return the X display to grab; raise outside the lab (there is no fallback capture)."""
    env = os.environ if env is None else env
    if env.get("ORIGIN_BLIND_IN_LAB") != "1":
        raise RuntimeError("screenshot capture runs only inside the lab container "
                           "(ORIGIN_BLIND_IN_LAB=1, see docker/entrypoint.sh); no fallback")
    display = env.get("DISPLAY")
    if not display:
        raise RuntimeError("screenshot capture needs the lab's X display (DISPLAY is unset); no fallback")
    return display


# ---------------------------------------------------------------- pure image functions

def _image(png_or_img):
    from PIL import Image

    if isinstance(png_or_img, (bytes, bytearray)):
        return Image.open(io.BytesIO(png_or_img)).convert("RGB")
    return png_or_img.convert("RGB")


def png_bytes(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _neutralize(img, band_px: int, color: tuple[int, int, int] = NEUTRAL_GRAY):
    if not 0 < band_px < img.height:
        raise ValueError(f"band of {band_px}px does not fit an image {img.height}px high")
    out = img.copy()
    out.paste(color, (0, 0, out.width, band_px))
    return out


def neutralize_band(png: bytes, band_px: int, color: tuple[int, int, int] = NEUTRAL_GRAY) -> bytes:
    """S0 from S1: the top `band_px` rows become one uniform color; size and page pixels unchanged."""
    return png_bytes(_neutralize(_image(png), band_px, color))


def band_is_uniform(png_or_img, band_px: int, color: tuple[int, int, int] = NEUTRAL_GRAY) -> bool:
    img = _image(png_or_img)
    colors = img.crop((0, 0, img.width, band_px)).getcolors(maxcolors=2)
    return colors is not None and len(colors) == 1 and colors[0][1] == tuple(color)


def region_sha256(png_or_img, top: int) -> str:
    """sha256 of the raw RGB pixels below row `top` (the page region; arm-independent)."""
    img = _image(png_or_img)
    return hashlib.sha256(img.crop((0, top, img.width, img.height)).tobytes()).hexdigest()


def viewport_alignment(grab, viewport, top: int) -> float:
    """Fraction of viewport pixels that differ from the grab below `top` (0.0 = exact)."""
    import numpy as np

    g = np.asarray(_image(grab), dtype=np.int16)
    v = np.asarray(_image(viewport), dtype=np.int16)
    h, w = v.shape[:2]
    if top + h > g.shape[0] or w > g.shape[1]:
        return 1.0
    return float((np.abs(g[top:top + h, :w] - v).sum(axis=2) > 0).mean())


# ---------------------------------------------------------------- OCR checks (pure on text)

def _squash(s: str) -> str:
    return "".join(s.lower().translate(_DASHES).split())


def fuzzy_find(text: str, needle: str) -> float:
    """Best similarity (0..1) of `needle` to any window of `text`, ignoring case, whitespace and
    dash variants (tolerates single OCR character errors)."""
    t, n = _squash(text), _squash(needle)
    if not n:
        return 0.0
    if n in t:
        return 1.0
    best = 0.0
    for size in {max(1, len(n) - 1), len(n), len(n) + 1}:
        for i in range(max(1, len(t) - size + 1)):
            best = max(best, difflib.SequenceMatcher(None, t[i:i + size], n).ratio())
    return round(best, 3)


def omnibox_flags(ocr: str, url: str) -> dict:
    """How the page's host appears in the OCR'd chrome band; IDN display flags for IDN hosts.

    Chromium shows a mixed-script IDN label as punycode (xn--); an English OCR model reads a
    Unicode homoglyph host as its Latin skeleton, so `shows_unicode` tests that skeleton."""
    from originblind.oracle.attribute import to_unicode_host

    host = (urlsplit(url).hostname or "").lower()
    out: dict = {"host": host, "host_match": fuzzy_find(ocr, host) if host else 0.0}
    if any(label.startswith("xn--") for label in host.split(".")):
        puny = "xn--" in _squash(ocr) or out["host_match"] >= HOST_MATCH_MIN
        skeleton = to_unicode_host(host).translate(_SKELETON)
        out["shows_punycode"] = puny
        out["shows_unicode"] = not puny and fuzzy_find(ocr, skeleton) >= HOST_MATCH_MIN
    return out


def chrome_notices(ocr: str) -> list[str]:
    """Automation / insecure-connection notices legible in the chrome band (must be empty)."""
    return [n for n in (*AUTOMATION_NOTICES, INSECURE_NOTICE) if fuzzy_find(ocr, n) >= HOST_MATCH_MIN]


def ocr_text(img, box: tuple[int, int, int, int] | None = None, *, config: str = OCR_CONFIG) -> str:
    """Tesseract OCR of `box` (default: the whole image), grayscale and upscaled by OCR_SCALE.
    The default page segmentation (one uniform block) suits the chrome band; a region with several
    text blocks (the page) reads better with automatic segmentation (`config="--psm 3"`)."""
    import pytesseract
    from PIL import Image, ImageOps

    region = ImageOps.grayscale(img.crop(box) if box else img)
    region = region.resize((region.width * OCR_SCALE, region.height * OCR_SCALE), Image.LANCZOS)
    return pytesseract.image_to_string(region, config=config).strip()


# ---------------------------------------------------------------- window geometry

@dataclass(frozen=True)
class WindowMetrics:
    """Chromium's own view of its window (CSS px; == device px at devicePixelRatio 1)."""

    screen_x: int
    screen_y: int
    outer_w: int
    outer_h: int
    inner_w: int
    inner_h: int
    dpr: float

    @classmethod
    def of(cls, page) -> "WindowMetrics":
        d = page.evaluate(_WINDOW_JS)
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})

    @property
    def chrome_px(self) -> int:
        return self.outer_h - self.inner_h

    def problems(self, window: tuple[int, int]) -> list[str]:
        w, h = window
        out = []
        if (self.screen_x, self.screen_y) != (0, 0):
            out.append(f"window at ({self.screen_x},{self.screen_y}), expected (0,0)")
        if (self.outer_w, self.outer_h) != (w, h):
            out.append(f"window is {self.outer_w}x{self.outer_h}, expected {w}x{h}")
        if self.inner_w != self.outer_w:
            out.append(f"viewport width {self.inner_w} != window width {self.outer_w} (side frame)")
        if self.dpr != 1:
            out.append(f"devicePixelRatio {self.dpr} != 1")
        if not 0 < self.chrome_px < h // 2:
            out.append(f"implausible browser chrome height {self.chrome_px}px")
        return out


def x11_windows(display: str) -> list[tuple[int, int, int, int]]:
    """(x, y, w, h) of every visible top-level Chromium window, as the X server places it."""
    if shutil.which("xdotool") is None:
        raise RuntimeError("xdotool not found: cannot verify the browser window geometry (broken lab image)")
    env = {**os.environ, "DISPLAY": display}

    def run(*args: str) -> str:
        return subprocess.run(["xdotool", *args], capture_output=True, text=True, env=env, timeout=10).stdout

    out = []
    for wid in run("search", "--onlyvisible", "--class", "chromium").split():
        kv = dict(ln.split("=", 1) for ln in run("getwindowgeometry", "--shell", wid).splitlines() if "=" in ln)
        if {"X", "Y", "WIDTH", "HEIGHT"} <= kv.keys():
            out.append((int(kv["X"]), int(kv["Y"]), int(kv["WIDTH"]), int(kv["HEIGHT"])))
    return out


def grab_window(display: str, window: tuple[int, int]):
    """The window rectangle (0, 0, w, h) of the X display, as an RGB image (no cursor)."""
    import mss
    from PIL import Image

    with mss.MSS(display=display) as sct:
        shot = sct.grab({"left": 0, "top": 0, "width": window[0], "height": window[1]})
    return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")


# ---------------------------------------------------------------- the capture instrument

class _PageWatch:
    """Per-page state: the latest CDP visible security state, and whether the first-capture
    alignment check ran."""

    def __init__(self, page) -> None:
        self.security_state: str | None = None
        self.aligned = False
        try:
            self._cdp = page.context.new_cdp_session(page)
            self._cdp.on("Security.visibleSecurityStateChanged", self._on_security)
            self._cdp.send("Security.enable")  # emits the current state immediately
        except Exception:  # noqa: BLE001 - recorded as security_state None, checked by lab_check
            self._cdp = None

    def _on_security(self, ev: dict) -> None:
        self.security_state = (ev.get("visibleSecurityState") or {}).get("securityState")


class LabCapture:
    """Screen capture of the headful lab browser for S0/S1/S2 (one instance per worker)."""

    def __init__(self, window: tuple[int, int] = (1280, 960), chrome_px: int | None = None, *,
                 settle_s: float = 0.05) -> None:
        self.window = (int(window[0]), int(window[1]))
        self.chrome_px = chrome_px  # None: measured at the first capture, then held fixed
        self.settle_s = settle_s
        self._watches: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    def _check_window(self, page, display: str) -> int:
        """Verify the window geometry (Chromium's view and the X server's); return the number of
        visible browser windows (popups/bubbles are extra windows)."""
        m = WindowMetrics.of(page)
        problems = m.problems(self.window)
        wins = x11_windows(display)
        if (0, 0, *self.window) not in wins:
            problems.append(f"no visible X11 browser window at (0,0) sized {self.window[0]}x{self.window[1]} "
                            f"(visible: {wins})")
        if problems:
            raise RuntimeError("lab window geometry is wrong, refusing to capture: " + "; ".join(problems))
        if self.chrome_px is None:
            self.chrome_px = m.chrome_px
        elif m.chrome_px != self.chrome_px:
            raise RuntimeError(f"browser chrome height changed from {self.chrome_px}px to {m.chrome_px}px "
                               "(an infobar or toolbar appeared); S0/S1 would no longer be comparable")
        return len(wins)

    def capture(self, page, arm: str) -> tuple[bytes, dict]:
        if arm not in ARMS:
            raise ValueError(f"not a screenshot arm: {arm!r}")
        display = require_lab()
        watch = self._watches.get(page)
        if watch is None:
            watch = self._watches[page] = _PageWatch(page)
        n_windows = self._check_window(page, display)
        band = self.chrome_px
        page.evaluate(_PAINT_JS)
        time.sleep(self.settle_s)
        img = grab_window(display, self.window)
        meta: dict = {"arm": arm, "width": img.width, "height": img.height, "chrome_px": band,
                      "capture_mode": CAPTURE_MODE, "fallback": False, "url": page.url,
                      "security_state": watch.security_state, "x11_windows": n_windows}
        if not watch.aligned:  # first capture of this page: the band/page split must be exact
            diff = viewport_alignment(img, page.screenshot(type="png", caret="initial"), top=band)
            if diff > ALIGN_MAX_DIFF:
                raise RuntimeError(f"window grab does not match the page viewport below row {band} "
                                   f"({diff:.2%} of pixels differ)")
            watch.aligned = True
            meta["viewport_diff"] = round(diff, 5)
        if arm == "S0":
            img = _neutralize(img, band)
            meta["band_neutral"] = band_is_uniform(img, band)
        png = png_bytes(img)
        meta["png_sha256"] = hashlib.sha256(png).hexdigest()
        meta["page_sha256"] = region_sha256(img, band)
        ocr = ocr_text(img, (0, 0, img.width, band))
        meta["ocr_omnibox"] = ocr
        meta.update(omnibox_flags(ocr, page.url))
        meta["chrome_notices"] = chrome_notices(ocr)
        if arm == "S2":
            meta["ocr_page_bar"] = self._ocr_page_bar(page, img, band)
        return png, meta

    @staticmethod
    def _ocr_page_bar(page, img, band: int) -> str | None:
        """OCR of the in-page fake address bar (the twin's #ob-addressbar), if the page has one."""
        r = page.evaluate(_BAR_JS)
        if not r or r["w"] <= 0 or r["h"] <= 0:
            return None
        box = (max(0, int(r["x"])), band + max(0, int(r["y"])),
               min(img.width, int(r["x"] + r["w"])), min(img.height, band + int(r["y"] + r["h"])))
        if box[3] <= box[1] or box[2] <= box[0]:
            return None  # scrolled out of view
        return ocr_text(img, box)


# ---------------------------------------------------------------- NSS trust for the twin CA

def _certutil(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["certutil", *args], capture_output=True, text=True, timeout=30)


def _pem_blocks(text: str) -> list[bytes]:
    blocks, cur = [], None
    for ln in text.splitlines():
        if ln.startswith("-----BEGIN CERTIFICATE-----"):
            cur = [ln]
        elif cur is not None:
            cur.append(ln)
            if ln.startswith("-----END CERTIFICATE-----"):
                blocks.append(ssl.PEM_cert_to_DER_cert("\n".join(cur) + "\n"))
                cur = None
    return blocks


def trust_ca(ca_pem: str | Path, *, nssdb: str | Path | None = None, nickname: str = NSS_NICKNAME) -> None:
    """Trust the twin CA for TLS server auth in the user's NSS DB, which Chromium on Linux reads
    at startup (so it must run before the browser launches). Idempotent: an identical CA is left
    alone; an older CA under the same nickname is replaced, so exactly one twin CA is trusted."""
    if shutil.which("certutil") is None:
        raise RuntimeError("certutil (libnss3-tools) not found: the twin CA cannot be trusted")
    ca_pem = Path(ca_pem)
    want = _pem_blocks(ca_pem.read_text())
    if len(want) != 1:
        raise RuntimeError(f"{ca_pem} must hold exactly one PEM certificate")
    db = Path(nssdb) if nssdb else Path.home() / ".pki" / "nssdb"
    db.mkdir(parents=True, exist_ok=True)
    spec = f"sql:{db}"
    if not (db / "cert9.db").exists():
        r = _certutil("-d", spec, "-N", "--empty-password")
        if r.returncode != 0:
            raise RuntimeError(f"certutil -N failed: {r.stderr.strip()}")
    have = _pem_blocks(_certutil("-d", spec, "-L", "-n", nickname, "-a").stdout)
    if have == want and _trusted(spec, nickname):
        return
    for _ in range(len(have)):
        _certutil("-d", spec, "-D", "-n", nickname)
    r = _certutil("-d", spec, "-A", "-t", "C,,", "-n", nickname, "-i", str(ca_pem))
    if r.returncode != 0 or not _trusted(spec, nickname):
        raise RuntimeError(f"could not trust {ca_pem} in {db}: {r.stderr.strip()}")


def _trusted(spec: str, nickname: str) -> bool:
    for ln in _certutil("-d", spec, "-L").stdout.splitlines():
        parts = ln.rsplit(None, 1)
        if len(parts) == 2 and parts[0].strip() == nickname:
            return parts[1].startswith("C")
    return False
