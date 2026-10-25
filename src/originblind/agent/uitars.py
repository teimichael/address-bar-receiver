"""UI-TARS-1.5 action adapter.

UI-TARS emits coordinate actions in the Qwen2.5-VL resized-image space (dimensions rounded to a
multiple of 28 by `smart_resize`). To act in the real viewport we invert that scaling. We parse
the common action vocabulary into the harness grammar; a bid is synthesised by nearest-element
hit-testing against the observed PageState element boxes (provided by the caller).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

_CLICK = re.compile(r"(?:click|left_click|tap)\s*\(\s*(?:start_box|point|box)?\s*=?\s*[\['\"(]*\s*(\d+)\s*,\s*(\d+)", re.I)
_TYPE = re.compile(r"type\s*\(\s*(?:content|text)?\s*=?\s*['\"](.+?)['\"]\s*\)", re.I | re.S)
_SCROLL = re.compile(r"scroll\s*\(", re.I)
_FINISH = re.compile(r"\b(finished|stop|done|terminate)\b", re.I)


def smart_resize(h: int, w: int, factor: int = 28, min_pixels: int = 56 * 56,
                 max_pixels: int = 1280 * 800) -> tuple[int, int]:
    """Qwen2.5-VL smart_resize: round to a multiple of `factor` within the pixel budget."""
    hbar = max(factor, round(h / factor) * factor)
    wbar = max(factor, round(w / factor) * factor)
    if hbar * wbar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        hbar = max(factor, math.floor(h / beta / factor) * factor)
        wbar = max(factor, math.floor(w / beta / factor) * factor)
    elif hbar * wbar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hbar = math.ceil(h * beta / factor) * factor
        wbar = math.ceil(w * beta / factor) * factor
    return hbar, wbar


def resized_to_viewport(x: int, y: int, vp_w: int, vp_h: int) -> tuple[int, int]:
    rh, rw = smart_resize(vp_h, vp_w)
    return round(x * vp_w / rw), round(y * vp_h / rh)


@dataclass
class UITarsAction:
    kind: str  # click | type | scroll | stop | unknown
    x: int | None = None
    y: int | None = None
    text: str | None = None


def parse_uitars(output: str) -> UITarsAction:
    m = _CLICK.search(output)
    if m:
        return UITarsAction("click", int(m.group(1)), int(m.group(2)))
    m = _TYPE.search(output)
    if m:
        return UITarsAction("type", text=m.group(1))
    if _SCROLL.search(output):
        return UITarsAction("scroll")
    if _FINISH.search(output):
        return UITarsAction("stop")
    return UITarsAction("unknown")


def nearest_bid(x: int, y: int, boxes: list[tuple[str, float, float, float, float]]) -> str | None:
    """boxes: (bid, left, top, right, bottom) in viewport px. Return the bid whose box contains
    (x,y), else the nearest box centre."""
    best, best_d = None, float("inf")
    for bid, l, t, r, b in boxes:
        if l <= x <= r and t <= y <= b:
            return bid
        cx, cy = (l + r) / 2, (t + b) / 2
        d = (cx - x) ** 2 + (cy - y) ** 2
        if d < best_d:
            best, best_d = bid, d
    return best
