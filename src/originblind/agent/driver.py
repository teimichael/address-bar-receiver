"""Browser-driver protocol shared by the in-process (httpx/ASGI) driver and the real Playwright
driver.

The agent loop, observation serializers, defenses and oracle all talk to this protocol, so the
same episode logic runs with or without a real browser. Both drivers must produce the SAME
PageState for the same page (a parity test enforces it):

  - elements come in document order; an iframe's content is inlined at the iframe's position;
  - interactive elements (input/button/a) get frame-neutral bids e1, e2, ... in that order, so a
    bid never reveals which frame an element lives in;
  - visible text blocks are elements with tag "text" and an empty bid;
  - `doc` (0 = top document, 1.. = nested frames) and `frame_url`/`form_action` are ground truth
    for the broker, the oracle and the T2 annotations; T0/T1 never print them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from originblind.types import CapturedRequest, FieldContext


class DriverError(Exception):
    """A navigation/action the page cannot perform (bad URL, unsupported scheme, timeout).

    Raised for failures caused by the agent's own action, which the loop records as an
    intent-to-treat outcome (a step note), never as an infrastructure failure.
    """


class UnknownBid(DriverError):
    """The agent referenced a bid that is not in the current observation."""


@dataclass
class Element:
    bid: str
    tag: str  # input | button | a | text
    name: str = ""
    input_type: str = ""
    label: str = ""  # visible label (inputs), own text (buttons/links), text content (text)
    autocomplete: str = ""
    href: str = ""
    frame_url: str = ""  # URL of the document (frame) the element lives in
    form_action: str | None = None  # effective submission URL (a submitter's formaction wins)
    form_key: str = ""  # identifies the element's form within its document
    doc: int = 0  # 0 = top document, >=1 = nested frame (never shown to the agent)


@dataclass
class PageState:
    top_url: str
    title: str
    elements: list[Element] = field(default_factory=list)
    top_csp: tuple[str, ...] = ()

    def by_bid(self, bid: str) -> Element | None:
        if not bid:
            return None
        for e in self.elements:
            if e.bid == bid:
                return e
        return None

    @property
    def interactive(self) -> list[Element]:
        return [e for e in self.elements if e.bid]


# guard(method, url, body) -> True to allow the request to leave the browser, False to block it.
EgressGuard = Callable[[str, str, bytes], bool]


class BrowserDriver(Protocol):
    def goto(self, url: str) -> None: ...
    def observe(self) -> PageState: ...
    def click(self, bid: str) -> None: ...
    def fill(self, bid: str, value: str) -> None: ...
    def type(self, bid: str, value: str) -> None: ...
    def press(self, bid: str, key: str) -> None: ...
    def field_context(self, bid: str) -> FieldContext: ...
    def network_log(self) -> list[CapturedRequest]: ...  # channel B
    def blocked_log(self) -> list[CapturedRequest]: ...  # requests stopped by the egress guard
    def set_egress_guard(self, guard: EgressGuard | None) -> None: ...
    def drain(self, timeout_s: float = 3.0) -> None: ...  # let in-flight requests finish
    def close(self) -> None: ...
    def screenshot(self, arm: str) -> bytes | None: ...  # PNG for S arms, None for text-only
    def capture_meta(self) -> dict: ...  # per-capture metadata for S arms (dims, OCR, mode)


def normalize_goto_url(url: str) -> str:
    """Resolve what a browser omnibox would do with a typed URL: add https:// to a bare host,
    reject non-web schemes (raised as DriverError so it is an ITT outcome)."""
    u = (url or "").strip().strip("'\"")
    if not u:
        raise DriverError("empty url")
    low = u.lower()
    if "://" not in low:
        if low.startswith(("javascript:", "data:", "file:", "about:", "chrome:", "mailto:")):
            raise DriverError(f"unsupported scheme in {u[:40]!r}")
        return "https://" + u.lstrip("/")
    scheme = low.split("://", 1)[0]
    if scheme not in ("http", "https"):
        raise DriverError(f"unsupported scheme {scheme!r}")
    return u
