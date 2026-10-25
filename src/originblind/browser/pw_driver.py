"""Real-browser driver: Playwright Chromium behind the sinkhole proxy.

Implements the BrowserDriver protocol over a live page with the SAME observation semantics as
the in-process driver (one shared extractor definition in agent/htmlitems.py; a parity test
diffs the two): document-order items, an iframe's content inlined at its position, frame-neutral
bids. Ground truth for the broker/oracle comes from Chromium itself: the frame a field lives in,
and the form action read through the HTMLFormElement prototype getter (immune to DOM clobbering
such as <input name=action>); a submitter's `formaction` is honoured.

Channel B is the browser's request log (page 'request' events, which include out-of-process
iframes). When an egress guard is set (the D6 broker), every request is routed through it and a
blocked request never leaves the browser; blocked requests are kept in `blocked_log()` and
excluded from channel B.
"""

from __future__ import annotations

import time
from urllib.parse import urljoin, urlsplit

from originblind.agent.driver import DriverError, EgressGuard, Element, PageState, UnknownBid, normalize_goto_url
from originblind.agent.htmlitems import COLLECT_JS
from originblind.types import CapturedRequest, FieldContext

_ACTION_TIMEOUT_MS = 10_000
_NAV_TIMEOUT_MS = 30_000


class PlaywrightDriver:
    def __init__(self, page, *, capture=None) -> None:
        self.page = page
        self.context = page.context
        self._log: list[tuple[int, CapturedRequest]] = []
        self._blocked: list[CapturedRequest] = []
        self._blocked_ids: set[int] = set()
        self._guard: EgressGuard | None = None
        self._routed = False
        self._top_csp: tuple[str, ...] = ()
        self._bids: dict[str, tuple[object, str, dict, int]] = {}  # bid -> (frame, tok, item, doc)
        self._capture = capture  # browser.capture.LabCapture for S arms (headful lab only)
        self._last_capture: dict = {}
        page.on("request", self._on_request)
        page.on("response", self._on_response)

    # ---- network capture ----
    def _on_request(self, req) -> None:
        try:
            body = req.post_data_buffer or b""
        except Exception:  # noqa: BLE001
            body = b""
        try:
            headers = dict(req.headers)
        except Exception:  # noqa: BLE001
            headers = {}
        self._log.append((id(req), CapturedRequest(
            channel="browser", method=req.method, url=req.url,
            host=urlsplit(req.url).hostname or "", headers=headers,
            body=body if isinstance(body, bytes) else str(body).encode(), ts=time.time(),
            frame_url=(req.frame.url if req.frame else None), resource_type=req.resource_type,
        )))

    def _on_response(self, resp) -> None:
        try:
            if resp.request.resource_type == "document" and resp.frame == self.page.main_frame:
                csp = resp.headers.get("content-security-policy")
                # reset on every main-frame navigation so a prior page's CSP never leaks forward
                self._top_csp = (csp,) if csp else ()
        except Exception:  # noqa: BLE001
            pass

    def _route(self, route) -> None:
        req = route.request
        guard = self._guard
        if guard is not None:
            try:
                body = req.post_data_buffer or b""
            except Exception:  # noqa: BLE001
                body = b""
            if not guard(req.method, req.url, body if isinstance(body, bytes) else str(body).encode()):
                self._blocked_ids.add(id(req))
                self._blocked.append(CapturedRequest(
                    channel="browser", method=req.method, url=req.url, host=urlsplit(req.url).hostname or "",
                    body=body if isinstance(body, bytes) else b"", ts=time.time(), resource_type="blocked"))
                route.abort("blockedbyclient")
                return
        route.continue_()

    def set_egress_guard(self, guard: EgressGuard | None) -> None:
        self._guard = guard
        if guard is not None and not self._routed:
            self.context.route("**/*", self._route)
            self._routed = True

    def network_log(self) -> list[CapturedRequest]:
        return [cr for rid, cr in self._log if rid not in self._blocked_ids]

    def blocked_log(self) -> list[CapturedRequest]:
        return list(self._blocked)

    # ---- navigation helpers ----
    def _settle(self) -> None:
        try:
            self.page.wait_for_load_state("load", timeout=_NAV_TIMEOUT_MS)
        except Exception:  # noqa: BLE001
            pass
        self.page.wait_for_timeout(120)
        for fr in self.page.frames:
            if fr is not self.page.main_frame:
                try:
                    fr.wait_for_load_state("load", timeout=5000)
                except Exception:  # noqa: BLE001
                    pass

    def goto(self, url: str) -> None:
        target = normalize_goto_url(url)
        try:
            self.page.goto(target, wait_until="load", timeout=_NAV_TIMEOUT_MS)
        except Exception as e:  # noqa: BLE001 - net errors on an agent-typed URL are ITT outcomes
            raise DriverError(f"navigation failed: {str(e).splitlines()[0][:120]}") from e
        self._settle()

    def drain(self, timeout_s: float = 3.0) -> None:
        try:
            self.page.wait_for_load_state("networkidle", timeout=int(timeout_s * 1000))
        except Exception:  # noqa: BLE001
            pass

    # ---- observation ----
    def observe(self) -> PageState:
        self._settle()
        main = self.page.main_frame
        collected: dict[object, dict] = {}
        for i, fr in enumerate(self.page.frames):
            try:
                collected[fr] = fr.evaluate(COLLECT_JS, f"fr{i}")
            except Exception:  # noqa: BLE001 - detached/blocked frames contribute nothing
                collected[fr] = {"title": "", "items": []}
        # map iframe placeholders (by their data-ob-id token) to child frames
        child_by_tok: dict[str, object] = {}
        for fr in self.page.frames:
            if fr is main:
                continue
            try:
                el = fr.frame_element()
                tok = el.get_attribute("data-ob-id")
                if tok:
                    child_by_tok[tok] = fr
            except Exception:  # noqa: BLE001
                continue

        elements: list[Element] = []
        self._bids = {}
        n = 0
        doc_counter = {"next": 1}

        def emit(fr, doc_idx: int, depth: int) -> None:
            nonlocal n
            for it in collected.get(fr, {}).get("items", []):
                kind = it.get("kind")
                if kind == "iframe":
                    child = child_by_tok.get(it.get("tok", ""))
                    if child is not None and depth < 3:
                        idx = doc_counter["next"]
                        doc_counter["next"] += 1
                        emit(child, idx, depth + 1)
                    continue
                if kind == "text":
                    elements.append(Element(bid="", tag="text", label=it.get("text", ""),
                                            frame_url=fr.url, doc=doc_idx))
                    continue
                n += 1
                bid = f"e{n}"
                self._bids[bid] = (fr, it.get("tok", ""), it, doc_idx)
                if kind == "a":
                    elements.append(Element(bid=bid, tag="a", label=it.get("label", ""),
                                            href=it.get("href", ""), frame_url=fr.url, doc=doc_idx))
                elif kind == "input":
                    elements.append(Element(
                        bid=bid, tag="input", name=it.get("name", ""), input_type=it.get("type", "text"),
                        autocomplete=it.get("autocomplete", ""), label=it.get("label", ""),
                        frame_url=fr.url, form_action=self._action(fr, it, for_submitter=False),
                        form_key=it.get("form_key", ""), doc=doc_idx))
                elif kind == "button":
                    elements.append(Element(
                        bid=bid, tag="button", label=it.get("label", "") or "Submit", frame_url=fr.url,
                        form_action=self._action(fr, it, for_submitter=True),
                        form_key=it.get("form_key", ""), doc=doc_idx))

        emit(main, 0, 0)
        try:
            title = self.page.title()
        except Exception:  # noqa: BLE001
            title = collected.get(main, {}).get("title", "")
        return PageState(top_url=self.page.url, title=title, elements=elements, top_csp=self._top_csp)

    @staticmethod
    def _action(fr, it: dict, *, for_submitter: bool) -> str | None:
        if not it.get("form_key"):
            return None
        if for_submitter and it.get("formaction"):
            return urljoin(fr.url, it["formaction"])
        return it.get("form_action") or fr.url

    # ---- actions ----
    def _locator(self, bid: str):
        hit = self._bids.get(bid)
        if hit is None:
            raise UnknownBid(f"unknown bid {bid!r}")
        fr, tok, it, _ = hit
        return fr.locator(f'[data-ob-id="{tok}"]'), it

    def click(self, bid: str) -> None:
        loc, _ = self._locator(bid)
        try:
            loc.click(timeout=_ACTION_TIMEOUT_MS)
        except Exception as e:  # noqa: BLE001
            raise DriverError(f"click failed: {str(e).splitlines()[0][:120]}") from e
        self._settle()

    def fill(self, bid: str, value: str) -> None:
        loc, it = self._locator(bid)
        if it.get("kind") != "input":
            raise UnknownBid(f"bid {bid!r} is not a text field")
        try:
            loc.fill(value, timeout=_ACTION_TIMEOUT_MS)
        except Exception as e:  # noqa: BLE001
            raise DriverError(f"fill failed: {str(e).splitlines()[0][:120]}") from e

    def type(self, bid: str, value: str) -> None:
        loc, it = self._locator(bid)
        if it.get("kind") != "input":
            raise UnknownBid(f"bid {bid!r} is not a text field")
        try:
            loc.press_sequentially(value, timeout=_ACTION_TIMEOUT_MS)
        except Exception as e:  # noqa: BLE001
            raise DriverError(f"type failed: {str(e).splitlines()[0][:120]}") from e

    def press(self, bid: str, key: str) -> None:
        loc, _ = self._locator(bid)
        try:
            loc.press(key, timeout=_ACTION_TIMEOUT_MS)
        except Exception as e:  # noqa: BLE001
            raise DriverError(f"press failed: {str(e).splitlines()[0][:120]}") from e
        self._settle()

    def field_context(self, bid: str) -> FieldContext:
        hit = self._bids.get(bid)
        if hit is None:
            raise UnknownBid(f"unknown bid {bid!r}")
        fr, _, it, doc = hit
        action = self._action(fr, it, for_submitter=it.get("kind") == "button")
        return FieldContext(top_url=self.page.url, frame_url=fr.url, form_action_url=action,
                            top_csp=self._top_csp, nested=doc > 0)

    def close(self) -> None:
        pass

    # ---- screenshot arms (lab only) ----
    def screenshot(self, arm: str) -> bytes | None:
        if self._capture is None:
            return None
        png, meta = self._capture.capture(self.page, arm)
        self._last_capture = meta
        return png

    def capture_meta(self) -> dict:
        return dict(self._last_capture)

    def episode_meta(self) -> dict:
        try:
            return {"chromium": self.context.browser.version if self.context.browser else ""}
        except Exception:  # noqa: BLE001
            return {}
