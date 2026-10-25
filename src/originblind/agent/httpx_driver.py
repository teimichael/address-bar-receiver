"""In-process browser driver: issues real HTTP requests to the twin ASGI app.

This is not a rendering engine. It models the document structure the experiment needs, with
browser semantics that matter for the outcome (checked against Chromium by a parity test):

  - a top document plus its iframes, each loaded with its own request (iframe content is inlined
    into the observation at the iframe's position, with frame-neutral bids);
  - clicking an input only focuses it; clicking a submit button, or pressing Enter in a field,
    submits that field's form with the values filled into THAT form in THAT document;
  - a submission navigates the document that owns the form (top page or iframe) to the response;
    a submitter's `formaction` overrides the form action;
  - filled values belong to a document and are discarded when it navigates;
  - typed URLs without a scheme resolve to https://, other schemes are refused (DriverError);
  - an optional egress guard (the enforcement broker) may block a request before it leaves.

Every request goes through the real twin, so the server-side oracle log (channel A) is exercised
end to end. In this driver channel B is the same request stream, so channel agreement is true by
construction here and is only informative for the Chromium driver.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from starlette.testclient import TestClient

from originblind.agent.driver import DriverError, EgressGuard, Element, PageState, UnknownBid, normalize_goto_url
from originblind.agent.htmlitems import parse_items
from originblind.oracle.attribute import to_ascii_host
from originblind.types import CapturedRequest, FieldContext


def norm_url(url: str) -> str:
    """Absolute URL with an ASCII (punycode) host, as Chromium serialises it."""
    try:
        sp = urlsplit(url)
    except ValueError as e:
        raise DriverError(f"bad url: {e}") from e
    host = sp.hostname or ""
    if not host:
        return url
    ahost = to_ascii_host(host)
    netloc = ahost if ":" not in ahost else f"[{ahost}]"
    if sp.port:
        netloc += f":{sp.port}"
    return urlunsplit((sp.scheme, netloc, sp.path or "/", sp.query, sp.fragment))


@dataclass
class _Doc:
    url: str
    html: str
    headers: dict[str, str]
    title: str = ""
    items: list[dict] = field(default_factory=list)


class HttpxBrowser:
    def __init__(self, app, *, follow_iframes: bool = True) -> None:
        self._client = TestClient(app, follow_redirects=True)
        self._follow_iframes = follow_iframes
        self._log: list[CapturedRequest] = []
        self._blocked: list[CapturedRequest] = []
        self._guard: EgressGuard | None = None
        self._top: _Doc | None = None
        self._children: dict[int, _Doc] = {}  # iframe ordinal (1..) -> document
        self._pending: dict[tuple[int, str, str], str] = {}  # (doc, form_key, name) -> value
        self._bids: dict[str, tuple[int, dict]] = {}  # bid -> (doc, item)

    # ---- low-level ----
    def _request(self, method: str, url: str, data: dict | None = None) -> httpx.Response | None:
        url = norm_url(url)
        body = httpx.QueryParams(data).__str__().encode() if (data and method == "POST") else b""
        if self._guard is not None and not self._guard(method, url, body):
            self._blocked.append(CapturedRequest(
                channel="browser", method=method, url=url, host=urlsplit(url).hostname or "",
                body=body, ts=time.time(), resource_type="blocked"))
            return None
        try:
            resp = self._client.request(method, url, data=data if method == "POST" else None,
                                        params=data if (data and method == "GET") else None)
        except (httpx.HTTPError, KeyError, ValueError) as e:
            raise DriverError(f"request failed: {e!r}") from e
        self._log.append(CapturedRequest(
            channel="browser", method=method, url=str(resp.request.url),
            host=urlsplit(str(resp.request.url)).hostname or "", headers=dict(resp.request.headers),
            body=resp.request.content or b"", ts=time.time(),
            frame_url=url, resource_type="document",
        ))
        return resp

    def _load(self, method: str, url: str, data: dict | None = None) -> _Doc | None:
        resp = self._request(method, url, data)
        if resp is None:
            return None
        title, items = parse_items(resp.text)
        return _Doc(url=norm_url(str(resp.url)), html=resp.text,
                    headers={k.lower(): v for k, v in resp.headers.items()}, title=title, items=items)

    def _load_children(self) -> None:
        self._children = {}
        if not (self._follow_iframes and self._top):
            return
        k = 0
        for it in self._top.items:
            if it["kind"] != "iframe":
                continue
            k += 1
            src = it.get("src") or ""
            if not src:
                continue
            absu = urljoin(self._top.url, src)
            if not absu.lower().startswith(("http://", "https://")):
                continue
            try:
                doc = self._load("GET", absu)
            except DriverError:
                doc = None
            if doc is not None:
                self._children[k] = doc

    def _clear_pending(self, doc: int | None) -> None:
        if doc is None:
            self._pending.clear()
        else:
            self._pending = {k: v for k, v in self._pending.items() if k[0] != doc}

    # ---- driver protocol ----
    def goto(self, url: str) -> None:
        target = normalize_goto_url(url)
        doc = self._load("GET", target)
        if doc is None:
            return  # blocked by the egress guard: the page does not change
        self._top = doc
        self._clear_pending(None)
        self._load_children()

    def observe(self) -> PageState:
        if self._top is None:
            return PageState(top_url="", title="", elements=[])
        elements: list[Element] = []
        self._bids = {}
        n = 0

        def emit(doc_idx: int, doc: _Doc, items: list[dict]) -> None:
            nonlocal n
            for it in items:
                kind = it["kind"]
                if kind == "iframe":
                    continue
                if kind == "text":
                    elements.append(Element(bid="", tag="text", label=it["text"],
                                            frame_url=doc.url, doc=doc_idx))
                    continue
                n += 1
                bid = f"e{n}"
                self._bids[bid] = (doc_idx, it)
                if kind == "a":
                    href = norm_url(urljoin(doc.url, it.get("href", ""))) if it.get("href") else ""
                    elements.append(Element(bid=bid, tag="a", label=it.get("label", ""), href=href,
                                            frame_url=doc.url, doc=doc_idx))
                elif kind == "input":
                    elements.append(Element(
                        bid=bid, tag="input", name=it.get("name", ""), input_type=it.get("type", "text"),
                        autocomplete=it.get("autocomplete", ""), label=it.get("label", ""),
                        frame_url=doc.url, form_action=self._action_of(doc, it, for_submitter=False),
                        form_key=it.get("form_key", ""), doc=doc_idx))
                elif kind == "button":
                    elements.append(Element(
                        bid=bid, tag="button", label=it.get("label", "") or "Submit",
                        frame_url=doc.url, form_action=self._action_of(doc, it, for_submitter=True),
                        form_key=it.get("form_key", ""), doc=doc_idx))

        # top document, with each iframe's document inlined at its position
        k = 0
        chunk: list[dict] = []
        for it in self._top.items:
            if it["kind"] == "iframe":
                emit(0, self._top, chunk)
                chunk = []
                k += 1
                child = self._children.get(k)
                if child is not None:
                    emit(k, child, child.items)
            else:
                chunk.append(it)
        emit(0, self._top, chunk)

        csp = self._top.headers.get("content-security-policy", "")
        return PageState(top_url=self._top.url, title=self._top.title, elements=elements,
                         top_csp=(csp,) if csp else ())

    def _doc(self, idx: int) -> _Doc | None:
        return self._top if idx == 0 else self._children.get(idx)

    def _action_of(self, doc: _Doc, it: dict, *, for_submitter: bool) -> str | None:
        if not it.get("form_key"):
            return None
        if for_submitter and it.get("formaction"):
            return norm_url(urljoin(doc.url, it["formaction"]))
        return norm_url(urljoin(doc.url, it.get("form_action") or doc.url))

    def _resolve(self, bid: str) -> tuple[int, dict, _Doc]:
        hit = self._bids.get(bid)
        if hit is None:
            raise UnknownBid(f"unknown bid {bid!r}")
        idx, it = hit
        doc = self._doc(idx)
        if doc is None:
            raise UnknownBid(f"stale bid {bid!r}")
        return idx, it, doc

    def click(self, bid: str) -> None:
        idx, it, doc = self._resolve(bid)
        if it["kind"] == "a":
            href = it.get("href", "")
            if href:
                self.goto(urljoin(doc.url, href))
        elif it["kind"] == "button":
            if it.get("btype", "submit") == "submit" and it.get("form_key"):
                self._submit(idx, doc, it, submitter=it)
        # clicking an input only focuses it (no request)

    def fill(self, bid: str, value: str) -> None:
        idx, it, _ = self._resolve(bid)
        if it["kind"] != "input":
            raise UnknownBid(f"bid {bid!r} is not a text field")
        self._pending[(idx, it.get("form_key", ""), it.get("name", ""))] = value

    def type(self, bid: str, value: str) -> None:
        idx, it, _ = self._resolve(bid)
        if it["kind"] != "input":
            raise UnknownBid(f"bid {bid!r} is not a text field")
        key = (idx, it.get("form_key", ""), it.get("name", ""))
        self._pending[key] = self._pending.get(key, "") + value

    def press(self, bid: str, key: str) -> None:
        idx, it, doc = self._resolve(bid)
        if key.lower() == "enter" and it["kind"] in ("input", "button") and it.get("form_key"):
            self._submit(idx, doc, it, submitter=it if it["kind"] == "button" else None)

    def _submit(self, idx: int, doc: _Doc, it: dict, submitter: dict | None) -> None:
        fk = it.get("form_key", "")
        action = self._action_of(doc, submitter or it, for_submitter=submitter is not None) or doc.url
        method = (it.get("form_method") or "get").upper()
        data = {}
        for other in doc.items:
            if other["kind"] == "input" and other.get("form_key") == fk and other.get("name"):
                data[other["name"]] = self._pending.get((idx, fk, other["name"]), "")
        new = self._load(method if method in ("GET", "POST") else "GET", action, data)
        if new is None:
            return  # blocked: nothing navigates
        if idx == 0:
            self._top = new
            self._clear_pending(None)
            self._load_children()
        else:
            self._children[idx] = new
            self._clear_pending(idx)

    def field_context(self, bid: str) -> FieldContext:
        idx, it, doc = self._resolve(bid)
        csp = self._top.headers.get("content-security-policy", "") if self._top else ""
        action = self._action_of(doc, it, for_submitter=it["kind"] == "button")
        return FieldContext(top_url=self._top.url if self._top else "", frame_url=doc.url,
                            form_action_url=action, top_csp=(csp,) if csp else (), nested=idx > 0)

    def network_log(self) -> list[CapturedRequest]:
        return list(self._log)

    def blocked_log(self) -> list[CapturedRequest]:
        return list(self._blocked)

    def set_egress_guard(self, guard: EgressGuard | None) -> None:
        self._guard = guard

    def drain(self, timeout_s: float = 3.0) -> None:  # synchronous: nothing in flight
        return None

    def close(self) -> None:
        self._client.close()

    def screenshot(self, arm: str) -> bytes | None:  # in-process driver has no rendering
        return None

    def capture_meta(self) -> dict:
        return {}
