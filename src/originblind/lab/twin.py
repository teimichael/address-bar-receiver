"""The twin ASGI app: one per worker, reconfigured per episode.

It dispatches by Host header to a role (hub/landing/bound/attacker/psp/idp), renders the matching
page, emits the episode's CSP header on the landing document where required, and logs EVERY
request (method, full URL, host, headers, raw body) to an in-memory channel-A log. The log is the
authoritative oracle input; a catch-all route ensures exfiltration to any path or unknown host is
still captured.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, PlainTextResponse, Response
from starlette.routing import Route

from originblind.lab import pages
from originblind.lab.site import EpisodeSite
from originblind.types import CapturedRequest


@dataclass
class TwinState:
    site: EpisodeSite | None = None
    log: list[CapturedRequest] = field(default_factory=list)

    def reset(self, site: EpisodeSite) -> None:
        self.site = site
        self.log = []


def _host_of_request(req: Request) -> str:
    host = req.headers.get("host", "")
    if host.startswith("["):
        return host[1:host.find("]")].lower() if "]" in host else host.lower()
    return host.split(":", 1)[0].lower()


async def _capture(req: Request, state: TwinState) -> None:
    body = await req.body()
    state.log.append(
        CapturedRequest(
            channel="server",
            method=req.method,
            url=str(req.url),
            host=_host_of_request(req),
            headers={k: v for k, v in req.headers.items()},
            body=body,
            ts=time.time(),
            resource_type=None,
        )
    )


def _secret_supplied(site: EpisodeSite, body: bytes) -> bool:
    """Server-side required-field check, identical on every host (no differential feedback):
    a submission is accepted iff the flow's secret field is non-empty."""
    from urllib.parse import parse_qs

    field = pages.SECRET_FIELD.get(site.flow, "password")
    try:
        vals = parse_qs(body.decode("utf-8", "replace"), keep_blank_values=True).get(field, [])
    except Exception:  # noqa: BLE001
        return False
    return any(v.strip() for v in vals)


def make_app(state: TwinState) -> Starlette:
    async def handler(req: Request) -> Response:
        await _capture(req, state)
        site = state.site
        if site is None:
            return PlainTextResponse("no episode", status_code=503)
        host = _host_of_request(req)
        role = site.role_of(host)
        path = req.url.path

        if path == "/favicon.ico":
            return PlainTextResponse("", status_code=404)

        if role == "hub":
            return HTMLResponse(pages.render_hub(site))

        body = await req.body()  # cached by Starlette after _capture; never another request's body
        accepted = req.method == "POST" and _secret_supplied(site, body)

        if role in ("landing", "bound"):
            if req.method == "POST":
                if accepted:
                    return HTMLResponse(pages.render_success(site))
                return HTMLResponse(pages.render_flow(site, error=pages.missing_message(site)))
            # landing renders the condition (A2 page / A1 twin / exact); the bound apex in an A1
            # episode renders the legitimate same-origin page
            html = pages.render_landing(site) if role == "landing" else pages.render_flow(site)
            headers = {}
            if site.landing_csp and role == "landing":
                headers["Content-Security-Policy"] = "; ".join(site.landing_csp)
            return HTMLResponse(html, headers=headers)

        if role == "attacker":
            if req.method == "POST":  # mimics the real site exactly: no differential feedback
                if accepted:
                    return HTMLResponse(pages.render_success(site))
                if site.origin_class.value == "a2_iframe":
                    return HTMLResponse(pages.render_attacker_embed(site, error=pages.missing_message(site)))
                return HTMLResponse(pages.render_flow(site, error=pages.missing_message(site)))
            if path.startswith("/embed"):
                return HTMLResponse(pages.render_attacker_embed(site))
            return PlainTextResponse("ok")

        if role == "psp":
            if req.method == "POST":
                if accepted:
                    return HTMLResponse(pages.render_success(site))
                return HTMLResponse(pages.render_psp_pay(site, error=pages.missing_message(site)))
            if path == "/pay":
                return HTMLResponse(pages.render_psp_pay(site))
            return PlainTextResponse("ok")

        if role == "idp":
            return HTMLResponse(pages.render_flow(site))

        # Unknown host: still 200 so requests to arbitrary hosts are logged, not errored.
        return PlainTextResponse("ok")

    routes = [Route("/{path:path}", handler, methods=["GET", "POST", "PUT", "HEAD"])]
    app = Starlette(routes=routes)
    app.state.twin = state
    return app
