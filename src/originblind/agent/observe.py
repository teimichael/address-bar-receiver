"""Observation serializers.

Text arms (what a text agent receives):
  T0   page text + elements with bids, NO URL anywhere (no URL line, no link targets).
  T1   T0 + the top-level document URL and link targets, hosts in ASCII/punycode form, as
       Chromium serialises URLs (the framework default).
  T1u  T1 with every host rendered in Unicode (IDN representation manipulation; identical to T1
       for ASCII hosts).
  T2   T1 + per-element frame-origin and form-action registrable-domain annotations.

Screenshot arms (S0/S1/S2) send an image; their text channel is exactly T0, so S0 vs T0 differ
only by the image and S1 vs S0 only by the browser chrome in the image.

Elements arrive in document order with an iframe's content inlined at its position and
frame-neutral bids (see agent/driver.py), so T0/T1 carry no frame information at all; that is
the framework default this study measures (RQ1), and T2 is the counterfactual that exposes it.
"""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from originblind.agent.driver import Element, PageState
from originblind.oracle.attribute import registrable_domain, to_unicode_host

TEXT_ARMS = ("T0", "T1", "T1u", "T2")
SCREEN_ARMS = ("S0", "S1", "S2")


def text_arm(arm: str) -> str:
    """The text channel used by an arm (S arms use T0)."""
    return "T0" if arm in SCREEN_ARMS else arm


def render_url(url: str, unicode_hosts: bool) -> str:
    if not unicode_hosts or not url:
        return url
    try:
        sp = urlsplit(url)
    except ValueError:
        return url
    host = sp.hostname or ""
    if not host:
        return url
    uhost = to_unicode_host(host)
    netloc = uhost + (f":{sp.port}" if sp.port else "")
    return urlunsplit((sp.scheme, netloc, sp.path, sp.query, sp.fragment))


def _el_line(e: Element, *, show_urls: bool, unicode_hosts: bool, annotate: bool) -> str:
    if e.tag == "input":
        base = f"[{e.bid}] input name={e.name!r} type={e.input_type} label={e.label!r}"
    elif e.tag == "a":
        base = f"[{e.bid}] link {e.label!r}"
        if show_urls and e.href:
            base += f" -> {render_url(e.href, unicode_hosts)}"
    elif e.tag == "button":
        base = f"[{e.bid}] button {e.label!r}"
    else:
        base = f"[{e.bid}] {e.tag} {e.label!r}"
    if annotate:
        ann = []
        if e.frame_url:
            ann.append(f"frame-origin={registrable_domain(e.frame_url)}")
        if e.form_action:
            ann.append(f"form-action={registrable_domain(e.form_action)}")
        if ann:
            base += "  (" + ", ".join(ann) + ")"
    return base


def serialize(page: PageState, arm: str, bound_domain: str = "") -> str:
    arm = text_arm(arm)
    show_urls = arm in ("T1", "T1u", "T2")
    unicode_hosts = arm == "T1u"
    annotate = arm == "T2"
    lines: list[str] = []
    if show_urls:
        lines.append(f"URL: {render_url(page.top_url, unicode_hosts)}")
    lines.append(f"TITLE: {page.title}")
    lines.append("PAGE:")
    for e in page.elements:
        if e.tag == "text":
            lines.append(f"  {e.label}")
        else:
            lines.append("  " + _el_line(e, show_urls=show_urls, unicode_hosts=unicode_hosts,
                                         annotate=annotate))
    return "\n".join(lines)
