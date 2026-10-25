"""Faithful replica of browser-use's shipped `sensitive_data` domain scoping (the deployed D5).

Verified against browser-use source (registry/service.py, utils.py) during design:
  - substitution is gated on the AGENT'S FOCUSED-TAB (top-level) URL, never the field's frame
    origin or form `action`;
  - `cross_origin_iframes` defaults to False;
  - `match_url_with_domain_pattern` defaults the scheme to https, matches bare domains exactly
    (no implicit subdomain match), fails closed on error, and does NO IDN/punycode normalization.

So the mechanism is a top-level-binding broker (our D5) with these matching quirks. We encode it
as a policy over FieldContext to measure H4: it withholds the secret on A1 (top-level is the
lookalike -> no match) but releases on A2 (top-level is the legit origin -> match -> the secret
flows into the cross-origin iframe / off-origin form). The real library can be swapped in for the
LLM-driven slice; this replica is the deterministic, dependency-free artifact.
"""

from __future__ import annotations

from urllib.parse import urlsplit

from originblind.types import FieldContext, Verdict


def _host(url: str) -> str:
    v = url if "://" in url else "https://" + url
    return (urlsplit(v).hostname or "").lower().rstrip(".")


def match_url_with_domain_pattern(url: str, allowed: list[str]) -> bool:
    """Replica of browser-use's matcher: exact host match (no subdomain, no IDN normalization),
    https default, fail-closed. A pattern may be a bare domain or scheme://domain."""
    try:
        host = _host(url)
        if not host:
            return False
        for pat in allowed:
            phost = _host(pat)
            if "*" in phost:
                # browser-use supports a single leading wildcard label
                if phost.startswith("*.") and host.endswith("." + phost[2:]):
                    return True
                continue
            if host == phost:  # exact, no IDN normalization
                return True
        return False
    except Exception:
        return False  # fail closed


def decide_deployed(ctx: FieldContext, bound_domain: str, allowed: list[str] | None = None) -> Verdict:
    """D5-like, using the browser-use matcher on the TOP-LEVEL url only."""
    allowed = allowed or [bound_domain]
    allow = match_url_with_domain_pattern(ctx.top_url, allowed)
    from originblind.oracle.attribute import registrable_domain

    return Verdict(
        allow=allow,
        reason="bu-top-level-match" if allow else "bu-no-match",
        frame_domain=registrable_domain(ctx.frame_url),
        action_domain=(registrable_domain(ctx.form_action_url) if ctx.form_action_url else None),
        top_domain=registrable_domain(ctx.top_url),
        bound_domain=bound_domain,
    )
