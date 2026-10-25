"""Strict Content-Security-Policy parsing for delegation decisions.

The broker (D6) accepts a cross-origin frame or form target only when the top-level
document's *enforced* CSP explicitly declares it via `frame-src`/`form-action`. The
parsing rules are deliberately strict and pre-registered:

  - Only enforced `Content-Security-Policy` headers count. `-Report-Only` is ignored,
    and a `<meta>` CSP in the page body cannot widen delegation (it is never passed here).
  - `frame-src` falls back to `child-src` then `default-src`. `form-action` has NO
    fallback (a missing `form-action` does not inherit `default-src`).
  - Multiple CSP policies INTERSECT: a source must be allowed by every policy that constrains
    the directive. Several header lines, and a single header value whose policies were joined
    with commas (how HTTP stacks fold repeated headers), both yield multiple policies.
  - `'self'` is origin-level: the target must have the top-level document's host (and port);
    an https top also matches only https targets.
  - `*`, scheme-only sources (`https:`), and `data:`/`blob:` are NOT treated as a
    registrable-domain delegation (they cannot name a specific principal).
  - `'none'` allows nothing.
  - Host sources may be wildcarded on the left label (`https://*.psp.tld`), matching
    subdomains but not the apex.
"""

from __future__ import annotations

from dataclasses import dataclass

from originblind.oracle.attribute import _host_of, registrable_domain, to_ascii_host


@dataclass(frozen=True)
class _Directive:
    present: bool
    sources: tuple[str, ...]


def _split_policies(csp_headers: tuple[str, ...]) -> list[dict[str, tuple[str, ...]]]:
    policies = []
    serialized = [pol for header in csp_headers for pol in header.split(",")]
    for header in serialized:
        if not header.strip():
            continue
        d: dict[str, tuple[str, ...]] = {}
        for part in header.split(";"):
            toks = part.split()
            if not toks:
                continue
            name = toks[0].lower()
            # CSP: within one policy a repeated directive keeps the FIRST occurrence.
            d.setdefault(name, tuple(toks[1:]))
        policies.append(d)
    return policies


def _directive(policy: dict[str, tuple[str, ...]], name: str, *, fallback: list[str]) -> _Directive:
    if name in policy:
        return _Directive(True, policy[name])
    for fb in fallback:
        if fb in policy:
            return _Directive(True, policy[fb])
    return _Directive(False, ())


def _origin(url: str) -> tuple[str, str, int | None]:
    from urllib.parse import urlsplit

    v = url if "://" in url else "https://" + url
    sp = urlsplit(v)
    scheme = sp.scheme.lower()
    port = sp.port or {"https": 443, "http": 80}.get(scheme)
    return scheme, to_ascii_host(sp.hostname or "").rstrip(".").lower(), port


def _source_allows(source: str, top_url: str, target_host: str) -> bool:
    """Does a single CSP source expression allow `target_host` as a named principal?"""
    s = source.strip()
    low = s.lower()
    if low in ("'none'", "*"):
        return False  # '*' is not a specific-principal delegation; 'none' allows nothing
    if low == "'self'":
        ts, th, tp = _origin(top_url)
        gs, gh, gp = _origin(target_host)
        scheme_ok = gs == ts or (ts == "http" and gs == "https")
        return scheme_ok and gh == th and (gp == tp or (ts == "http" and gs == "https"))
    if low in ("https:", "http:", "data:", "blob:", "'unsafe-inline'", "'unsafe-eval'"):
        return False
    if low.startswith("'"):
        return False  # nonces/hashes/keywords are not host delegations
    # strip scheme and any path; if the source names a scheme, require it to match the target's
    host_expr = s
    if "://" in host_expr:
        src_scheme, host_expr = host_expr.split("://", 1)
        tgt_scheme = target_host.split("://", 1)[0].lower() if "://" in target_host else "https"
        if src_scheme.lower() != tgt_scheme:
            return False
    host_expr = host_expr.split("/", 1)[0]
    host_expr = host_expr.rsplit(":", 1)[0] if host_expr.count(":") == 1 else host_expr
    host_expr = host_expr.rstrip(".").lower()
    target = to_ascii_host(_host_of(target_host)).rstrip(".").lower()
    if host_expr.startswith("*."):
        suffix = host_expr[2:]
        return target.endswith("." + suffix)  # subdomain only, not the apex
    return target == host_expr


def directive_allows(
    csp_headers: tuple[str, ...], directive: str, target_host: str, top_url: str
) -> bool:
    """True iff every enforced policy's relevant directive allows `target_host`.

    directive: "frame-src" or "form-action".
    """
    policies = _split_policies(csp_headers)
    if not policies:
        return False
    fallback = {
        "frame-src": ["child-src", "default-src"],
        "form-action": [],  # no fallback, by spec
    }[directive]

    constraining = 0
    for policy in policies:
        d = _directive(policy, directive, fallback=fallback)
        if not d.present:
            continue  # this policy does not constrain the directive
        constraining += 1
        if not any(_source_allows(src, top_url, target_host) for src in d.sources):
            return False  # one policy forbids it → intersection forbids it
    # If no enforced policy constrains the directive, there is no explicit delegation.
    return constraining > 0
