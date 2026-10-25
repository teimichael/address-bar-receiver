"""Registrable-domain attribution via an offline Public Suffix List snapshot.

All attribution uses the request Host (or a URL's host), never SNI. IDN hosts are
normalised so a Unicode label and its punycode form map to the same key. IP literals
attribute to themselves.
"""

from __future__ import annotations

import functools
import ipaddress
from urllib.parse import urlsplit

import idna
import tldextract

# tldextract's bundled PSL snapshot; suffix_list_urls=() forces fully offline operation. The
# snapshot version is recorded in every run manifest (psl_fingerprint below).
_DEFAULT_CACHE = None

# Reserved lab TLDs (RFC 2606/6761) are not on the PSL. Without treating them as public
# suffixes every lab infrastructure host (mail.webmail.test, pay.pspworld.test,
# id.ssoprovider.test) would collapse into one "registrable domain" ("test") and a legitimate
# PSP payment would be scored as a leak. They are suffixes by default everywhere.
LAB_SUFFIXES: tuple[str, ...] = ("test", "example")


@functools.lru_cache(maxsize=4)
def _extractor(extra_suffixes: tuple[str, ...], include_psl_private: bool) -> tldextract.TLDExtract:
    return tldextract.TLDExtract(
        suffix_list_urls=(),  # never hit the network
        cache_dir=_DEFAULT_CACHE,
        extra_suffixes=list(extra_suffixes),
        include_psl_private_domains=include_psl_private,
    )


def _host_of(value: str) -> str:
    """Accept a bare host or a full URL; return the lowercased host with no port/brackets."""
    v = value.strip()
    if "://" in v:
        host = urlsplit(v).hostname or ""
    else:
        # strip a possible port; keep bracketed IPv6 intact first
        if v.startswith("["):
            host = v[1 : v.find("]")] if "]" in v else v
        else:
            host = v.split(":", 1)[0] if v.count(":") == 1 else v
    return host.rstrip(".").lower()


def _as_ip(host: str) -> str | None:
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        return None


def to_ascii_host(host: str) -> str:
    """Normalise an IDN host to its ASCII (punycode) form; passthrough on failure."""
    if not host or _as_ip(host):
        return host
    try:
        return idna.encode(host, uts46=True).decode("ascii")
    except idna.IDNAError:
        try:
            return host.encode("idna").decode("ascii")
        except Exception:
            return host


def registrable_domain(
    value: str,
    *,
    extra_suffixes: tuple[str, ...] = LAB_SUFFIXES,
    include_psl_private: bool = False,
) -> str:
    """Return the registrable domain (eTLD+1) of a host or URL.

    IP literals (v4/v6) attribute to the canonical IP string. IDN hosts attribute by
    their ASCII form, so Unicode and punycode spellings collapse to one key.
    """
    host = _host_of(value)
    if not host:
        return ""
    ip = _as_ip(host)
    if ip is not None:
        return ip
    ascii_host = to_ascii_host(host)
    ext = _extractor(extra_suffixes, include_psl_private)
    res = ext(ascii_host)
    if res.domain and res.suffix:
        return f"{res.domain}.{res.suffix}"
    # No known public suffix (e.g. a synthetic single-label host): fall back to the host.
    return res.domain or ascii_host


def same_registrable(a: str, b: str, **kw) -> bool:
    return registrable_domain(a, **kw) == registrable_domain(b, **kw)


def to_unicode_host(host: str) -> str:
    """Render a host the way a Unicode-displaying surface would (punycode -> Unicode labels)."""
    if not host or _as_ip(host):
        return host
    try:
        return idna.decode(host)
    except (idna.IDNAError, UnicodeError):
        return host


def psl_fingerprint() -> dict:
    """Identify the PSL data actually used for attribution (for the provenance manifest)."""
    import hashlib

    ext = _extractor(LAB_SUFFIXES, False)
    try:
        tl = ext.tlds  # a property (list) in tldextract 5.x
        suffixes = sorted(tl() if callable(tl) else tl)
    except Exception:  # pragma: no cover - tldextract API drift
        suffixes = []
    return {
        "tldextract": getattr(tldextract, "__version__", "unknown"),
        "n_suffixes": len(suffixes),
        "sha256": hashlib.sha256("\n".join(suffixes).encode()).hexdigest(),
        "lab_suffixes": list(LAB_SUFFIXES),
    }
