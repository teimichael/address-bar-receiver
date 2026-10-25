"""Normalised DOM hashing for the A1 byte-identity check and A2 render-parity diff.

For A1 twins we require the served HTML to be byte-identical across hostnames (guaranteed
by never interpolating the host). `normalized_hash` additionally strips any absolute URL to
the configured host set, so a post-load DOM comparison is robust to the browser resolving
relative refs against different origins.
"""

from __future__ import annotations

import hashlib
import re

_WS = re.compile(rb"\s+")


def raw_hash(html: bytes) -> str:
    return hashlib.sha256(html).hexdigest()


def normalized_hash(html: bytes, hosts: list[str]) -> str:
    """Hash after collapsing whitespace and replacing each known host with a placeholder,
    so A1 twins whose only difference is a resolved absolute URL hash equally."""
    b = html if isinstance(html, bytes) else html.encode()
    for h in sorted(hosts, key=len, reverse=True):
        b = b.replace(f"https://{h}".encode(), b"https://HOST")
        b = b.replace(h.encode(), b"HOST")
    b = _WS.sub(b" ", b).strip()
    return hashlib.sha256(b).hexdigest()
