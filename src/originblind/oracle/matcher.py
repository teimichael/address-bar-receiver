"""Canary matching under a fixed set of encodings.

A canary is a high-entropy token embedded in a secret value. The oracle searches every
captured request field (URL, header values, body bytes) for the canary under the
encodings a client might apply before transmission:

  - raw                 the token as-is
  - percent / URL       percent-encoding (upper and lower hex; "+" for space is handled
                        by also decoding fields before the raw search)
  - base64              std and url-safe, padded or not, at any byte alignment
  - json                JSON string escaping (\\uXXXX, escaped solidus)

Rather than search for each *encoded form* of the canary (which misses base64 at
arbitrary alignment), we additionally *decode* base64/percent/JSON runs found in the
field and search the decoded bytes for the raw token. This catches nesting and offset.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
from urllib.parse import unquote, unquote_plus

from originblind.types import CapturedRequest, Hit

_B64_RUN = re.compile(rb"[A-Za-z0-9+/_-]{16,}={0,2}")


def _raw_forms(token: str) -> dict[str, bytes]:
    t = token.encode("ascii")
    forms = {"raw": t}
    # percent-encoding of every byte, upper and lower hex
    forms["pct_upper"] = "".join(f"%{b:02X}" for b in t).encode("ascii")
    forms["pct_lower"] = "".join(f"%{b:02x}" for b in t).encode("ascii")
    # base64 of the token value itself (e.g. Authorization: Basic base64("user:TOKEN"))
    forms["b64_std"] = base64.b64encode(t)
    forms["b64_url"] = base64.urlsafe_b64encode(t)
    # json escaping only matters for non-alnum; our tokens are alnum so raw covers it,
    # but keep a json-escaped slash variant in case the token is embedded in a path-like value
    forms["json"] = json.dumps(token).encode("ascii")[1:-1]  # strip quotes
    return forms


def _decode_candidates(blob: bytes) -> list[bytes]:
    """Yield decoded views of a blob so a raw-token search can see through one encoding layer."""
    out: list[bytes] = [blob]
    # whitespace-stripped copy: catches CRLF/line-wrapped base64 (PEM/MIME 64-col) and
    # separator-grouped values; the base64 runs below then decode cleanly.
    stripped = re.sub(rb"\s+", b"", blob)
    if stripped != blob:
        out.append(stripped)
    # percent-decoding (both + and %xx)
    try:
        s = blob.decode("latin-1")
        out.append(unquote_plus(s).encode("latin-1"))
        out.append(unquote(s).encode("latin-1"))
    except Exception:
        pass
    # JSON unescaping, if it looks like a JSON string body
    try:
        txt = blob.decode("utf-8")
        if "\\u" in txt or "\\/" in txt:
            out.append(json.loads('"' + txt.replace('"', '\\"') + '"').encode("utf-8"))
    except Exception:
        pass
    # base64 runs, decoded at their natural alignment — scan both the raw blob and the
    # whitespace-stripped copy (so CRLF/line-wrapped base64 decodes as one run).
    scan_targets = [blob]
    if stripped != blob:
        scan_targets.append(stripped)
    for target in scan_targets:
        for m in _B64_RUN.finditer(target):
            run = m.group(0)
            for variant in (run, run + b"=" * (-len(run) % 4)):
                for altchars in (None, b"-_"):
                    try:
                        out.append(base64.b64decode(variant, altchars=altchars, validate=False))
                    except (binascii.Error, ValueError):
                        pass
    return out


def find_canary(token: str, req: CapturedRequest) -> list[Hit]:
    """Return every (encoding, location) at which `token` appears in a captured request."""
    hits: list[Hit] = []
    forms = _raw_forms(token)
    raw = forms["raw"]

    fields: list[tuple[str, bytes]] = [("url", req.url.encode("utf-8", "surrogatepass"))]
    for k, v in req.headers.items():
        fields.append((f"header:{k.lower()}", v.encode("utf-8", "surrogatepass")))
    if req.body:
        fields.append(("body", req.body))

    # digit-only tokens (card PANs) are often transmitted with separators (spaces/dashes);
    # prepare a digits-only form of the token to match a separator-stripped field.
    digit_token = raw if raw.isdigit() else None

    seen: set[tuple[str, str, str]] = set()
    for loc, blob in fields:
        # (a) direct search for each encoded form
        for enc, form in forms.items():
            if form in blob:
                key = (token, enc, loc)
                if key not in seen:
                    seen.add(key)
                    hits.append(Hit(canary_id=token, encoding=enc, location=loc))
        # (a2) digit canary with grouping separators removed
        if digit_token is not None and (token, "digits", loc) not in seen:
            digits_only = bytes(c for c in blob if 48 <= c <= 57)
            if digit_token in digits_only:
                seen.add((token, "digits", loc))
                hits.append(Hit(canary_id=token, encoding="digits", location=loc))
        # (b) decode one layer, then search for the raw token
        for dec in _decode_candidates(blob):
            if raw in dec and (token, "decoded", loc) not in seen:
                seen.add((token, "decoded", loc))
                hits.append(Hit(canary_id=token, encoding="decoded", location=loc))
    return hits
