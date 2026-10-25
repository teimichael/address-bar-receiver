"""Deterministic seed and canary derivation.

Everything random in an episode is derived by HMAC from a master seed and the episode's
identity, so a given (master_seed, config) reproduces byte-identical plans and canaries.
Canaries are attempt-scoped: a retry uses attempt+1 and therefore fresh canaries, so a
hit from a crashed attempt can never be attributed to its retry.
"""

from __future__ import annotations

import hashlib
import hmac

from originblind.types import Canary, SecretKind

_ALPHANUM = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789"  # no look-alikes
_HANDLES = {
    SecretKind.PASSWORD: "{{password}}",
    SecretKind.OTP: "{{otp}}",
    SecretKind.CARD: "{{card}}",
    SecretKind.PII: "{{pii}}",
}


def _prf(master_seed: bytes, *parts: str) -> bytes:
    msg = "\x1f".join(parts).encode("utf-8")
    return hmac.new(master_seed, msg, hashlib.sha256).digest()


def derive_seed(master_seed: bytes, *parts: str) -> int:
    return int.from_bytes(_prf(master_seed, "seed", *parts), "big")


def _b(digest: bytes, alphabet: str, n: int) -> str:
    x = int.from_bytes(digest, "big")
    out = []
    base = len(alphabet)
    for _ in range(n):
        x, r = divmod(x, base)
        out.append(alphabet[r])
    return "".join(out)


def _luhn_check_digit(num: str) -> str:
    s, alt = 0, True
    for ch in reversed(num):
        d = ord(ch) - 48
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        s += d
        alt = not alt
    return str((10 - (s % 10)) % 10)


def make_canary(master_seed: bytes, episode_id: str, attempt: int, kind: SecretKind) -> Canary:
    """Mint the unique secret value for one (episode, attempt, kind).

    The canary token `ob<24 alphanum>` carries >=96 bits over an alphanumeric alphabet, so
    URL- and JSON-encoding leave it unchanged; it is always a substring of `value`.
    """
    digest = _prf(master_seed, "canary", episode_id, str(attempt), kind.value)
    token = "ob" + _b(digest, _ALPHANUM, 24)
    canary_id = token

    if kind is SecretKind.PASSWORD:
        value = token  # the password is the token itself
    elif kind is SecretKind.OTP:
        # OTPs must stay matchable: embed the token; a 6-digit display code is derived
        # separately and is NOT the canary (digit runs are too collision-prone to match).
        value = token
    elif kind is SecretKind.CARD:
        # Luhn-valid 16-digit number under the 400000 test BIN used in payment-processor test
        # documentation (never a live card); 9 random digits make it unique within an episode's
        # traffic. Digits are matched after stripping separators.
        body = _b(_prf(master_seed, "card", episode_id, str(attempt)), "0123456789", 9)
        digits = "400000" + body
        value = digits + _luhn_check_digit(digits)
        canary_id = value  # the full PAN is the canary for cards
    elif kind is SecretKind.PII:
        value = token
    else:  # pragma: no cover
        raise ValueError(kind)

    return Canary(canary_id=canary_id, kind=kind, value=value, handle=_HANDLES[kind])


def display_otp(master_seed: bytes, episode_id: str, attempt: int) -> str:
    """A 6-digit code shown to the user for the OTP flow. Not used as a canary."""
    return _b(_prf(master_seed, "otp-display", episode_id, str(attempt)), "0123456789", 6)
