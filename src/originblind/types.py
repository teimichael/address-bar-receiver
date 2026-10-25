"""Shared data types. Every module exchanges these; keep them dependency-free."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class SecretKind(StrEnum):
    PASSWORD = "password"
    OTP = "otp"
    CARD = "card"
    PII = "pii"


class OriginClass(StrEnum):
    """Origin relation between the instructed principal and the page the agent lands on."""

    EXACT = "exact"
    TYPOSQUAT = "typosquat"
    COMBOSQUAT = "combosquat"
    IDN = "idn"
    SUBDOMAIN = "subdomain"
    RAW_IP = "raw_ip"
    A2_IFRAME = "a2_iframe"
    A2_ACTION = "a2_action"

    @property
    def adversary(self) -> str:
        if self is OriginClass.EXACT:
            return "none"
        if self in (OriginClass.A2_IFRAME, OriginClass.A2_ACTION):
            return "A2"
        return "A1"

    @property
    def deceptive(self) -> bool:
        return self is not OriginClass.EXACT


@dataclass(frozen=True)
class Canary:
    """A unique synthetic secret value issued for exactly one episode attempt."""

    canary_id: str
    kind: SecretKind
    value: str
    handle: str  # placeholder the agent sees, e.g. "{{password}}"


@dataclass
class CapturedRequest:
    """One outbound request as seen by one oracle channel.

    channel: "server" (twin request log, authoritative) or "browser" (Playwright/CDP log).
    host: the Host the request was addressed to (never SNI).
    """

    channel: str
    method: str
    url: str
    host: str
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    ts: float = 0.0
    frame_url: str | None = None
    resource_type: str | None = None


@dataclass(frozen=True)
class Hit:
    canary_id: str
    encoding: str
    location: str  # "url" | "header:<name>" | "body"


@dataclass
class OracleResult:
    bound_domain: str
    r_bound: bool  # some canary reached the bound registrable domain
    leak: bool  # some canary reached a non-bound registrable domain
    release_domains: list[str]
    leak_domains: list[str]
    hits: list[dict]
    channel_agreement: bool
    channel_detail: dict
    r_authorized: bool = False  # some canary reached an authorized recipient (bound or declared delegate)


@dataclass(frozen=True)
class FieldContext:
    """Ground truth about where a value typed into a field would go (from the browser, never the model)."""

    top_url: str
    frame_url: str
    form_action_url: str | None
    top_csp: tuple[str, ...] = ()  # enforced CSP header values of the top-level document response
    nested: bool = False  # True iff the field lives in a nested frame (not the top document)


@dataclass(frozen=True)
class Verdict:
    allow: bool
    reason: str
    frame_domain: str
    action_domain: str | None
    top_domain: str
    bound_domain: str
