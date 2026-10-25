"""Secret handles: the agent's input is identical across defense arms.

The agent always sees and emits a handle (e.g. "{{password}}"), never the real secret. The
harness substitutes the real canary value at fill time:

  - D0..D4: substitution is unconditional (the broker is absent or advisory).
  - D5/D6: substitution is gated on the broker verdict for the field's ground-truth context;
    a denied fill leaves the field empty and records a broker refusal.

Because the agent's tokens never contain the secret, no canary can appear in any prompt — a
property asserted in tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from originblind.types import Canary

# A missing final brace ("{{card}") is accepted as the same handle (amendment v2.2): Ministral-3-14B
# writes it in 21% of its secret fills (62% of card, 73% of OTP fills; results/pilot_v2r_l), and no other
# model ever did in any pilot (0 of ~13,000 fills). Strictly, such a fill typed the literal text, so
# its card/OTP outcomes measured a typing slip instead of the origin decision. Unknown handle names
# stay verbatim. Every accepted slip is counted (telemetry `handle_repairs`).
_HANDLE_RE = re.compile(r"\{\{\s*(password|otp|card|pii(?::[a-z_]+)?)\s*\}(\})?")


@dataclass
class HandleRegistry:
    """Maps a handle string to the issued canary for this episode/attempt."""

    by_handle: dict[str, Canary]

    @classmethod
    def from_canaries(cls, canaries: list[Canary]) -> "HandleRegistry":
        return cls({c.handle: c for c in canaries})

    def contains_handle(self, text: str) -> bool:
        return bool(_HANDLE_RE.search(text or ""))

    def count_repaired(self, text: str) -> int:
        """Known handles in `text` written without their final brace ("{{card}")."""
        return sum(1 for m in _HANDLE_RE.finditer(text or "") if m.group(2) is None)

    def resolve(self, text: str) -> tuple[str, list[Canary]]:
        """Replace every known handle in `text` with its canary value.

        Returns (substituted_text, canaries_used). Unknown handle names are left verbatim (and
        therefore never leak a secret); a known handle missing its final brace is substituted.
        """
        used: list[Canary] = []

        def _sub(m: re.Match) -> str:
            key = "{{" + m.group(1).split(":", 1)[0] + "}}"
            # exact-handle match first (covers {{pii}}); typed PII sub-fields fall back to {{pii}}
            c = self.by_handle.get("{{" + m.group(1) + "}}") or self.by_handle.get(key)
            if c is None:
                return m.group(0)
            used.append(c)
            return c.value

        return _HANDLE_RE.sub(_sub, text or ""), used
