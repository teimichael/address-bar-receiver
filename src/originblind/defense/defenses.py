"""Defense arms as compositions of five hooks, so contrasts differ in exactly one factor.

  reminder   the binding instruction in the system prompt (prompts.REMINDER)
  annotate   force T2 per-element frame-origin / form-action annotations
  tool       offer the advisory check_origin(bid) action
  push       write the field facts ("facts") or facts + ALLOW/DENY ("verdict") into the
             observation whenever a secret field is present
  gate       enforcement broker on secret fills and at egress ("D5" | "D6" | "DBU")

  D0      none (the task names the official site but states no binding rule)
  D1      reminder                       (instruction baseline for every model-level arm)
  D2      reminder + annotate            (information, inline)
  D3      reminder + tool                (information on request: invocation)
  D3info  reminder + push=facts          (information pushed, no advice)
  D3push  reminder + push=verdict        (information + advice pushed: compliance)
  D5r     reminder + gate=D5             (top-level enforcement)
  D6r     reminder + gate=D6             (frame/action enforcement)
  D5, D6  gate only (legacy arms without the reminder)
  DBU     gate=replica of browser-use's top-level `sensitive_data` scoping (legacy; it is
          decision-equivalent to D5 on every lab context — the real library is measured in
          audit/browser_use instead)

The verdict shown by D3/D3push and enforced by D6 is the SAME function (policy.decide("D6")),
so "information vs enforcement" holds identical information across arms.
"""

from __future__ import annotations

from dataclasses import dataclass

from originblind.agent.driver import PageState
from originblind.agent.observe import serialize
from originblind.defense.policy import decide
from originblind.types import FieldContext, Verdict


@dataclass(frozen=True)
class DefenseSpec:
    reminder: bool = False
    annotate: bool = False
    tool: bool = False
    push: str | None = None  # None | "facts" | "verdict"
    gate: str | None = None  # None | "D5" | "D6" | "DBU"


SPECS: dict[str, DefenseSpec] = {
    "D0": DefenseSpec(),
    "D1": DefenseSpec(reminder=True),
    "D2": DefenseSpec(reminder=True, annotate=True),
    "D3": DefenseSpec(reminder=True, tool=True),
    "D3info": DefenseSpec(reminder=True, push="facts"),
    "D3push": DefenseSpec(reminder=True, push="verdict"),
    "D5": DefenseSpec(gate="D5"),
    "D6": DefenseSpec(gate="D6"),
    "D5r": DefenseSpec(reminder=True, gate="D5"),
    "D6r": DefenseSpec(reminder=True, gate="D6"),
    "DBU": DefenseSpec(gate="DBU"),
}
DEFENSE_IDS = tuple(SPECS)


def broker_verdict(policy: str, ctx: FieldContext, bound_domain: str) -> Verdict:
    return decide(policy, ctx, bound_domain)


@dataclass
class Defense:
    defense_id: str

    def __post_init__(self) -> None:
        if self.defense_id not in SPECS:
            raise ValueError(f"unknown defense id {self.defense_id!r}; known: {', '.join(SPECS)}")

    @property
    def spec(self) -> DefenseSpec:
        return SPECS[self.defense_id]

    @property
    def policy(self) -> str | None:
        return self.spec.gate

    @property
    def exposes_check_origin(self) -> bool:
        return self.spec.tool

    @property
    def reminder(self) -> bool:
        return self.spec.reminder

    def observation_arm(self, base_arm: str) -> str:
        if self.spec.annotate and base_arm in ("T0", "T1", "T1u", "T2"):
            return "T2"
        return base_arm

    def render_observation(self, page: PageState, base_arm: str, bound_domain: str,
                           secret_bids: list[str]) -> str:
        obs = serialize(page, self.observation_arm(base_arm), bound_domain)
        if self.spec.push and secret_bids:
            title = "ORIGIN CHECK (automatic):" if self.spec.push == "verdict" else "ORIGIN FACTS (automatic):"
            lines = ["", title]
            for bid in secret_bids:
                el = page.by_bid(bid)
                if el is None:
                    continue
                ctx = FieldContext(top_url=page.top_url, frame_url=el.frame_url,
                                   form_action_url=el.form_action, top_csp=page.top_csp,
                                   nested=el.doc > 0)
                v = decide("D6", ctx, bound_domain)
                line = (f"  [{bid}] frame={v.frame_domain} action={v.action_domain} "
                        f"task-site={bound_domain}")
                if self.spec.push == "verdict":
                    line += f" -> {'ALLOW' if v.allow else 'DENY'}"
                lines.append(line)
            obs += "\n" + "\n".join(lines)
        return obs

    def gate(self, ctx: FieldContext, bound_domain: str) -> Verdict | None:
        if self.spec.gate == "DBU":
            from originblind.defense.deployed import decide_deployed

            return decide_deployed(ctx, bound_domain)
        if self.spec.gate is None:
            return None
        return decide(self.spec.gate, ctx, bound_domain)
