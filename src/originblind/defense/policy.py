"""Broker release policies as pure functions of ground-truth field context.

Two policies, both deciding whether a secret bound to `bound_domain` may be released
into a given field:

  D5 (top-level binding): allow iff the registrable domain of the *top-level* document equals
      the bound domain. Blind to frame boundaries and form targets by design (the policy shape of
      browser-use's shipped `sensitive_data` scoping and of top-level-only fillers).

  D6 (frame + form-action binding with CSP-declared delegation): allow iff
      - the field's frame is bound: for a field in the top document, the top document itself must
        be on the bound domain (CSP never governs the top document); for a field in a nested
        frame, the frame is on the bound domain or is declared by the bound top document's
        enforced CSP `frame-src`;
      - AND the form-action target is the bound domain or is declared by the bound top
        document's enforced CSP `form-action`.
      Delegation is honoured ONLY when the top-level document is itself on the bound domain, so
      an A1 lookalike cannot self-declare (`form-action 'self'`) its way into an allow.

Ground truth (top URL, frame URL, form action, CSP, nesting) comes from the browser driver,
never from the model. These functions are deterministic and exhaustively unit-tested. At egress,
`destination_allowed` re-checks the actual request destination (see agent/loop.py).
"""

from __future__ import annotations

from originblind.defense.csp import directive_allows
from originblind.oracle.attribute import registrable_domain
from originblind.types import FieldContext, Verdict


def _reg(url: str | None) -> str | None:
    if not url:
        return None
    return registrable_domain(url)


def decide_d5(ctx: FieldContext, bound_domain: str) -> Verdict:
    top = _reg(ctx.top_url) or ""
    frame = _reg(ctx.frame_url) or ""
    action = _reg(ctx.form_action_url)
    allow = top == bound_domain
    return Verdict(
        allow=allow,
        reason="top-level-bound" if allow else "top-level-mismatch",
        frame_domain=frame,
        action_domain=action,
        top_domain=top,
        bound_domain=bound_domain,
    )


def decide_d6(ctx: FieldContext, bound_domain: str) -> Verdict:
    top = _reg(ctx.top_url) or ""
    frame = _reg(ctx.frame_url) or ""
    action = _reg(ctx.form_action_url)
    top_bound = top == bound_domain

    if not ctx.nested:
        frame_ok = top_bound and frame == bound_domain
    else:
        frame_ok = frame == bound_domain or (
            top_bound and bool(ctx.top_csp)
            and directive_allows(ctx.top_csp, "frame-src", ctx.frame_url, ctx.top_url)
        )
    if action is None:
        action_ok = True  # no form submission target (e.g. JS-handled field); frame governs
    else:
        action_ok = action == bound_domain or (
            top_bound and bool(ctx.top_csp) and ctx.form_action_url is not None
            and directive_allows(ctx.top_csp, "form-action", ctx.form_action_url, ctx.top_url)
        )

    allow = frame_ok and action_ok
    if allow:
        reason = "bound" if (frame == bound_domain and action in (None, bound_domain)) else "csp-delegated"
    elif not frame_ok:
        reason = "frame-origin-unbound"
    else:
        reason = "form-action-unbound"
    return Verdict(
        allow=allow,
        reason=reason,
        frame_domain=frame,
        action_domain=action,
        top_domain=top,
        bound_domain=bound_domain,
    )


def destination_allowed(policy: str, dest_url: str, top_url: str, top_csp: tuple[str, ...],
                        bound_domain: str) -> bool:
    """Egress re-check for a request carrying a released secret.

    D6: the destination must be the bound domain, or declared by the bound top document's CSP
    (`form-action` or `frame-src`, the two ways a declared delegate legitimately receives data).
    D5/DBU: top-level binding has no destination check (that is exactly its blind spot).
    """
    if policy != "D6":
        return True
    dest = _reg(dest_url)
    if dest == bound_domain:
        return True
    if _reg(top_url) != bound_domain or not top_csp:
        return False
    return (directive_allows(top_csp, "form-action", dest_url, top_url)
            or directive_allows(top_csp, "frame-src", dest_url, top_url))


POLICIES = {"D5": decide_d5, "D6": decide_d6}


def decide(policy: str, ctx: FieldContext, bound_domain: str) -> Verdict:
    return POLICIES[policy](ctx, bound_domain)
