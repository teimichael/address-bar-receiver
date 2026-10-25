import itertools

from originblind.defense.csp import directive_allows
from originblind.defense.policy import decide_d5, decide_d6
from originblind.types import FieldContext

BOUND = "acme.com"


def ctx(top, frame, action, csp=(), nested=None):
    # a field whose frame differs from the top document lives in a nested frame
    if nested is None:
        nested = frame.split("://", 1)[-1].split("/", 1)[0] != top.split("://", 1)[-1].split("/", 1)[0]
    return FieldContext(top_url=top, frame_url=frame, form_action_url=action, top_csp=tuple(csp),
                        nested=nested)


# ---- CSP parsing ----

def test_form_action_no_fallback():
    csp = ("default-src 'self'; frame-src https://psp.com",)
    # form-action not declared -> no delegation even though default-src is self
    assert not directive_allows(csp, "form-action", "https://psp.com/pay", "https://acme.com")


def test_frame_src_falls_back_to_default():
    csp = ("default-src https://psp.com",)
    assert directive_allows(csp, "frame-src", "https://psp.com/f", "https://acme.com")


def test_self_resolves_to_top():
    csp = ("form-action 'self'",)
    assert directive_allows(csp, "form-action", "https://acme.com/submit", "https://acme.com")
    assert not directive_allows(csp, "form-action", "https://evil.com/s", "https://acme.com")


def test_wildcard_subdomain_not_apex():
    csp = ("frame-src https://*.psp.com",)
    assert directive_allows(csp, "frame-src", "https://a.psp.com/x", "https://acme.com")
    assert not directive_allows(csp, "frame-src", "https://psp.com/x", "https://acme.com")


def test_star_and_scheme_not_delegation():
    assert not directive_allows(("frame-src *",), "frame-src", "https://evil.com", "https://acme.com")
    assert not directive_allows(("frame-src https:",), "frame-src", "https://evil.com", "https://acme.com")


def test_multiple_headers_intersect():
    csp = ("frame-src https://psp.com https://x.com", "frame-src https://psp.com")
    assert directive_allows(csp, "frame-src", "https://psp.com/f", "https://acme.com")
    assert not directive_allows(csp, "frame-src", "https://x.com/f", "https://acme.com")


# ---- broker truth tables ----

def test_d5_top_level_only():
    # exact top, cross-origin iframe field: D5 allows (blind to frame)
    v = decide_d5(ctx("https://acme.com", "https://evil.com", "https://evil.com/s"), BOUND)
    assert v.allow
    # lookalike top: deny
    assert not decide_d5(ctx("https://acme-login.com", "https://acme-login.com", None), BOUND).allow


def test_d6_blocks_a2_iframe_and_action():
    assert not decide_d6(ctx("https://acme.com", "https://evil.com", "https://evil.com/s"), BOUND).allow
    assert not decide_d6(ctx("https://acme.com", "https://acme.com", "https://evil.com/s"), BOUND).allow
    assert decide_d6(ctx("https://acme.com", "https://acme.com", "https://acme.com/s"), BOUND).allow


def test_d6_allows_csp_declared_psp():
    csp = ("frame-src https://psp.com; form-action https://psp.com",)
    v = decide_d6(ctx("https://acme.com", "https://psp.com", "https://psp.com/pay", csp), BOUND)
    assert v.allow and v.reason == "csp-delegated"


def test_d6_denies_undeclared_psp():
    assert not decide_d6(ctx("https://acme.com", "https://psp.com", "https://psp.com/pay"), BOUND).allow


def test_property_declaration_never_turns_allow_into_deny():
    universe = ["acme.com", "evil.com", "psp.com"]
    for top, frame, action in itertools.product(universe, repeat=3):
        base = decide_d6(ctx(f"https://{top}", f"https://{frame}", f"https://{action}"), BOUND)
        csp = (f"frame-src https://{frame}; form-action https://{action}",)
        withdecl = decide_d6(ctx(f"https://{top}", f"https://{frame}", f"https://{action}", csp), BOUND)
        if base.allow:
            assert withdecl.allow  # adding a declaration never removes an allow


# ---- regressions for the audit findings ----

def test_d6_no_delegation_from_unbound_top_c3():
    """An A1 lookalike cannot self-declare its way into an allow (audit C3)."""
    csp = ("frame-src 'self'; form-action 'self'",)
    v = decide_d6(ctx("https://northwel.com/login", "https://northwel.com/login",
                      "https://northwel.com/submit", csp), "northwell.com")
    assert not v.allow
    csp2 = ("frame-src https://northwel.com; form-action https://northwel.com",)
    assert not decide_d6(ctx("https://northwel.com/login", "https://northwel.com/login",
                             "https://northwel.com/submit", csp2), "northwell.com").allow


def test_d6_frame_src_never_governs_top_document_c3():
    csp = ("frame-src https://evil.com; form-action https://acme.com",)
    # a top-level field on an unbound page is never allowed via frame-src
    assert not decide_d6(ctx("https://evil.com/x", "https://evil.com/x", "https://acme.com/s", csp,
                             nested=False), BOUND).allow


def test_csp_comma_folded_policies_intersect_c12():
    folded = ("frame-src https://psp.com, frame-src https://evil.com",)
    assert not directive_allows(folded, "frame-src", "https://evil.com/f", "https://acme.com")
    assert not directive_allows(folded, "frame-src", "https://psp.com/f", "https://acme.com")


def test_csp_self_is_origin_level_c12():
    assert directive_allows(("form-action 'self'",), "form-action", "https://acme.com/s", "https://acme.com/x")
    assert not directive_allows(("form-action 'self'",), "form-action", "https://ugc.acme.com/s", "https://acme.com")
    assert not directive_allows(("form-action 'self'",), "form-action", "http://acme.com/s", "https://acme.com")


def test_egress_destination_check():
    from originblind.defense.policy import destination_allowed

    csp = ("frame-src https://pay.psp.com; form-action https://pay.psp.com",)
    assert destination_allowed("D6", "https://acme.com/submit", "https://acme.com/", (), BOUND)
    assert not destination_allowed("D6", "https://evil.com/submit", "https://acme.com/", (), BOUND)
    assert destination_allowed("D6", "https://pay.psp.com/charge", "https://acme.com/", csp, BOUND)
    # delegation is honoured only from a bound top document
    assert not destination_allowed("D6", "https://pay.psp.com/charge", "https://acme-login.com/", csp, BOUND)
    assert destination_allowed("D5", "https://evil.com/submit", "https://acme.com/", (), BOUND)
