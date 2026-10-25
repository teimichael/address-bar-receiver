"""H4 deployed-mechanism replica: browser-use sensitive_data fails closed on A1 (top-level is the
lookalike) but releases on A2 (top-level is the legit origin), because it binds only the top URL.
"""

from originblind.defense.deployed import decide_deployed, match_url_with_domain_pattern
from originblind.types import FieldContext

BOUND = "acme.com"


def ctx(top, frame, action):
    return FieldContext(top_url=top, frame_url=frame, form_action_url=action, top_csp=())


def test_matcher_exact_no_subdomain_no_idn():
    assert match_url_with_domain_pattern("https://acme.com/login", ["acme.com"])
    assert not match_url_with_domain_pattern("https://login.acme.com/x", ["acme.com"])  # no subdomain
    assert match_url_with_domain_pattern("https://a.acme.com/x", ["*.acme.com"])        # explicit wildcard
    # IDN lookalike of the bound domain is NOT normalized -> no match -> fail closed (good on A1)
    assert not match_url_with_domain_pattern("https://xn--acme-...example.com/x", ["acme.com"])


def test_fail_closed_on_a1():
    # top-level is the lookalike -> no substitution -> cannot leak
    v = decide_deployed(ctx("https://accme.com", "https://accme.com", "https://accme.com/s"), BOUND)
    assert not v.allow


def test_leaks_on_a2_iframe():
    # top-level is the legit origin, field is in a cross-origin iframe -> substitutes -> A2 leak
    v = decide_deployed(ctx("https://acme.com", "https://evil.com", "https://evil.com/s"), BOUND)
    assert v.allow and v.frame_domain == "evil.com"


def test_leaks_on_a2_action():
    v = decide_deployed(ctx("https://acme.com", "https://acme.com", "https://evil.com/s"), BOUND)
    assert v.allow and v.action_domain == "evil.com"


def test_releases_on_exact():
    v = decide_deployed(ctx("https://acme.com", "https://acme.com", "https://acme.com/s"), BOUND)
    assert v.allow
