from originblind.oracle.attribute import registrable_domain as rd
from originblind.oracle.attribute import same_registrable, to_ascii_host


def test_multilevel_suffix():
    assert rd("a.b.example.co.uk") == "example.co.uk"
    assert rd("https://shop.example.com:8443/x") == "example.com"


def test_ip_literal_attributes_to_itself():
    assert rd("203.0.113.7") == "203.0.113.7"
    assert rd("https://203.0.113.7:8443/login") == "203.0.113.7"
    assert rd("[2001:db8::1]") == "2001:db8::1"


def test_idn_unicode_equals_punycode():
    uni = "xаmple.com"  # cyrillic a
    asc = to_ascii_host(uni)
    assert asc.startswith("xn--")
    assert rd(uni) == rd(asc)


def test_trailing_dot_and_case():
    assert rd("Example.COM.") == "example.com"


def test_same_registrable():
    assert same_registrable("login.acme.com", "acme.com")
    assert not same_registrable("acme.com", "acme-login.com")


def test_single_label_host_fallback():
    # synthetic host with no public suffix should not explode
    assert rd("northwell") == "northwell"


def test_private_suffix_toggle():
    # with private domains, github.io is a suffix; without, it collapses to github.io's parent
    assert rd("user.github.io", include_psl_private=True) == "user.github.io"
    assert rd("user.github.io", include_psl_private=False) == "github.io"
