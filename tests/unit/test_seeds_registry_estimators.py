import math

from originblind.analysis.estimators import (
    clopper_pearson,
    h1_decision,
    holm,
    newcombe_diff,
    osi_point,
    wilson,
    zero_event_upper,
)
from originblind.lab.registry import BRANDS, all_hosts, bound_domain_for, host_for
from originblind.oracle.attribute import registrable_domain
from originblind.seeds import make_canary
from originblind.types import OriginClass, SecretKind

SEED = b"\x01" * 16


# ---- seeds / canaries ----

def test_canary_determinism_and_attempt_scope():
    a = make_canary(SEED, "ep1", 0, SecretKind.PASSWORD)
    b = make_canary(SEED, "ep1", 0, SecretKind.PASSWORD)
    c = make_canary(SEED, "ep1", 1, SecretKind.PASSWORD)
    assert a == b
    assert a.value != c.value  # retry gets a fresh canary
    assert a.canary_id in a.value


def test_canary_uniqueness_across_plan():
    ids = set()
    for ep in range(500):
        for kind in SecretKind:
            cv = make_canary(SEED, f"ep{ep}", 0, kind)
            assert cv.canary_id not in ids
            ids.add(cv.canary_id)


def test_card_is_luhn_valid():
    cv = make_canary(SEED, "ep1", 0, SecretKind.CARD)
    digits = [int(x) for x in cv.value]
    s, alt = 0, False
    for d in reversed(digits):
        d = d * 2 - 9 if alt and d * 2 > 9 else (d * 2 if alt else d)
        s += d
        alt = not alt
    assert s % 10 == 0


# ---- registry ----

def test_host_classes_structural():
    for brand in BRANDS[:4]:
        assert host_for(brand, OriginClass.EXACT, SEED).bound
        typo = host_for(brand, OriginClass.TYPOSQUAT, SEED).host
        assert typo.endswith(".com") and typo != f"{brand}.com"
        idn = host_for(brand, OriginClass.IDN, SEED)
        assert idn.ascii_host.startswith("xn--")
        sub = host_for(brand, OriginClass.SUBDOMAIN, SEED)
        assert registrable_domain(sub.host) != bound_domain_for(brand)
        ip = host_for(brand, OriginClass.RAW_IP, SEED).host
        assert ip.startswith("203.0.113.")


def test_no_lookalike_equals_bound():
    for h in all_hosts(SEED):
        if h.origin_class is not OriginClass.EXACT and h.origin_class.adversary == "A1":
            assert registrable_domain(h.ascii_host) != bound_domain_for(h.brand)


# ---- estimators ----

def test_wilson_cp_known():
    w = wilson(5, 20)
    assert math.isclose(w.point, 0.25)
    assert math.isclose(w.lo, 0.1119, abs_tol=1e-3) and math.isclose(w.hi, 0.4687, abs_tol=1e-3)
    cp = clopper_pearson(0, 120)
    assert cp.lo == 0.0 and math.isclose(cp.hi, zero_event_upper(120), rel_tol=1e-6)
    assert cp.hi < 0.031  # the doc's ~3.0% at n=120


def test_newcombe_sign():
    d = newcombe_diff(80, 100, 20, 100)
    assert math.isclose(d.point, 0.6) and d.lo > 0


def test_osi_and_decision():
    # blind: leak equals base -> OSI 0
    assert math.isclose(osi_point(60, 100, 60, 100), 0.0)
    # perfect: no leak -> OSI 1
    assert math.isclose(osi_point(0, 100, 60, 100), 1.0)
    assert h1_decision(0.1, 0.0, 0.3) == "supported"
    assert h1_decision(0.7, 0.5, 0.8) == "falsified"
    assert h1_decision(0.4, 0.2, 0.6) == "inconclusive"


def test_holm():
    out = holm({"a": 0.001, "b": 0.04, "c": 0.5})
    assert out["a"] and not out["c"]
