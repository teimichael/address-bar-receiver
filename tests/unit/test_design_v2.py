"""Design-level guards added after the October-2026 audit (see README.md).

Each test pins a property whose violation was found in the pilot design: incoherent secret/flow
cells, brand imbalance, lexical cues in sink names, the A2 sink being shown in T0/T1, probe
parsing of Unicode answers, and ITT handling of agent-caused navigation errors.
"""

import collections

import pytest

from originblind.agent.driver import DriverError, normalize_goto_url
from originblind.agent.probe import extract_domain, extract_yesno, probe_key, seed_for
from originblind.config import ExperimentSpec, ModelSpec
from originblind.defense.defenses import Defense
from originblind.lab.registry import BRANDS, LOADED_TOKENS, all_hosts, bound_domain_for, host_for
from originblind.oracle.attribute import registrable_domain
from originblind.run.plan import expand
from originblind.types import OriginClass

SEED = bytes.fromhex("0b1d0f0f0b1d0f0f0b1d0f0f0b1d0f0f")
TEXT = ModelSpec(key="m", hf_repo="x/m", served_name="m", pipeline="text")
VLM = ModelSpec(key="v", hf_repo="x/v", served_name="v", pipeline="both")


def _spec(**kw):
    base = dict(name="t", master_seed_hex=SEED.hex(), models=[TEXT], origin_classes=list(OriginClass),
                flows=["login"], observation_arms=["T1"], defenses=["D0"], replicates_per_brand=1)
    base.update(kw)
    return ExperimentSpec(**base)


# ---------------- registry ----------------

def test_no_lexical_giveaway_in_any_generated_host():
    for h in all_hosts(SEED):
        assert not any(t in h.ascii_host for t in LOADED_TOKENS), h


@pytest.mark.parametrize("oc", list(OriginClass))
def test_every_class_has_twelve_distinct_registrable_domains(oc):
    regs = {registrable_domain(host_for(b, oc, SEED).ascii_host) for b in BRANDS}
    assert len(regs) == 12


def test_typosquat_is_one_edit_and_never_the_brand():
    def lev(a, b):
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

    for b in BRANDS:
        t = host_for(b, OriginClass.TYPOSQUAT, SEED).host[:-4]
        assert t != b
        # an adjacent transposition is one Damerau edit (two Levenshtein edits)
        assert lev(t, b) in (1, 2)


def test_sinks_are_off_origin_and_not_bound():
    for b in BRANDS:
        for oc in (OriginClass.A2_IFRAME, OriginClass.A2_ACTION, OriginClass.SUBDOMAIN):
            assert registrable_domain(host_for(b, oc, SEED).host) != bound_domain_for(b)


# ---------------- config / plan ----------------

def test_incoherent_secret_flow_rejected_in_legacy_mode():
    with pytest.raises(ValueError):
        ExperimentSpec(name="x", models=[TEXT], origin_classes=["exact"], flows=["login"],
                       secrets=["otp"], replicates=2)


def test_balanced_plan_uses_every_brand_equally_and_coherent_secrets():
    spec = _spec(flows=["login", "otp", "checkout"], exact_oversample=2)
    items = expand(spec)
    per_brand = collections.Counter(it.brand for it in items)
    assert set(per_brand) == set(BRANDS) and len(set(per_brand.values())) == 1
    pairs = {(it.flow, it.secret) for it in items}
    assert pairs == {("login", "password"), ("otp", "otp"), ("checkout", "card")}
    exact = sum(1 for it in items if it.origin_class == "exact")
    typo = sum(1 for it in items if it.origin_class == "typosquat")
    assert exact == 2 * typo


def test_arm_origin_restriction_and_screen_arms_only_for_vision_models():
    spec = _spec(models=[TEXT, VLM], observation_arms=["T1", "T1u", "S1"],
                 arm_origins={"T1u": ["exact", "idn"]}, driver="chromium-headful")
    items = expand(spec)
    assert {it.origin_class for it in items if it.arm == "T1u"} == {"exact", "idn"}
    assert not any(it.arm == "S1" and it.model_key == "m" for it in items)
    assert any(it.arm == "S1" and it.model_key == "v" for it in items)


def test_screen_arms_require_the_lab_driver():
    with pytest.raises(ValueError):
        _spec(models=[VLM], observation_arms=["S1"], driver="chromium")


def test_unknown_defense_rejected():
    with pytest.raises(ValueError):
        _spec(defenses=["D3-push"])
    with pytest.raises(ValueError):
        Defense("d6")


# ---------------- probe parsing ----------------

def test_extract_domain_handles_unicode_examples_and_noise():
    assert extract_domain("e.g. northwell.com") == "northwell.com"
    assert extract_domain("nοrthwell.com") == "xn--nrthwell-zdg.com"  # Greek omicron, IDNA-normalised
    assert extract_domain("The value goes to **formsworks.net**.") == "formsworks.net"
    assert extract_domain("I cannot tell") is None
    assert extract_yesno("NO.") == "no" and extract_yesno("Yes, it will") == "yes" and extract_yesno("maybe") is None


def test_probe_key_ignores_defense_and_seed_differs_by_replicate():
    k0 = probe_key("m", "northwell", "login", "idn", "neutral", "T2", 0)
    k1 = probe_key("m", "northwell", "login", "idn", "neutral", "T2", 1)
    assert k0 != k1 and seed_for(k0) != seed_for(k1)
    assert probe_key("m", "b", "login", "exact", "neutral", "S1", 0).split("::")[5] == "T0"


# ---------------- ITT: agent-caused navigation errors are outcomes ----------------

def test_goto_normalisation():
    assert normalize_goto_url("northwell.com/login") == "https://northwell.com/login"
    for bad in ("ftp://x", "javascript:alert(1)", "file:///etc/passwd", ""):
        with pytest.raises(DriverError):
            normalize_goto_url(bad)


def test_lab_screen_matches_entrypoint_and_per_worker_display_is_opt_in(monkeypatch):
    """Private worker displays must reproduce the entrypoint's X screen exactly (pixel parity), and
    stay off unless the lab runner opts in."""
    import re
    from pathlib import Path

    from originblind.browser import run_browser as rb

    ep = (Path(__file__).resolve().parents[2] / "docker" / "entrypoint.sh").read_text()
    assert re.search(r"^SCREEN=(\S+)$", ep, flags=re.M).group(1) == rb.LAB_SCREEN
    assert "-dpi 96" in ep
    monkeypatch.delenv(rb.PER_WORKER_DISPLAY_ENV, raising=False)
    assert not rb.per_worker_displays()
    monkeypatch.setenv(rb.PER_WORKER_DISPLAY_ENV, "1")
    assert rb.per_worker_displays()
