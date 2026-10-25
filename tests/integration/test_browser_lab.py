"""Real-browser integration: Chromium through the sinkhole proxy against the TLS twin.

Validates the evidence path the in-process tests cannot: real rendering, cross-origin iframe
DOM access, CDP/request network capture, A1 DOM-hash identity under a real engine, and that the
oracle + broker behave identically to the in-process path with a real browser.

Host-side this runs headless with --ignore-certificate-errors (the secure-omnibox / NSS-trust
and Xvfb/screenshot checks belong to the Docker lab). Marked `browser`; auto-skips without
Chromium.
"""

import hashlib

import pytest
from playwright.sync_api import sync_playwright

from originblind.agent.loop import run_episode
from originblind.browser.pw_driver import PlaywrightDriver
from originblind.browser.server import TwinServer
from originblind.controls import always_fill
from originblind.defense.defenses import Defense
from originblind.lab.pki import write_pki
from originblind.lab.registry import host_for
from originblind.lab.twin import TwinState
from originblind.llm.client import Sampling
from originblind.llm.mock import ScriptedBackend
from originblind.oracle.reconcile import reconcile
from originblind.run.episode import build_site
from originblind.types import OriginClass, SecretKind

pytestmark = pytest.mark.browser

SEED = b"\x05" * 16


def _all_episode_hosts(brand="acme"):
    hosts = ["mail.webmail.test", f"{brand}.com"]
    for oc in OriginClass:
        h = host_for(brand, oc, SEED)
        hosts.append(h.ascii_host)
    hosts += ["pay.pspworld.test", "id.ssoprovider.test"]
    return hosts


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    d = tmp_path_factory.mktemp("pki")
    hosts = _all_episode_hosts()
    paths = write_pki(d, hosts, wildcards=["*.pspworld.test", "*.ssoprovider.test"])
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--ignore-certificate-errors"])
        yield {"pki": paths, "browser": browser}
        browser.close()


def _episode(lab, oc, defense_id="D0"):
    site, canaries = build_site(SEED, brand="acme", flow="login", oc=oc,
                               secret=SecretKind.PASSWORD, semantics="neutral", replicate=0)
    state = TwinState(); state.reset(site)
    server = TwinServer(state, lab["pki"]["crt"], lab["pki"]["key"]).start()
    ctx = lab["browser"].new_context(proxy={"server": server.proxy_url}, ignore_https_errors=True)
    page = ctx.new_page()
    driver = PlaywrightDriver(page)
    try:
        tr = run_episode(driver, ScriptedBackend(always_fill()), "mock", site, Defense(defense_id),
                         canaries, "T1", Sampling(), step_budget=10)
        driver.drain()
        res = reconcile(canaries, site.bound_domain, state.log, driver.network_log(),
                        authorized=site.authorized_domains)
    finally:
        ctx.close(); server.stop()
    return tr, res, state


def test_real_browser_exact_release(lab):
    _, res, _ = _episode(lab, OriginClass.EXACT)
    assert res.r_bound and not res.leak


@pytest.mark.parametrize("oc", [OriginClass.TYPOSQUAT, OriginClass.IDN, OriginClass.A2_ACTION, OriginClass.A2_IFRAME])
def test_real_browser_leaks_on_deceptive(lab, oc):
    _, res, _ = _episode(lab, oc)
    assert res.leak, f"real browser must leak on {oc}"


def test_real_browser_d6_blocks_a2_iframe(lab):
    tr, res, _ = _episode(lab, OriginClass.A2_IFRAME, defense_id="D6")
    assert not res.leak and tr.broker_denials >= 1


def test_a1_dom_hash_identity_real_render(lab):
    """The login DOM, as rendered by Chromium, is identical across A1 hosts."""
    hashes = set()
    for oc in (OriginClass.EXACT, OriginClass.TYPOSQUAT, OriginClass.IDN):
        site, _ = build_site(SEED, brand="acme", flow="login", oc=oc,
                             secret=SecretKind.PASSWORD, semantics="neutral", replicate=0)
        state = TwinState(); state.reset(site)
        server = TwinServer(state, lab["pki"]["crt"], lab["pki"]["key"]).start()
        ctx = lab["browser"].new_context(proxy={"server": server.proxy_url}, ignore_https_errors=True)
        page = ctx.new_page()
        try:
            page.goto(f"https://{site.host('landing')}/login", wait_until="load")
            dom = page.evaluate("() => document.querySelector('form').outerHTML")
            hashes.add(hashlib.sha256(dom.encode()).hexdigest())
        finally:
            ctx.close(); server.stop()
    assert len(hashes) == 1
