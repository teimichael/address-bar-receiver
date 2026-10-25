"""Driver parity and A2 render parity in real Chromium (pre-registered manipulation checks).

1. The in-process driver and real Chromium produce byte-identical T0/T1/T1u/T2 observations for
   every flow x origin class, on the hub and on the landing page. So in-process results are a
   faithful text-surface replica, and the real-browser runs differ only in browser behaviour.
2. In Chromium, the exact page and its A1/A2 twins render pixel-identically in the viewport:
   the A2 manipulation (cross-origin iframe / off-origin form action) is invisible on the page.
"""

import io

import pytest
from playwright.sync_api import sync_playwright

from originblind.agent.httpx_driver import HttpxBrowser
from originblind.agent.observe import serialize
from originblind.browser.pw_driver import PlaywrightDriver
from originblind.browser.server import TwinServer
from originblind.lab.pages import entry_path
from originblind.lab.pki import write_pki
from originblind.lab.registry import all_hosts, legit_apex
from originblind.lab.twin import TwinState, make_app
from originblind.run.episode import FLOW_SECRET, IDP_HOST, PSP_HOST, build_site
from originblind.types import OriginClass

pytestmark = pytest.mark.browser

SEED = bytes.fromhex("5107e5107e5107e5107e5107e5107e51")
BRAND = "northwell"


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    hosts = ["mail.webmail.test", PSP_HOST, IDP_HOST, legit_apex(BRAND)]
    hosts += [h.ascii_host for h in all_hosts(SEED, (BRAND,))]
    pki = write_pki(tmp_path_factory.mktemp("pki"), hosts, wildcards=["*.pspworld.test"])
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--ignore-certificate-errors"])
        yield {"pki": pki, "browser": browser}
        browser.close()


def _open(lab, site):
    st = TwinState(); st.reset(site)
    srv = TwinServer(st, lab["pki"]["crt"], lab["pki"]["key"]).start()
    ctx = lab["browser"].new_context(proxy={"server": srv.proxy_url}, ignore_https_errors=True,
                                     viewport={"width": 1280, "height": 840})
    page = ctx.new_page()
    return st, srv, ctx, page


CASES = [(f, oc) for f in ("login", "otp", "checkout") for oc in OriginClass] + \
        [("psp", OriginClass.EXACT), ("psp_nocsp", OriginClass.EXACT)]


@pytest.mark.parametrize("flow,oc", CASES, ids=[f"{f}-{o.value}" for f, o in CASES])
def test_observation_parity(lab, flow, oc):
    site, _ = build_site(SEED, brand=BRAND, flow=flow, oc=oc, secret=FLOW_SECRET[flow],
                         semantics="persuasive", replicate=0)
    st, srv, ctx, page = _open(lab, site)
    try:
        pd, hd = PlaywrightDriver(page), HttpxBrowser(make_app(st))
        for url in (f"https://{site.HUB_HOST}/", f"https://{site.host('landing')}/{entry_path(site)}"):
            pd.goto(url); hd.goto(url)
            po, ho = pd.observe(), hd.observe()
            for arm in ("T0", "T1", "T1u", "T2"):
                assert serialize(po, arm, site.bound_domain) == serialize(ho, arm, site.bound_domain), (url, arm)
            # ground truth used by the broker agrees too
            for el in ho.interactive:
                if el.tag == "input":
                    a, b = pd.field_context(el.bid), hd.field_context(el.bid)
                    assert (a.frame_url, a.form_action_url, a.nested) == (b.frame_url, b.form_action_url, b.nested)
    finally:
        ctx.close(); srv.stop()


def test_a2_and_a1_pixel_parity(lab):
    from PIL import Image, ImageChops

    shots = {}
    for oc in (OriginClass.EXACT, OriginClass.TYPOSQUAT, OriginClass.IDN, OriginClass.A2_ACTION,
               OriginClass.A2_IFRAME):
        site, _ = build_site(SEED, brand=BRAND, flow="login", oc=oc, secret=FLOW_SECRET["login"],
                             semantics="neutral", replicate=0)
        st, srv, ctx, page = _open(lab, site)
        try:
            page.goto(f"https://{site.host('landing')}/login", wait_until="load")
            page.wait_for_timeout(200)
            shots[oc] = Image.open(io.BytesIO(page.screenshot(type="png"))).convert("RGB")
        finally:
            ctx.close(); srv.stop()
    for oc, img in shots.items():
        assert ImageChops.difference(shots[OriginClass.EXACT], img).getbbox() is None, f"{oc} differs from exact"
