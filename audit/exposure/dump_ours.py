"""RQ1 exposure dumps from the MAIN venv: our observation arms and Playwright's aria snapshots.

Runs against `originblind lab serve` (ports.json) exactly like the external-framework dumps: reset the
twin per condition, open the landing URL through the sinkhole proxy in headless Chromium (Playwright
1.63, certificate errors ignored, the harness's `driver: chromium` launch flags), and save the text a
model would receive on its first look at the landing page:

  ours_T0 / ours_T1 / ours_T1u / ours_T2
      originblind.agent.observe.serialize(PlaywrightDriver(page).observe(), arm, bound_domain): the
      harness's own first observation for each text arm (what the agent loop sends)
  pw_aria      (Playwright, default)
      page.locator("body").aria_snapshot(): Playwright's aria snapshot in its default mode (what
      aria-snapshot assertions and many thin wrappers hand to a model)
  pw_aria_ai   (Playwright, non-default option)
      page.aria_snapshot(mode="ai"): the AI-mode snapshot (element refs + iframe contents) that
      Playwright MCP builds its page snapshot from; MCP's own tool response is dumped by dump_mcp.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from labclient import CLASSES, FLOWS, Lab, write_dump, write_error  # noqa: E402

ARMS = ("T0", "T1", "T1u", "T2")
PIPELINES = (*(f"ours_{a}" for a in ARMS), "pw_aria", "pw_aria_ai")
_LAUNCH_ARGS = ["--test-type", "--disable-features=Translate,AutofillServerCommunication",
                "--no-first-run", "--no-default-browser-check", "--ignore-certificate-errors"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ports", required=True)
    ap.add_argument("--out", required=True, help="results/<ns>/exposure/raw")
    ap.add_argument("--flows", nargs="+", default=list(FLOWS))
    ap.add_argument("--classes", nargs="+", default=list(CLASSES))
    args = ap.parse_args(argv)

    from importlib.metadata import version

    from playwright.sync_api import sync_playwright

    from originblind.agent.observe import serialize
    from originblind.browser.pw_driver import PlaywrightDriver

    lab = Lab(args.ports)
    out = Path(args.out)
    n = failed = 0
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=_LAUNCH_ARGS, ignore_default_args=["--enable-automation"])
        fv = f"playwright {version('playwright')}, Chromium {browser.version}"
        for flow in args.flows:
            for oc in args.classes:
                ctx = None
                try:
                    ep = lab.episode(oc, flow)
                    ctx = browser.new_context(proxy={"server": lab.proxy_url}, ignore_https_errors=True,
                                              viewport={"width": 1280, "height": 840})
                    page = ctx.new_page()
                    drv = PlaywrightDriver(page)
                    drv.goto(ep["landing_url"])
                    state = drv.observe()
                    base = {"framework_version": fv, "episode": ep, "observed_top_url": page.url}
                    for arm in ARMS:
                        write_dump(out, f"ours_{arm}", flow, oc, serialize(state, arm, ep["bound_domain"]), {
                            **base, "pipeline": f"ours_{arm}", "framework": "originblind", "framework_family": "ours",
                            "default_config": True, "config": f"harness text arm {arm}",
                            "api": f"originblind.agent.observe.serialize(PlaywrightDriver(page).observe(), {arm!r})"})
                    write_dump(out, "pw_aria", flow, oc, page.locator("body").aria_snapshot(), {
                        **base, "pipeline": "pw_aria", "framework": "playwright", "framework_family": "playwright",
                        "default_config": True, "config": "aria snapshot, default mode",
                        "api": "page.locator('body').aria_snapshot()"})
                    write_dump(out, "pw_aria_ai", flow, oc, page.aria_snapshot(mode="ai"), {
                        **base, "pipeline": "pw_aria_ai", "framework": "playwright", "framework_family": "playwright",
                        "default_config": False, "config": "aria snapshot, mode='ai' (Playwright MCP's source)",
                        "api": "page.aria_snapshot(mode='ai')"})
                    n += 1
                except Exception as e:  # noqa: BLE001 - recorded; never scored as "absent"
                    failed += 1
                    for p in PIPELINES:
                        write_error(out, p, flow, oc, e)
                    print(f"[dump_ours] {flow}/{oc} FAILED: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                finally:
                    if ctx is not None:
                        ctx.close()
        browser.close()
    print(f"[dump_ours] {n} conditions x {len(PIPELINES)} pipelines, {failed} failed -> {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
