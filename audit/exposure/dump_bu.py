"""RQ1 exposure dumps from browser-use (runs in .venv-bu: browser-use 0.13.11, CDP via cdp-use 1.4.5).

browser-use launches the Chromium binary itself and drives it over CDP (no Playwright). We give it the
binary its own discovery picks on this host (LocalBrowserWatchdog._find_installed_browser_path(): the
Playwright-cached chromium-* of the main venv), the sinkhole proxy (BrowserProfile.proxy) and
`--ignore-certificate-errors`; default extensions are disabled (enable_default_extensions=False) so the
browser makes no third-party fetches. Everything else is the BrowserProfile default.

The LLM-facing text is produced by the code path Agent.step uses at 0.13.11:
  summary = await BrowserSession.get_browser_state_summary(include_screenshot=True)
            (Agent._prepare_context always captures the screenshot, even with use_vision=False)
  AgentMessagePrompt(browser_state_summary=summary, file_system=None,
                     include_attributes=None)._get_browser_state_description()
which renders the page stats, the tab list (`Tab <id4>: <url> - <title[:30]>`), the page info and
summary.dom_state.llm_representation(DEFAULT_INCLUDE_ATTRIBUTES) (the Agent passes its default
include_attributes, which resolves to the same list); AgentMessagePrompt.get_user_message wraps it in
<browser_state> tags inside the user message, as we do here. The other parts of that message
(<agent_history>, <agent_state> with the task, step info) do not depend on the page.
Pipelines written:
  bu_default   BrowserProfile defaults (cross_origin_iframes=True at 0.13.11)
  bu_coi_off   cross_origin_iframes=False (the documented low-complexity setting)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from labclient import CLASSES, FLOWS, Lab, write_dump, write_error  # noqa: E402

# pipeline -> (default_config, BrowserProfile overrides, description)
VARIANTS: dict[str, tuple[bool, dict, str]] = {
    "bu_default": (True, {}, "browser-use Agent state message, default BrowserProfile (cross_origin_iframes=True)"),
    "bu_coi_off": (False, {"cross_origin_iframes": False},
                   "browser-use Agent state message, BrowserProfile(cross_origin_iframes=False)"),
}


def chrome_path() -> str:
    from browser_use.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog

    p = os.environ.get("OB_BU_CHROME") or LocalBrowserWatchdog._find_installed_browser_path()
    if not p:
        raise SystemExit("browser-use found no Chromium (install one with playwright, or set OB_BU_CHROME)")
    return p


def make_profile(lab: Lab, chrome: str, **overrides):
    from browser_use import BrowserProfile
    from browser_use.browser.profile import ProxySettings

    return BrowserProfile(headless=True, executable_path=chrome, proxy=ProxySettings(server=lab.proxy_url),
                          args=["--ignore-certificate-errors"], enable_default_extensions=False, **overrides)


async def browser_state_text(session) -> tuple[str, object]:
    from browser_use.agent.prompts import AgentMessagePrompt

    summary = await session.get_browser_state_summary(include_screenshot=True)
    body = AgentMessagePrompt(browser_state_summary=summary, file_system=None,
                              include_attributes=None)._get_browser_state_description()
    return "<browser_state>\n" + body.strip("\n") + "\n</browser_state>\n", summary


async def run(args) -> int:
    from importlib.metadata import version

    from browser_use import BrowserSession

    lab = Lab(args.ports)
    out = Path(args.out)
    chrome = chrome_path()
    versions = {d: version(d) for d in ("browser-use", "cdp-use", "bubus")}
    n = failed = 0
    for flow in args.flows:
        for oc in args.classes:
            try:
                ep = lab.episode(oc, flow)
            except Exception as e:  # noqa: BLE001
                failed += 1
                for pipeline in VARIANTS:
                    write_error(out, pipeline, flow, oc, e)
                continue
            for pipeline, (default, overrides, desc) in VARIANTS.items():
                profile = make_profile(lab, chrome, **overrides)
                session = BrowserSession(browser_profile=profile)
                try:
                    await session.start()
                    await session.navigate_to(ep["landing_url"])
                    text, summary = await browser_state_text(session)
                    write_dump(out, pipeline, flow, oc, text, {
                        "pipeline": pipeline, "framework": "browser-use", "framework_family": "browser_use",
                        "default_config": default, "config": desc,
                        "framework_version": f"browser-use {versions['browser-use']}, cdp-use {versions['cdp-use']}",
                        "versions": versions, "chrome_executable": chrome,
                        "cross_origin_iframes": profile.cross_origin_iframes, "episode": ep,
                        "observed_top_url": summary.url, "tabs": [t.model_dump() for t in summary.tabs],
                        "api": "BrowserSession.navigate_to(url); get_browser_state_summary(include_screenshot=True)"
                               " -> AgentMessagePrompt(...)._get_browser_state_description()"})
                except Exception as e:  # noqa: BLE001 - recorded; never scored as "absent"
                    failed += 1
                    write_error(out, pipeline, flow, oc, e)
                    print(f"[dump_bu] {pipeline} {flow}/{oc} FAILED: {type(e).__name__}: {e}", file=sys.stderr,
                          flush=True)
                finally:
                    try:
                        await session.kill()
                    except Exception:  # noqa: BLE001
                        pass
            n += 1
    print(f"[dump_bu] {n} conditions x {len(VARIANTS)} pipelines, {failed} failed -> {out}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ports", required=True)
    ap.add_argument("--out", required=True, help="results/<ns>/exposure/raw")
    ap.add_argument("--flows", nargs="+", default=list(FLOWS))
    ap.add_argument("--classes", nargs="+", default=list(CLASSES))
    return asyncio.run(run(ap.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
