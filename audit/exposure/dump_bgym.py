"""RQ1 exposure dumps from BrowserGym + AgentLab's GenericAgent (runs in .venv-bgym: browsergym-core
0.14.3 with its pinned playwright 1.44 and Chromium 125 = playwright build 1117 under
.venv-bgym/ms-playwright; agentlab 0.4.2 for the prompt layer).

Environment side, BrowserGym's own: `BrowserEnv` launches Chromium through its pinned Playwright, the
built-in open-ended task (`OpenEndedTask.setup`: page.goto(start_url)) opens the landing URL, and
`env.reset()` returns BrowserGym's observation dict (bid marking in every frame, CDP DOMSnapshot and
per-frame Accessibility.getFullAXTree merged across iframes). The only deviations from a stock
environment are network plumbing: pw_chromium_kwargs={"proxy": {"server": <sinkhole>}} and
pw_context_kwargs={"ignore_https_errors": True}.

Agent side, AgentLab's GenericAgent (the reference agent of the BrowserGym ecosystem) with
FLAGS_GPT_4o, the flags of AGENT_4o / AGENT_4o_MINI and of every Azure / Claude preset in
agentlab/agents/generic_agent/agent_configs.py. The text is produced by AgentLab's own code:
    pobs = dynamic_prompting.make_obs_preprocessor(flags.obs)(obs)   # = GenericAgent.obs_preprocessor
    text = dynamic_prompting.Observation(pobs, flags.obs).prompt     # the "# Observation of current
                                                                       step:" block of MainPrompt
The rest of GenericAgent's prompt (goal, action space, hints, examples) does not depend on the page.
GenericAgentArgs.set_benchmark() sets flags.obs.use_tabs = benchmark.is_multi_tab and, for MiniWoB
only, flags.obs.use_html = True (agentlab/agents/generic_agent/generic_agent.py), so the default
observation exists in three variants (browsergym/experiments/benchmark/configs.py: is_multi_tab):
  bgym_axtree            use_tabs=True: multi-tab benchmarks (WebArena, VisualWebArena, WorkArena
                         L2/L3, AssistantBench, WebLINX): "Currently open tabs" (title + URL) + AXTree
  bgym_axtree_1tab       use_tabs=False: WorkArena L1 (single-tab): AXTree only
  bgym_axtree_html_1tab  use_tabs=False, use_html=True: MiniWoB (single-tab, HTML switched on by
                         set_benchmark): pruned HTML + AXTree
and one non-default variant:
  bgym_dom          use_html=True, use_ax_tree=False, use_tabs=True: the pruned HTML
                    (flatten_dom_to_str -> prune_html, raw attributes) GenericAgent shows with HTML on
The intermediate texts (axtree_txt, dom_txt, pruned_html) are kept in each .json sidecar.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from labclient import CLASSES, FLOWS, Lab, write_dump, write_error  # noqa: E402

# pipeline -> (default_config, obs flag overrides on FLAGS_GPT_4o, description)
VARIANTS: dict[str, tuple[bool, dict, str]] = {
    "bgym_axtree": (True, {"use_tabs": True},
                    "GenericAgent FLAGS_GPT_4o on a multi-tab benchmark (use_tabs=True): tabs + AXTree"),
    "bgym_axtree_1tab": (True, {"use_tabs": False},
                         "GenericAgent FLAGS_GPT_4o on a single-tab benchmark (WorkArena L1; use_tabs=False): "
                         "AXTree only"),
    "bgym_axtree_html_1tab": (True, {"use_tabs": False, "use_html": True},
                              "GenericAgent FLAGS_GPT_4o on MiniWoB (set_benchmark: use_html=True, use_tabs=False): "
                              "pruned HTML + AXTree"),
    "bgym_dom": (False, {"use_tabs": True, "use_html": True, "use_ax_tree": False},
                 "GenericAgent FLAGS_GPT_4o with use_html=True, use_ax_tree=False (use_tabs=True): pruned HTML"),
}


def observation_text(obs: dict, overrides: dict) -> tuple[str, dict, dict]:
    """GenericAgent's observation block for one flag variant: (text, flags used, preprocessed pieces)."""
    from dataclasses import asdict

    from agentlab.agents import dynamic_prompting as dp
    from agentlab.agents.generic_agent.agent_configs import FLAGS_GPT_4o

    flags = FLAGS_GPT_4o.copy()  # deep copy: the module-level preset is never mutated
    for k, v in overrides.items():
        setattr(flags.obs, k, v)
    pobs = dp.make_obs_preprocessor(flags.obs)(obs)
    text = dp.Observation(pobs, flags.obs).prompt
    pieces = {k: pobs[k] for k in ("axtree_txt", "dom_txt", "pruned_html")}
    return text, asdict(flags.obs), pieces


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ports", required=True)
    ap.add_argument("--out", required=True, help="results/<ns>/exposure/raw")
    ap.add_argument("--flows", nargs="+", default=list(FLOWS))
    ap.add_argument("--classes", nargs="+", default=list(CLASSES))
    args = ap.parse_args(argv)

    from importlib.metadata import version

    from browsergym.core.env import BrowserEnv
    from browsergym.core.task import OpenEndedTask

    lab = Lab(args.ports)
    out = Path(args.out)
    versions = {d: version(d) for d in ("browsergym-core", "agentlab", "playwright")}
    n = failed = 0
    for flow in args.flows:
        for oc in args.classes:
            ep = None
            env = None
            try:
                ep = lab.episode(oc, flow)
                env = BrowserEnv(task_entrypoint=OpenEndedTask,
                                 task_kwargs={"start_url": ep["landing_url"], "goal": "Sign in."},
                                 headless=True, pw_chromium_kwargs={"proxy": {"server": lab.proxy_url}},
                                 pw_context_kwargs={"ignore_https_errors": True})
                obs, _info = env.reset()
                fv = (f"browsergym-core {versions['browsergym-core']}, agentlab {versions['agentlab']}, "
                      f"playwright {versions['playwright']}, Chromium {env.browser.version}")
                for pipeline, (default, overrides, desc) in VARIANTS.items():
                    text, flags, pieces = observation_text(obs, overrides)
                    write_dump(out, pipeline, flow, oc, text, {
                        "pipeline": pipeline, "framework": "browsergym-core", "framework_family": "browsergym",
                        "default_config": default, "config": desc, "framework_version": fv,
                        "versions": versions, "episode": ep, "observed_top_url": obs["url"],
                        "api": "BrowserEnv(task_entrypoint=OpenEndedTask).reset() -> agentlab dynamic_prompting."
                               f"Observation(make_obs_preprocessor(flags.obs)(obs), flags.obs).prompt, "
                               f"flags = FLAGS_GPT_4o with obs {overrides}",
                        "obs_flags": flags, "pieces": pieces,
                        "open_pages_urls": list(obs["open_pages_urls"])})
                n += 1
            except Exception as e:  # noqa: BLE001 - recorded per pipeline; never scored as "absent"
                failed += 1
                for pipeline in VARIANTS:
                    write_error(out, pipeline, flow, oc, e)
                print(f"[dump_bgym] {flow}/{oc} FAILED: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            finally:
                if env is not None:
                    env.close()
    print(f"[dump_bgym] {n} conditions x {len(VARIANTS)} pipelines, {failed} failed -> {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
