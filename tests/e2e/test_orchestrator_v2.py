"""Orchestrator v2: intent-to-treat on agent errors, fresh canaries on infra retries, circuit
breaker, resume guard, probe-once-per-key, tool path, D3info/D3push rendering, success terminal."""

import gzip
import json

import pytest

from originblind.agent.httpx_driver import HttpxBrowser
from originblind.agent.loop import run_episode
from originblind.config import ExperimentSpec, ModelSpec
from originblind.controls import always_fill
from originblind.defense.defenses import Defense
from originblind.lab.twin import TwinState, make_app
from originblind.llm.client import InfraError, Sampling
from originblind.llm.mock import ScriptedBackend
from originblind.oracle.reconcile import reconcile
from originblind.run.episode import FLOW_SECRET, build_site
from originblind.run.orchestrator import ResumeMismatch, read_records, run_plan
from originblind.run.plan import ordered
from originblind.types import OriginClass

SEED = b"\x07" * 16


def _spec(**kw):
    base = dict(name="v2", master_seed_hex=SEED.hex(), brands=["northwell", "vantafin"],
                models=[ModelSpec(key="mock", hf_repo="x", served_name="mock", pipeline="text")],
                origin_classes=["exact", "typosquat", "a2_iframe"], flows=["login"],
                observation_arms=["T1"], defenses=["D1"], replicates_per_brand=1,
                recognition_probe=False)
    base.update(kw)
    return ExperimentSpec(**base)


def _episode(policy, oc=OriginClass.EXACT, defense="D1", arm="T1", flow="login"):
    site, can = build_site(SEED, brand="northwell", flow=flow, oc=oc, secret=FLOW_SECRET[flow],
                           semantics="neutral", replicate=0)
    st = TwinState(); st.reset(site)
    drv = HttpxBrowser(make_app(st))
    tr = run_episode(drv, ScriptedBackend(policy), "mock", site, Defense(defense), can, arm, Sampling(),
                     step_budget=8)
    res = reconcile(can, site.bound_domain, st.log, drv.network_log(), authorized=site.authorized_domains)
    return tr, res


def test_agent_errors_are_itt_outcomes_not_infra():
    seq = iter(['goto("ftp://evil")', "click(e99)", 'fill(e42, "{{password}}")', 'stop("give up")'])
    tr, res = _episode(lambda obs, s: next(seq))
    assert tr.nav_errors == 1 and tr.invalid_bid_actions == 2
    assert tr.terminal_type == "stop" and not res.leak and not res.r_bound


def test_check_origin_tool_reports_verdict_and_rejects_non_fields():
    def policy(obs, step):
        if step == 0:
            return "click(e1)"  # follow the inbox link
        if step == 1:
            import re

            pw = re.search(r"\[(e\d+)\] input name='password'", obs).group(1)
            return f"check_origin({pw})"
        if step == 2:
            return "check_origin(e99)"  # not a field on this page
        return 'refuse("origin mismatch")'

    tr, _ = _episode(policy, oc=OriginClass.A2_IFRAME, defense="D3")
    notes = [s.note for s in tr.steps if s.action_name == "check_origin"]
    assert tr.check_origin_calls == 2 and "verdict=DENY" in notes[0] and "error" in notes[1]


@pytest.mark.parametrize("defense,needle,absent", [("D3push", "-> DENY", None), ("D3info", "ORIGIN FACTS", "-> DENY")])
def test_push_rendering(defense, needle, absent):
    seen = []

    def policy(obs, step):
        seen.append(obs)
        return "click(e1)" if step == 0 else 'stop("x")'

    _episode(policy, oc=OriginClass.A2_ACTION, defense=defense)
    landing = seen[1]
    assert needle in landing
    if absent:
        assert absent not in landing


def test_infra_error_retry_uses_fresh_canary(tmp_path):
    spec = _spec(origin_classes=["exact"], brands=["northwell"])
    items = ordered(spec)
    out = tmp_path / "episodes.jsonl"
    calls = {"n": 0}

    class Flaky:
        def __init__(self):
            self.inner = ScriptedBackend(always_fill())

        def complete(self, model, messages, sampling):
            calls["n"] += 1
            if calls["n"] == 1:
                raise InfraError("vLLM unreachable")
            return self.inner.complete(model, messages, sampling)

    r1 = run_plan(spec, items, out, lambda s: Flaky(), probe=False, namespace="t")
    assert r1["infra_error"] == 1 and r1["run"] == 0
    r2 = run_plan(spec, items, out, lambda s: Flaky(), probe=False, namespace="t")
    assert r2["run"] == 1
    recs, _ = read_records(out)
    assert [r["status"] for r in recs] == ["infra_error", "complete"]
    assert recs[1]["attempt"] == 1  # the retry is attempt 1 -> a fresh canary was minted
    assert recs[1]["outcome"]["r_bound"] and recs[1]["outcome"]["terminal_type"] == "success"


def test_circuit_breaker_aborts(tmp_path):
    spec = _spec()
    items = ordered(spec)

    class Down:
        def complete(self, *a, **k):
            raise InfraError("down")

    r = run_plan(spec, items, tmp_path / "e.jsonl", lambda s: Down(), probe=False, namespace="t",
                 max_consecutive_infra=3, health_check=lambda: False)
    assert r["aborted"] and r["infra_error"] == 3


def test_resume_guard_refuses_changed_design(tmp_path):
    out = tmp_path / "episodes.jsonl"
    spec = _spec()
    run_plan(spec, ordered(spec), out, lambda s: ScriptedBackend(always_fill()), probe=False, namespace="t")
    changed = _spec(step_budget=5)
    with pytest.raises(ResumeMismatch):
        run_plan(changed, ordered(changed), out, lambda s: ScriptedBackend(always_fill()), probe=False, namespace="t")


def test_probe_computed_once_per_key_and_transcripts_written(tmp_path):
    spec = _spec(defenses=["D1", "D3push", "D6r"], recognition_probe=True, probe_k=2)
    items = ordered(spec)
    out = tmp_path / "episodes.jsonl"
    probe_calls = {"n": 0}

    class Counting:
        def __init__(self):
            self.inner = ScriptedBackend(always_fill())

        def complete(self, model, messages, sampling):
            if messages[0].content == "Answer concisely.":
                probe_calls["n"] += 1
                return "YES" if "YES or NO" in messages[1].content else "northwell.com"
            return self.inner.complete(model, messages, sampling)

    r = run_plan(spec, items, out, lambda s: Counting(), probe=True, namespace="t", workers=4)
    assert r["run"] == len(items)
    n_keys = len({(it.brand, it.origin_class) for it in items})
    assert probe_calls["n"] == n_keys * 2 * 2  # readout + discrimination, k=2, once per key
    probes = [json.loads(l) for l in (tmp_path / "probes.jsonl").read_text().splitlines()]
    assert len(probes) == n_keys
    with gzip.open(tmp_path / "transcripts.jsonl.gz", "rt") as fh:
        trs = [json.loads(l) for l in fh]
    assert len(trs) == len(items) and all(t["steps"] for t in trs)


@pytest.mark.parametrize("defense,legend", [("D3push", "ORIGIN CHECK section"), ("D3info", "ORIGIN FACTS section"),
                                            ("D1", None), ("D3", None)])
def test_push_legend_is_opt_in_and_only_for_pushed_arms(defense, legend):
    """push_legend=False reproduces the pilot-v2 prompts byte for byte; True adds the D3-tool-style
    description only to the arms that push facts or a verdict."""
    systems = {}
    for flag in (False, True):
        seen = []

        class Rec:
            def complete(self, model, messages, sampling):
                seen.append(messages[0].content)
                return 'stop("x")'

        site, can = build_site(SEED, brand="northwell", flow="login", oc=OriginClass.A2_IFRAME,
                               secret=FLOW_SECRET["login"], semantics="neutral", replicate=0)
        st = TwinState(); st.reset(site)
        run_episode(HttpxBrowser(make_app(st)), Rec(), "mock", site, Defense(defense), can, "T1", Sampling(),
                    step_budget=2, push_legend=flag)
        systems[flag] = seen[0]
    if legend is None:
        assert systems[False] == systems[True]
    else:
        assert legend not in systems[False] and legend in systems[True]
        assert systems[True].replace("\n" + systems[True].split("\n")[[i for i, l in enumerate(systems[True].split("\n")) if legend in l][0]], "") == systems[False]
