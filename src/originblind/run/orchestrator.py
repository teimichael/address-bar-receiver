"""Run a plan to a resumable JSONL episode store.

Each PlanItem becomes one episode: build the site, (optionally) run the stimulus probes, run the
agent loop against the real twin through a browser driver (in-process httpx, or real Chromium via
browser/run_browser.py), drain, reconcile the oracle over channel A (twin log) and channel B
(driver log), and append one record. Sidecars: transcripts.jsonl.gz (every step) and
probes.jsonl (one line per probe key).

Statuses (one JSONL line per attempt; analysis keeps the first `complete` per episode_id):
  complete        a measured episode (every agent behaviour is an intent-to-treat outcome)
  infra_error     the serving stack failed (vLLM down, timeout, 5xx, unknown model): retried on
                  resume with attempt+1, i.e. FRESH canaries
  harness_error   a bug in the instrument: retried on resume, flagged by `validate`
  excluded        a pre-registered instrument failure (e.g. unresolved oracle channels in the
                  browser): replaced on resume, counted per cell

Resumption is guarded by the run identity (config hash + plan hash + code hash), so episodes
produced under a different design or code can never be appended to the same file. A circuit
breaker aborts after K consecutive infra errors (exit code 3 at the CLI).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from originblind.agent.httpx_driver import HttpxBrowser
from originblind.agent.loop import HarnessError, run_episode, secret_bids
from originblind.agent.observe import SCREEN_ARMS
from originblind.agent.probe import ProbeResult, probe_key, run_probe
from originblind.agent.prompts import task_goal
from originblind.config import ExperimentSpec
from originblind.defense.defenses import Defense
from originblind.lab.pages import entry_path
from originblind.lab.twin import TwinState, make_app
from originblind.llm.client import Backend, InfraError, Sampling, preset_for
from originblind.oracle.attribute import registrable_domain
from originblind.oracle.matcher import find_canary
from originblind.oracle.reconcile import reconcile
from originblind.run.episode import build_site
from originblind.run.plan import PlanItem, run_identity
from originblind.types import OriginClass, SecretKind

BackendFactory = Callable[[str], Backend]
SCHEMA_VERSION = 2


class ResumeMismatch(RuntimeError):
    pass


class ExcludedEpisode(RuntimeError):
    """A pre-registered instrument failure: the episode is excluded and replaced on resume."""

    excluded = True


# ---------------------------------------------------------------- store helpers

def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_records(path: Path) -> tuple[list[dict], int]:
    """All parseable records plus the number of corrupt lines (tolerant reader)."""
    recs, corrupt = [], 0
    if not path.exists():
        return recs, 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            corrupt += 1
    return recs, corrupt


def _attempt_state(path: Path) -> tuple[set[str], dict[str, int]]:
    """(ids with a complete record, number of attempts recorded per id = next attempt index)."""
    done: set[str] = set()
    counts: dict[str, int] = {}
    recs, _ = read_records(path)
    for r in recs:
        eid = r.get("episode_id")
        if not eid:
            continue
        counts[eid] = counts.get(eid, 0) + 1
        if r.get("status") == "complete":
            done.add(eid)
    return done, counts


def guard_resume(spec: ExperimentSpec, out_path: Path, *, allow_code_change: bool = False) -> dict:
    """Refuse to append to results produced under a different design or code version."""
    ident = run_identity(spec)
    man = out_path.parent / "manifest.json"
    if out_path.exists() and out_path.stat().st_size > 0:
        if not man.exists():
            raise ResumeMismatch(f"{out_path} exists without a manifest.json; refusing to append")
        prev = json.loads(man.read_text())
        for k in ("config_hash", "plan_sha256"):
            if prev.get(k) and prev[k] != ident[k]:
                raise ResumeMismatch(f"Refusing to resume: {k} differs ({prev[k][:12]} != {ident[k][:12]}). "
                                     "Use a fresh --namespace or remove the directory.")
        if prev.get("code_sha256") and prev["code_sha256"] != ident["code_sha256"] and not allow_code_change:
            raise ResumeMismatch("Refusing to resume: harness code changed since these results were "
                                 "produced (code_sha256 differs). Use --allow-code-change only for "
                                 "documented, outcome-irrelevant changes.")
    return ident


def write_manifests(spec: ExperimentSpec, items: list[PlanItem], out_path: Path, extra: dict) -> dict:
    """manifest.json = the run identity (written once); manifests/<utc>.json = this invocation."""
    from originblind.run.provenance import build_manifest

    repo_root = Path(__file__).resolve().parents[3]
    models = [m.model_dump() for m in spec.models]
    manifest = build_manifest(repo_root, config_hash=spec.config_hash(), plan_sha256="", models=models,
                              images=["vllm/vllm-openai:v0.31.0", "mcr.microsoft.com/playwright/python:v1.63.0-noble"])
    manifest.update(run_identity(spec))
    manifest.update({"n_planned": len(items), "schema_version": SCHEMA_VERSION, "spec": spec.model_dump(mode="json")})
    manifest.update(extra)
    man = out_path.parent / "manifest.json"
    if not man.exists():
        man.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    inv = out_path.parent / "manifests"
    inv.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    (inv / f"{stamp}.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


# ---------------------------------------------------------------- probe cache

class ProbeCache:
    """Compute each probe key once (thread-safe); persist to probes.jsonl."""

    def __init__(self, path: Path | None) -> None:
        self._lock = threading.Lock()
        self._events: dict[str, threading.Event] = {}
        self._vals: dict[str, dict | None] = {}
        self._path = path
        if path is not None and path.exists():
            for line in path.read_text().splitlines():
                try:
                    d = json.loads(line)
                    self._vals[d["probe_key"]] = d
                except Exception:  # noqa: BLE001
                    continue

    def get_or_compute(self, key: str, fn: Callable[[], dict | None]) -> dict | None:
        with self._lock:
            if key in self._vals:
                return self._vals[key]
            ev = self._events.get(key)
            owner = ev is None
            if owner:
                ev = self._events[key] = threading.Event()
        if not owner:
            ev.wait(timeout=600)
            return self._vals.get(key)
        try:
            val = fn()
        except Exception:
            with self._lock:
                self._events.pop(key, None)
            ev.set()
            raise
        with self._lock:
            self._vals[key] = val
            if val is not None and self._path is not None:
                with self._path.open("a") as fh:
                    fh.write(json.dumps(val) + "\n")
        ev.set()
        return val


# ---------------------------------------------------------------- one episode

def _seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256("::".join(map(str, parts)).encode()).digest()[:4], "big") % (2**31 - 10_000)


def compute_probe(spec: ExperimentSpec, it: PlanItem, site, backend: Backend, sampling: Sampling) -> dict | None:
    """Readout + discrimination probes on the landing-page snapshot (in-process driver; the
    driver-parity test guarantees the identical serialization in Chromium)."""
    if it.arm in SCREEN_ARMS:
        return None
    state = TwinState()
    state.reset(site)
    drv = HttpxBrowser(make_app(state))
    try:
        drv.goto(f"https://{site.host('landing')}/{entry_path(site)}")
        page = drv.observe()
        sb = secret_bids(page, site.secret)
        if not sb:
            return None
        el = page.by_bid(sb[0])
        sink = el.form_action or el.frame_url or page.top_url
        key = probe_key(it.model_key, it.brand, it.flow, it.origin_class, it.semantics, it.arm, it.replicate)
        pr: ProbeResult = run_probe(backend, it.model_served, page, sb[0], sink, it.arm, site.bound_domain,
                                    sampling, k=spec.probe_k, key=key, goal=task_goal(site),
                                    secret=site.secret.value, discrimination=spec.discrimination_probe)
        return pr.to_json()
    finally:
        drv.close()


def _success_checker(state: TwinState, canaries, authorized: tuple[str, ...]) -> Callable[[], bool]:
    seen = {"i": 0}

    def check() -> bool:
        log = state.log
        for req in log[seen["i"]:]:
            if registrable_domain(req.host or req.url) in authorized and any(
                    find_canary(c.canary_id, req) for c in canaries):
                return True
        seen["i"] = len(log)
        return False

    return check


def episode_record(spec: ExperimentSpec, it: PlanItem, attempt: int, namespace: str, site, canaries,
                   tr, res, probe: dict | None, *, wall_s: float, started: str, driver_meta: dict) -> dict:
    oc = OriginClass(it.origin_class)
    utility = site.flow in ("psp", "psp_nocsp")
    return {
        "status": "complete",
        "schema_version": SCHEMA_VERSION,
        "namespace": namespace,
        "episode_id": it.episode_id,
        "attempt": attempt,
        "started_at": started,
        "wall_s": round(wall_s, 3),
        "factors": {
            "model_key": it.model_key, "brand": it.brand, "flow": it.flow,
            "origin_class": it.origin_class, "adversary": oc.adversary, "secret": it.secret,
            "semantics": it.semantics, "arm": it.arm, "defense": it.defense, "pii_mode": it.pii_mode,
            "replicate": it.replicate, "driver": it.driver, "environment": spec.environment,
        },
        "bound_domain": site.bound_domain,
        "authorized_domains": list(site.authorized_domains),
        "outcome": {
            "r_bound": res.r_bound, "r_authorized": res.r_authorized, "leak": res.leak,
            "release_domains": res.release_domains, "leak_domains": res.leak_domains,
            "task_success": res.r_authorized if utility else res.r_bound,
            "self_corrected": bool(res.r_bound and oc.adversary == "A1"),
            "terminal_type": tr.terminal_type,
            "abstained": tr.abstained,
            "entry_offorigin": tr.entry_offorigin,
            "secret_filled": tr.secret_fills > 0,
            "submitted_after_secret": tr.submitted_after_secret,
            "channel_agreement": res.channel_agreement,
            "channel_detail": res.channel_detail,
        },
        "telemetry": {
            "steps": len(tr.steps), "parse_failures": tr.parse_failures,
            "check_origin_calls": tr.check_origin_calls, "d3push_shown_steps": tr.d3push_shown_steps,
            "broker_denials": tr.broker_denials, "egress_blocks": tr.egress_blocks,
            "fills_performed": tr.fills_performed, "invalid_bid_actions": tr.invalid_bid_actions,
            "nav_errors": tr.nav_errors, "secret_fill_attempts": tr.secret_fill_attempts,
            "secret_fills": tr.secret_fills, "steps_to_first_secret_fill": tr.steps_to_first_secret_fill,
            "handle_repairs": tr.handle_repairs,
            "secret_fill_contexts": tr.secret_fill_contexts, "llm_calls": tr.llm_calls,
            "llm_latency_ms": round(tr.llm_latency_ms, 1), **driver_meta,
        },
        "probe": probe,
        "hits": res.hits,
    }


def run_one(spec: ExperimentSpec, it: PlanItem, attempt: int, namespace: str, backend: Backend, *,
            driver_factory: Callable[[TwinState], tuple[object, Callable[[], None]]] | None = None,
            twin_state: TwinState | None = None, probes: ProbeCache | None = None,
            use_probe: bool = True, model_thinking: bool = False) -> tuple[dict, dict]:
    """Run one episode. Returns (record, transcript). Raises InfraError / HarnessError / others."""
    t0 = time.time()
    started = _utc()
    oc = OriginClass(it.origin_class)
    secret = SecretKind(it.secret)
    site, canaries = build_site(
        spec.master_seed, brand=it.brand, flow=it.flow, oc=oc, secret=secret, semantics=it.semantics,
        replicate=it.replicate, attempt=attempt, eid=it.episode_id, bitb=(it.arm == "S2"),
        environment=spec.environment,
    )
    sampling = preset_for(it.model_key, thinking=model_thinking)

    probe = None
    if use_probe and probes is not None and it.arm not in SCREEN_ARMS:
        key = probe_key(it.model_key, it.brand, it.flow, it.origin_class, it.semantics, it.arm, it.replicate)
        probe = probes.get_or_compute(key, lambda: compute_probe(spec, it, site, backend, sampling))

    state = twin_state or TwinState()
    state.reset(site)
    if driver_factory is None:
        driver = HttpxBrowser(make_app(state))
        cleanup = driver.close
    else:
        driver, cleanup = driver_factory(state)
    try:
        tr = run_episode(driver, backend, it.model_served, site, Defense(it.defense), canaries, it.arm,
                         sampling, step_budget=spec.step_budget, reasoning=spec.reasoning,
                         push_legend=spec.push_legend,
                         success_check=_success_checker(state, canaries, site.authorized_domains),
                         seed_base=_seed(it.episode_id, attempt))
        driver.drain()
        res = reconcile(canaries, site.bound_domain, state.log, driver.network_log(),
                        authorized=site.authorized_domains)
        # A canary fact seen by the browser (B) but not (yet) by the twin (A) is either a request
        # still in flight or a capture failure. Wait for A to catch up; if it never does, the
        # outcome is unknowable and the episode is a pre-registered instrument exclusion (G4).
        deadline = time.time() + 3.0
        while res.channel_detail.get("only_b") and time.time() < deadline:
            time.sleep(0.25)
            res = reconcile(canaries, site.bound_domain, state.log, driver.network_log(),
                            authorized=site.authorized_domains)
        if res.channel_detail.get("only_b"):
            raise ExcludedEpisode(f"unresolved channel-B canary facts: {res.channel_detail['only_b']}")
        meta = {}
        if hasattr(driver, "episode_meta"):
            meta = driver.episode_meta()
    finally:
        cleanup()

    rec = episode_record(spec, it, attempt, namespace, site, canaries, tr, res, probe,
                         wall_s=time.time() - t0, started=started, driver_meta={})
    rec["driver_meta"] = meta
    if tr.captures:
        rec["captures"] = tr.captures
    transcript = {"episode_id": it.episode_id, "attempt": attempt,
                  "steps": [s.to_json() for s in tr.steps]}
    return rec, transcript


# ---------------------------------------------------------------- the plan runner

def classify(exc: BaseException) -> str:
    if isinstance(exc, InfraError):
        return "infra_error"
    try:
        import openai

        if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError,
                            openai.NotFoundError, openai.RateLimitError)):
            return "infra_error"
    except Exception:  # noqa: BLE001
        pass
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return "infra_error"
    if getattr(exc, "excluded", False):
        return "excluded"
    return "harness_error"


def run_plan(
    spec: ExperimentSpec,
    items: list[PlanItem],
    out_path: Path,
    backend_factory: BackendFactory,
    *,
    probe: bool = True,
    namespace: str = "pilot",
    workers: int = 1,
    max_consecutive_infra: int = 8,
    health_check: Callable[[], bool] | None = None,
    allow_code_change: bool = False,
    manifest_extra: dict | None = None,
    transcripts: bool = True,
    model_thinking: dict[str, bool] | None = None,
) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    guard_resume(spec, out_path, allow_code_change=allow_code_change)
    write_manifests(spec, items, out_path, manifest_extra or {})
    done, counts = _attempt_state(out_path)
    todo = [it for it in items if it.episode_id not in done]
    stats = {"planned": len(items), "skipped": len(items) - len(todo), "run": 0, "infra_error": 0,
             "harness_error": 0, "excluded": 0, "aborted": False}
    t_start = time.time()
    model_thinking = model_thinking or {m.key: m.thinking for m in spec.models}

    if spec.driver != "inprocess" and todo:
        from originblind.browser.run_browser import run_items_browser

        res = run_items_browser(spec, todo, counts, out_path, backend_factory, namespace=namespace,
                                probe=probe, workers=workers, max_consecutive_infra=max_consecutive_infra,
                                health_check=health_check, transcripts=transcripts, model_thinking=model_thinking)
        stats.update({k: stats.get(k, 0) + v if isinstance(v, int) and not isinstance(v, bool) else v
                      for k, v in res.items()})
        stats["wall_s"] = round(time.time() - t_start, 2)
        return stats

    probes = ProbeCache(out_path.parent / "probes.jsonl") if probe else None
    lock = threading.Lock()
    tpath = out_path.parent / "transcripts.jsonl.gz"
    consecutive_infra = 0
    abort = threading.Event()

    def work(it: PlanItem) -> tuple[dict, dict | None]:
        if abort.is_set():
            return {"episode_id": it.episode_id, "status": "skipped_abort"}, None
        attempt = counts.get(it.episode_id, 0)
        try:
            backend = backend_factory(it.model_served)
            return run_one(spec, it, attempt, namespace, backend, probes=probes, use_probe=probe,
                           model_thinking=model_thinking.get(it.model_key, False))
        except Exception as e:  # noqa: BLE001 - classified below, never silently dropped
            return ({"episode_id": it.episode_id, "status": classify(e), "attempt": attempt,
                     "namespace": namespace, "error": repr(e)[:500],
                     "traceback": traceback.format_exc()[-2000:]}, None)

    def write(rec: dict, tr: dict | None) -> None:
        nonlocal consecutive_infra
        if rec.get("status") == "skipped_abort":
            return
        with lock:
            with out_path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            if tr is not None and transcripts:
                with gzip.open(tpath, "at") as gz:
                    gz.write(json.dumps(tr) + "\n")
            st = rec.get("status")
            if st == "complete":
                stats["run"] += 1
                consecutive_infra = 0
            else:
                stats[st] = stats.get(st, 0) + 1
                if st == "infra_error":
                    consecutive_infra += 1
                    if consecutive_infra >= max_consecutive_infra:
                        healthy = health_check() if health_check else False
                        if not healthy:
                            abort.set()
                            stats["aborted"] = True
                        else:
                            consecutive_infra = 0

    if workers <= 1:
        for it in todo:
            if abort.is_set():
                break
            write(*work(it))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(work, it) for it in todo]
            for f in as_completed(futs):
                write(*f.result())

    stats["wall_s"] = round(time.time() - t_start, 2)
    return stats


def completeness(spec: ExperimentSpec, items: list[PlanItem], out_path: Path) -> dict:
    done, _ = _attempt_state(out_path)
    missing = [it.episode_id for it in items if it.episode_id not in done]
    return {"planned": len(items), "complete": len(items) - len(missing), "missing": len(missing)}


__all__ = ["run_plan", "run_one", "completeness", "read_records", "guard_resume", "ProbeCache",
           "ResumeMismatch", "HarnessError", "classify"]
