"""Run plan items in real Chromium (driver: chromium = headless on the host; chromium-headful =
headful under Xvfb inside the lab container, required for the screenshot arms).

Process model: a pool of worker processes (spawned, never forked after threads). Each worker owns
one Playwright Chromium and one TwinServer (TLS + HTTP twin listeners behind a SinkholeProxy, the
browser's only route). Per episode the worker resets the twin to the episode's site and opens a
fresh browser context (no shared cookies/cache), so episodes are independent. Probes are computed
once per probe key in the parent before dispatch (they use the frozen landing snapshot; the
driver-parity test guarantees the in-process serialization equals Chromium's), and records stream
back to the parent, which is the only writer of the JSONL store.
"""

from __future__ import annotations

import gzip
import json
import multiprocessing as mp
import os
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

from originblind.config import SCREEN_ARMS, ExperimentSpec
from originblind.lab.registry import BRANDS, all_hosts, legit_apex
from originblind.run.episode import infra_hosts, infra_wildcards
from originblind.run.plan import PlanItem

_W: dict = {}  # per-worker globals

# The lab's X screen geometry (docker/entrypoint.sh SCREEN): one pixel wider/taller than the window.
LAB_SCREEN = "1281x961x24"
# Opt-in (set by the lab runner): each headful worker gets its own Xvfb display with the lab's
# geometry, so screenshot-arm episodes can run concurrently; captures grab the worker's own display.
PER_WORKER_DISPLAY_ENV = "ORIGIN_BLIND_LAB_PER_WORKER_DISPLAY"


def per_worker_displays() -> bool:
    return os.environ.get(PER_WORKER_DISPLAY_ENV) == "1"


def start_private_display(screen: str = LAB_SCREEN, timeout_s: float = 15.0):
    """Start a private Xvfb (-displayfd picks a free display number); return (":N", Popen).

    Same server, geometry, depth and DPI as the entrypoint's display, so pixels are identical."""
    import select
    import subprocess

    r, w = os.pipe()
    proc = subprocess.Popen(["Xvfb", "-displayfd", str(w), "-screen", "0", screen, "-dpi", "96",
                             "-nolisten", "tcp"], pass_fds=(w,), stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    os.close(w)
    buf = b""
    deadline = time.monotonic() + timeout_s
    try:
        while not buf.endswith(b"\n"):
            left = deadline - time.monotonic()
            if left <= 0 or proc.poll() is not None:
                raise RuntimeError(f"private Xvfb did not come up (rc={proc.poll()})")
            ready, _, _ = select.select([r], [], [], left)
            if ready:
                chunk = os.read(r, 16)
                if not chunk:
                    raise RuntimeError("private Xvfb closed -displayfd without a display number")
                buf += chunk
    finally:
        os.close(r)
    return f":{int(buf.strip())}", proc


def plan_hosts(spec: ExperimentSpec, items: list[PlanItem]) -> list[str]:
    brands = sorted({it.brand for it in items}) or list(spec.brands or BRANDS)
    hosts = list(infra_hosts())
    for h in all_hosts(spec.master_seed, tuple(brands)):
        hosts.append(h.ascii_host)
    hosts += [legit_apex(b) for b in brands]
    return list(dict.fromkeys(hosts))


def _worker_init(cfg: dict) -> None:
    from playwright.sync_api import sync_playwright

    from originblind.browser.server import TwinServer
    from originblind.lab.twin import TwinState

    os.environ.setdefault("ORIGIN_BLIND_VLLM_URL", cfg["vllm_url"])
    headful = cfg["driver"] == "chromium-headful"
    if headful and cfg.get("per_worker_display"):
        import atexit

        display, xvfb = start_private_display()
        os.environ["DISPLAY"] = display  # before Playwright starts: Chromium and capture use it
        atexit.register(xvfb.terminate)
        _W["xvfb"] = xvfb
    pw = sync_playwright().start()
    args = ["--test-type", "--disable-features=Translate,AutofillServerCommunication",
            "--no-first-run", "--no-default-browser-check"]
    launch_kw: dict = {"headless": not headful, "args": args,
                       "ignore_default_args": ["--enable-automation"]}
    if headful:
        w, h = cfg["window"]
        args += [f"--window-size={w},{h}", "--window-position=0,0"]
    else:
        args += ["--ignore-certificate-errors"]
    browser = pw.chromium.launch(**launch_kw)
    state = TwinState()
    twin = TwinServer(state, Path(cfg["crt"]), Path(cfg["key"])).start()
    capture = None
    if headful:
        from originblind.browser.capture import LabCapture

        capture = LabCapture(window=tuple(cfg["window"]), chrome_px=cfg.get("chrome_px"))
    _W.update(pw=pw, browser=browser, state=state, twin=twin, headful=headful, capture=capture,
              window=tuple(cfg["window"]))


def _driver_factory(state):
    from originblind.browser.pw_driver import PlaywrightDriver

    kw: dict = {"proxy": {"server": _W["twin"].proxy_url}}
    if _W["headful"]:
        kw["no_viewport"] = True  # the page fills the real window: S1/S2 capture the real chrome
    else:
        kw["ignore_https_errors"] = True
        kw["viewport"] = {"width": _W["window"][0], "height": _W["window"][1] - 120}
    ctx = _W["browser"].new_context(**kw)
    page = ctx.new_page()
    drv = PlaywrightDriver(page, capture=_W["capture"])

    def cleanup() -> None:
        try:
            ctx.close()
        except Exception:  # noqa: BLE001
            pass

    return drv, cleanup


def _worker_run(spec_json: str, item: dict, attempt: int, namespace: str, probe: dict | None,
                thinking: bool) -> tuple[dict, dict | None]:
    from originblind.cli import _mock_factory, _openai_factory
    from originblind.run.orchestrator import classify, run_one

    spec = ExperimentSpec.model_validate_json(spec_json)
    it = PlanItem(**item)
    factory = _mock_factory if os.environ.get("ORIGIN_BLIND_BACKEND") == "mock" else _openai_factory
    try:
        backend = factory(it.model_served)
        rec, tr = run_one(spec, it, attempt, namespace, backend, driver_factory=_driver_factory,
                          twin_state=_W["state"], probes=None, use_probe=False, model_thinking=thinking)
        rec["probe"] = probe
        return rec, tr
    except Exception as e:  # noqa: BLE001
        return ({"episode_id": it.episode_id, "status": classify(e), "attempt": attempt,
                 "namespace": namespace, "error": repr(e)[:500],
                 "traceback": traceback.format_exc()[-2000:]}, None)


def run_items_browser(spec: ExperimentSpec, todo: list[PlanItem], counts: dict[str, int], out_path: Path,
                      backend_factory, *, namespace: str, probe: bool, workers: int,
                      max_consecutive_infra: int, health_check, transcripts: bool,
                      model_thinking: dict[str, bool]) -> dict:
    from dataclasses import asdict

    from originblind.lab.pki import write_pki
    from originblind.run.orchestrator import ProbeCache, compute_probe
    from originblind.agent.probe import probe_key
    from originblind.run.episode import build_site
    from originblind.types import OriginClass, SecretKind
    from originblind.llm.client import preset_for

    stats = {"run": 0, "infra_error": 0, "harness_error": 0, "excluded": 0, "aborted": False}
    headful = spec.driver == "chromium-headful"
    if headful and os.environ.get("ORIGIN_BLIND_IN_LAB") != "1":
        raise RuntimeError("driver chromium-headful runs only inside the lab container (Xvfb + NSS-trusted CA)")
    window = (1280, 960)

    # one PKI for the whole plan's host set; in the lab, the CA is trusted via NSS by the entrypoint
    pki_dir = Path(os.environ.get("ORIGIN_BLIND_PKI_DIR") or tempfile.mkdtemp(prefix="ob-pki-"))
    if not (pki_dir / "twin.crt").exists():
        write_pki(pki_dir, plan_hosts(spec, todo), wildcards=infra_wildcards())
    if headful:
        from originblind.browser.capture import trust_ca

        trust_ca(pki_dir / "ca.pem")

    # probes: once per key, in the parent, before dispatch
    probe_by_item: dict[str, dict | None] = {}
    if probe:
        cache = ProbeCache(out_path.parent / "probes.jsonl")

        def do_probe(it: PlanItem):
            if it.arm in SCREEN_ARMS:
                return it.episode_id, None
            key = probe_key(it.model_key, it.brand, it.flow, it.origin_class, it.semantics, it.arm, it.replicate)
            site, _ = build_site(spec.master_seed, brand=it.brand, flow=it.flow, oc=OriginClass(it.origin_class),
                                 secret=SecretKind(it.secret), semantics=it.semantics, replicate=it.replicate,
                                 eid=it.episode_id, environment=spec.environment)
            backend = backend_factory(it.model_served)
            val = cache.get_or_compute(key, lambda: compute_probe(
                spec, it, site, backend, preset_for(it.model_key, thinking=model_thinking.get(it.model_key, False))))
            return it.episode_id, val

        with ThreadPoolExecutor(max_workers=max(1, min(8, workers * 2))) as ex:
            for eid, val in ex.map(do_probe, todo):
                probe_by_item[eid] = val

    cfg = {"driver": spec.driver, "crt": str(pki_dir / "twin.crt"), "key": str(pki_dir / "twin.key"),
           "vllm_url": os.environ.get("ORIGIN_BLIND_VLLM_URL", "http://localhost:8000/v1"),
           "window": window, "per_worker_display": headful and per_worker_displays()}
    # one shared display serializes the lab to one browser; private displays allow concurrency
    n_workers = max(1, workers) if (not headful or per_worker_displays()) else 1
    spec_json = spec.model_dump_json()
    tpath = out_path.parent / "transcripts.jsonl.gz"
    consecutive_infra = 0
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx, initializer=_worker_init,
                             initargs=(cfg,)) as ex:
        futs = {}
        for it in todo:
            futs[ex.submit(_worker_run, spec_json, asdict(it), counts.get(it.episode_id, 0), namespace,
                           probe_by_item.get(it.episode_id), model_thinking.get(it.model_key, False))] = it
        for f in as_completed(futs):
            try:
                rec, tr = f.result()
            except Exception as e:  # noqa: BLE001 - a crashed worker process
                it = futs[f]
                rec, tr = ({"episode_id": it.episode_id, "status": "harness_error",
                            "attempt": counts.get(it.episode_id, 0), "namespace": namespace,
                            "error": f"worker crashed: {e!r}"[:500]}, None)
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
                        if not (health_check() if health_check else False):
                            stats["aborted"] = True
                            for g in futs:
                                g.cancel()
                            break
                        consecutive_infra = 0
    return stats
