"""Read-only, config-aware preflight: enumerate named execution blockers before any GPU time is spent.

Each check returns ok | WARN | BLOCKED with a detail line. BLOCKED fails the preflight (exit 1).
`--smoke-only` skips GPU/serving checks (the GPU-free smoke needs only Chromium); `--full` (the
confirmatory namespace) additionally blocks on an unfrozen pre-registration, unmaterialized weights
and a dirty git tree.

B08/B08b require the images pinned by digest (scripts/00_setup.sh pulls the digests and tags them;
the run manifests record the tags' RepoDigests).

Host safety (README.md; B15-B21, B27, B28) runs with `--full` or `--host-checks`: it blocks in
full mode, warns otherwise, and `OB_ACCEPT_HOST_RISK=1` downgrades the host hazards (B15, B16, B19,
B20, B27 Windows commit headroom, B28 local LLM servers) to warnings, recorded under "overrides".
With `--namespace` the namespace codes (B22-B26) check a pending STOP alarm, resume identity, store
integrity, lab results that exist only in the lab volume (B25), and the serving stack of earlier
invocations.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

OK, WARN, BLOCKED = "ok", "WARN", "BLOCKED"


def _run(cmd: list[str], timeout: float = 30) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return subprocess.CompletedProcess(cmd, 127, "", f"{cmd[0]}: not found")
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, "", f"{cmd[0]}: timed out")


def _pinned(image: str, digest: str) -> tuple[bool | None, str]:
    """Is the local `repo:tag` the pinned build? True when its .Id (the containerd image store reports
    the index digest) or a RepoDigests entry is `digest`; False (with its id) for another build; None
    when the tag is missing."""
    r = _run(["docker", "image", "inspect", "-f", '{{.Id}} {{join .RepoDigests " "}}', image])
    if r.returncode != 0:
        return None, "missing"
    ids = r.stdout.split()
    repo = image.rsplit(":", 1)[0]
    return (digest in ids or f"{repo}@{digest}" in ids), (ids[0] if ids else "?")


def lab_base(repo_root: Path) -> tuple[str, str] | None:
    """(repo:tag, sha256:digest) of docker/lab.Dockerfile's FROM line."""
    try:
        m = re.search(r"^FROM\s+(\S+?:[^\s@]+)@(sha256:[0-9a-f]{64})", (repo_root / "docker" / "lab.Dockerfile")
                      .read_text(), re.M)
    except OSError:
        return None
    return (m.group(1), m.group(2)) if m else None


def _configs(repo_root: Path, configs: list[str] | None) -> list[Path]:
    if configs:
        return [Path(c) if Path(c).is_absolute() else repo_root / c for c in configs]
    return sorted(p for p in (repo_root / "configs").glob("*.yaml") if p.name not in ("prereg.yaml", "roster.yaml"))


def check(repo_root: Path, configs: list[str] | None = None, *, smoke_only: bool = False,
          full: bool = False, namespace: str | None = None, results: Path | None = None,
          host_checks: bool = False, probe: dict | None = None) -> dict:
    res: dict[str, dict] = {}

    def add(code: str, status: str, detail: str) -> None:
        res[code] = {"status": status, "detail": detail}

    # B01-B02 docker
    has_docker = bool(shutil.which("docker"))
    add("B01_docker", OK if has_docker else BLOCKED, "docker CLI on PATH" if has_docker else "docker not found")
    daemon = has_docker and _run(["docker", "info"], 20).returncode == 0
    add("B02_docker_daemon", OK if daemon else BLOCKED, "docker daemon reachable" if daemon else "docker daemon unreachable")

    # B03 GPU through docker (not needed for the GPU-free smoke)
    if smoke_only:
        add("B03_docker_gpu", OK, "skipped (smoke only)")
    else:
        gpu = daemon and _run(["docker", "run", "--rm", "--gpus", "all", "busybox:1.37.0@sha256:bdf57e528e45e4433820e045b29b4597825a1c9e38353532d90a01445013f82e", "true"], 90).returncode == 0
        add("B03_docker_gpu", OK if gpu else BLOCKED, "docker --gpus all works" if gpu else "docker cannot use the GPU")

    # B04 lockfile
    lock = (repo_root / "poetry.lock").exists()
    add("B04_lock", OK if lock else BLOCKED, "poetry.lock present" if lock else "poetry.lock missing")
    if shutil.which("poetry"):
        r = _run(["poetry", "check", "--lock"], 60)
        add("B04b_lock_consistent", OK if r.returncode == 0 else WARN,
            "poetry.lock consistent with pyproject" if r.returncode == 0 else (r.stdout + r.stderr).strip()[:200])
    else:
        add("B04b_lock_consistent", WARN, "poetry not on PATH; lock consistency not checked")

    # B05 host Chromium
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            b = p.chromium.launch()
            version = b.version
            b.close()
        add("B05_chromium", OK, f"Playwright Chromium {version} launches")
    except Exception as e:  # noqa: BLE001
        add("B05_chromium", BLOCKED, f"Playwright Chromium does not launch: {str(e)[:160]} "
                                     "(fix: python -m playwright install chromium)")

    # B06 offline PSL
    try:
        from originblind.oracle.attribute import registrable_domain

        ok = (registrable_domain("a.b.example.co.uk") == "example.co.uk"
              and registrable_domain("pay.pspworld.test") == "pspworld.test")
        add("B06_psl", OK if ok else BLOCKED, "offline PSL attribution works (incl. lab TLDs)" if ok else "PSL attribution wrong")
    except Exception as e:  # noqa: BLE001
        add("B06_psl", BLOCKED, f"PSL attribution failed: {e!r}"[:200])

    # B07 configs + prereg
    cfgs = _configs(repo_root, configs)
    bad, models, headful = [], {}, False
    from originblind.config import load_spec

    for c in cfgs:
        try:
            spec = load_spec(c)
            headful = headful or spec.driver == "chromium-headful"
            for m in spec.models:
                if not m.hf_repo.startswith("none/"):
                    models.setdefault(m.key, m)
        except Exception as e:  # noqa: BLE001
            bad.append(f"{c.name}: {str(e).splitlines()[0][:120]}")
    add("B07_configs", OK if cfgs and not bad else BLOCKED,
        f"{len(cfgs)} config(s) parse and validate" if not bad else "; ".join(bad))
    try:
        import yaml

        pr = yaml.safe_load((repo_root / "configs" / "prereg.yaml").read_text())
        frozen = bool(pr.get("frozen"))
        # a confirmatory (--full) run starts only from a frozen design; pilots may run against a draft
        status = OK if (frozen or not full) else BLOCKED
        add("B07b_prereg", status, f"prereg v{pr.get('version')} parses; frozen={frozen}"
            + ("" if frozen or not full else " (freeze it before the confirmatory run: README.md)"))
    except Exception as e:  # noqa: BLE001
        add("B07b_prereg", BLOCKED, f"configs/prereg.yaml unreadable: {e!r}"[:200])

    # B08 vLLM image, B08b lab base image (both pinned by digest), B09 weights
    if smoke_only:
        add("B08_vllm_image", OK, "skipped (smoke only)")
        add("B08b_base_image", OK, "skipped (smoke only)")
        add("B09_weights", OK, "skipped (smoke only)")
    else:
        from originblind.serving.vllm import IMAGE, IMAGE_DIGEST

        base = lab_base(repo_root)
        for code, ref in (("B08_vllm_image", (IMAGE, IMAGE_DIGEST)), ("B08b_base_image", base)):
            if ref is None:
                add(code, BLOCKED, "docker/lab.Dockerfile has no FROM repo:tag@sha256:digest line")
                continue
            img, dig = ref
            pinned, have = _pinned(img, dig) if daemon else (None, "docker daemon unreachable")
            if pinned:
                add(code, OK, f"{img} pinned at {dig[:19]}")
            elif pinned is None:  # no server without vLLM; without the base tag only provenance suffers
                add(code, BLOCKED if (full or code == "B08_vllm_image") else WARN,
                    f"{img} missing (scripts/00_setup.sh pulls {dig[:19]} and tags it)")
            else:
                add(code, BLOCKED if full else WARN,
                    f"{img} is {have[:19]}, not the pinned {dig[:19]}: rerun scripts/00_setup.sh, which pulls the "
                    "digest and retags it (the run manifests record the tag's digest)")
        not_ready = []
        if daemon:
            from originblind.serving.models import volume_manifest

            groups: dict[tuple, str] = {}
            for m in models.values():
                groups.setdefault((m.hf_repo, m.revision), m.key)
            for (repo, rev), key in groups.items():
                vm = volume_manifest(key)
                if not vm or vm.get("revision") != rev or vm.get("repo") != repo:
                    not_ready.append(f"{key}@{str(rev)[:12]}")
        sev = BLOCKED if (full and not_ready) else (WARN if not_ready else OK)
        add("B09_weights", sev, "all selected models materialized at their pinned revision" if not not_ready
            else f"not materialized: {', '.join(not_ready)} (run_full.sh fetches and materializes them)")

    # B10 lab image for screenshot arms
    if headful:
        lab = daemon and _run(["docker", "image", "inspect", "originblind-lab:latest"]).returncode == 0
        add("B10_lab_image", OK if lab else WARN, "originblind-lab:latest present" if lab
            else "lab image missing (00_setup.sh builds it)")
    else:
        add("B10_lab_image", OK, "not needed by the selected configs")

    # B11 free VRAM, B12 port
    if smoke_only:
        add("B11_vram", OK, "skipped (smoke only)")
        add("B12_port", OK, "skipped (smoke only)")
    else:
        r = _run(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"], 20)
        if r.returncode == 0 and r.stdout.strip():
            total, used = (int(x) for x in r.stdout.strip().splitlines()[0].split(","))
            ours = _run(["docker", "inspect", "-f", "{{.State.Running}}", "ob-vllm"]).stdout.strip() == "true"
            free = total - used
            sev = OK if (free >= 20_000 or ours) else WARN
            add("B11_vram", sev, f"{free} MiB free of {total} MiB" + (" (ob-vllm running)" if ours else ""))
        else:
            add("B11_vram", BLOCKED, "nvidia-smi unavailable")
        import socket

        s = socket.socket()
        busy = s.connect_ex(("127.0.0.1", 8000)) == 0
        s.close()
        ours = _run(["docker", "inspect", "-f", "{{.State.Running}}", "ob-vllm"]).stdout.strip() == "true"
        add("B12_port", OK if (not busy or ours) else BLOCKED,
            "port 8000 free" if not busy else ("port 8000 held by ob-vllm" if ours else "port 8000 in use by another process"))

    # B13 disk
    free_gb = shutil.disk_usage(repo_root).free / 1e9
    add("B13_disk", OK if free_gb >= 50 else (WARN if free_gb >= 20 else BLOCKED), f"{free_gb:.0f} GB free")

    # B14 clean tree (provenance)
    r = _run(["git", "-C", str(repo_root), "status", "--porcelain"], 30)
    dirty = r.returncode != 0 or bool(r.stdout.strip())
    allow = os.environ.get("ORIGIN_BLIND_ALLOW_DIRTY") == "1"
    if full and dirty and not allow:
        add("B14_clean_tree", BLOCKED, "git tree is dirty: commit before a confirmatory run (provenance)")
    else:
        add("B14_clean_tree", WARN if dirty else OK, "git tree dirty" if dirty else "git tree clean")

    extra: dict = {}
    if not smoke_only and (full or host_checks):
        hc = host_safety(models.values(), full=full, probe=probe)
        res.update(hc["blockers"])
        extra["overrides"] = hc["overrides"]
        extra["host"] = hc["host"]
    if not smoke_only and namespace:
        res.update(namespace_checks(cfgs, namespace, Path(results or repo_root / "results"), full=full,
                                    probe=probe))

    n_blocked = sum(1 for v in res.values() if v["status"] == BLOCKED)
    n_warn = sum(1 for v in res.values() if v["status"] == WARN)
    return {"blockers": res, "n_blocked": n_blocked, "n_warn": n_warn, "ready": n_blocked == 0,
            "mode": "smoke" if smoke_only else ("full" if full else "pilot"),
            "configs": [str(c) for c in cfgs], **extra}


# ---------------------------------------------------------------- host safety (B15-B21)

_OTHER_VM = re.compile(r"^(vmmem(?!wsl)|vmwp|cowork|codex|virtualbox|vboxheadless|vmware)", re.IGNORECASE)


def _util_of(m) -> float:
    """The --gpu-memory-utilization vLLM applies for a model (roster vllm_args win over 0.90)."""
    from originblind.serving.vllm import gpu_utilization

    return gpu_utilization(["--gpu-memory-utilization", "0.9", *list(m.vllm_args or [])])


def _hours_left(expiry) -> float | None:
    if not expiry:
        return None
    try:
        t = datetime.fromisoformat(str(expiry).replace("Z", "+00:00"))
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return (t - datetime.now(timezone.utc)).total_seconds() / 3600


def host_safety(models=(), *, full: bool, probe: dict | None = None) -> dict:
    """B15-B21, B27, B28: the host conditions under which a confirmatory run may start (README.md).
    `probe` injects {"host", "gpu", "vm", "ob_vllm_running"} (tests); absent keys are measured."""
    from originblind.ops import probes
    from originblind.ops.budget import GIB, load_budget

    b = load_budget()
    accept = os.environ.get("OB_ACCEPT_HOST_RISK") == "1"
    probe = probe or {}
    host = probe["host"] if "host" in probe else probes.host_probe()
    gpu = probe["gpu"] if "gpu" in probe else probes.gpu_query()
    vm = probe["vm"] if "vm" in probe else probes.vm_memory()
    res: dict[str, dict] = {}

    def add(code: str, status: str, detail: str, hazard: bool = False) -> None:
        if hazard and status == BLOCKED and not full:
            status = WARN
        if hazard and status == BLOCKED and accept:
            status, detail = WARN, detail + " [accepted: OB_ACCEPT_HOST_RISK=1]"
        res[code] = {"status": status, "detail": detail}

    if host is None:
        for code in ("B15_wsl_memory", "B16_host_memory", "B19_windows_update", "B20_windows_drives",
                     "B27_commit_headroom", "B28_local_llm"):
            add(code, WARN, "cannot verify: no Windows interop (powershell.exe unavailable)")
    else:
        total = host.get("total_phys") or 0
        mem = (vm or {}).get("total") or 0
        frac = mem / total if total else 1.0
        ok = bool(total) and frac <= b.max_wsl_memory_fraction
        add("B15_wsl_memory", OK if ok else BLOCKED,
            f"WSL MemTotal {mem / GIB:.1f} GiB = {frac:.0%} of host RAM {total / GIB:.1f} GiB"
            + ("" if ok else f" (> {b.max_wsl_memory_fraction:.0%}: Windows, Docker Desktop and the GPU driver "
               "keep too little; set memory=32GB, swap=8GB in .wslconfig, then `wsl --shutdown`)"), hazard=True)
        avail = host.get("available")
        ok = avail is not None and avail >= b.host_min_available_gb * 1e9
        add("B16_host_memory", OK if ok else BLOCKED,
            f"Windows available memory {0 if avail is None else avail / 1e9:.1f} GB"
            + ("" if ok else f" (< {b.host_min_available_gb:g} GB: close other applications)"), hazard=True)
        left = _hours_left(host.get("pause_updates_expiry"))
        ok = left is not None and left >= b.update_pause_min_h
        add("B19_windows_update", OK if ok else BLOCKED,
            (f"Windows Update paused for another {left:.0f} h" if left is not None else "Windows Update is not paused")
            + ("" if ok else f" (it restarted this host twice on 2026-10-14; pause it for >= {b.update_pause_min_h:g} h:"
               " Settings > Windows Update > Pause updates)"), hazard=True)
        drives = host.get("drives") or {}
        if not drives:
            add("B20_windows_drives", WARN, "no fixed Windows drive reported")
        else:
            d, v = min(drives.items(), key=lambda kv: kv[1]["free"])
            gb = v["free"] / 1e9
            st = OK if gb >= b.windows_drive_warn_gb else (WARN if gb >= b.windows_drive_block_gb else BLOCKED)
            add("B20_windows_drives", st,
                ", ".join(f"{k} {x['free'] / 1e9:.0f} GB free" for k, x in sorted(drives.items()))
                + ("" if st == OK else f" ({d} is low; it may hold the WSL/Docker disks and the pagefile: free "
                   "space or move them, README.md)"), hazard=True)
        head = probes.commit_headroom(host)
        if head is None or gpu is None:
            add("B27_commit_headroom", WARN, "cannot verify the Windows commit headroom (no counters or no GPU)")
        else:
            needs = _commit_needs(models, gpu["total_mib"] * 1024 * 1024, b, probe.get("checkpoints"))
            wk, need = max(needs.items(), key=lambda kv: kv[1]) if needs else ("a server", probes.commit_needed(
                0.9, gpu["total_mib"] * 1024 * 1024, None, b.host_commit_margin_gb))
            pf = ", ".join(f"{n} {sz / 1e9:.1f} GB" for n, sz in host.get("pagefiles") or []) or "unknown"
            ok = head >= need
            rec_gb = max(64, int((need - head) / 1e9 + 32) // 16 * 16)
            add("B27_commit_headroom", OK if ok else BLOCKED,
                f"Windows commit headroom {head / 1e9:.1f} GB (limit {host['commit_limit'] / 1e9:.1f}, committed "
                f"{host['committed'] / 1e9:.1f}; pagefile {pf}{', system-managed' if host.get('pagefile_auto') else ''}); "
                f"the largest server ({wk}) needs {need / 1e9:.1f} GB"
                + ("" if ok else f": every VRAM allocation is charged to the commit limit, and exhausting it is the "
                   f"leading explanation of the 2026-10-14 hangs. Set a fixed pagefile of >= {rec_gb} GB on a drive with "
                   "space (System Properties > Performance > Virtual memory; e.g. D:), restart Windows "
                   "(README.md)"), hazard=True)
        others = [(n, ws) for n, ws in host.get("procs") or [] if _OTHER_VM.match(n) and ws > b.other_vm_warn_gb * 1e9]
        add("B21_other_vms", WARN if others else OK,
            ("other VMs/sandboxes hold host memory: " + ", ".join(f"{n} {ws / 1e9:.1f} GB" for n, ws in others))
            if others else f"no other VM or sandbox holds more than {b.other_vm_warn_gb:g} GB")
        llm = sorted({n for n, _ in host.get("llm_servers") or []})
        add("B28_local_llm", BLOCKED if llm else OK,
            (f"local LLM server(s) running: {', '.join(llm)}. Idle they hold no VRAM, but any client can make them "
             "load a model onto the GPU vLLM uses, mid-run; quit them for the run (Ollama: tray icon > Quit; it "
             "starts again at login)") if llm else "no local LLM server (Ollama, LM Studio, llama-server) running",
            hazard=True)

    if gpu is None:
        add("B17_gpu_idle", BLOCKED if full else WARN, "nvidia-smi unavailable: the GPU cannot be measured")
    else:
        ours = probe["ob_vllm_running"] if "ob_vllm_running" in probe else (
            _run(["docker", "inspect", "-f", "{{.State.Running}}", "ob-vllm"]).stdout.strip() == "true")
        utils = sorted({_util_of(m) for m in models}) or [0.90]
        slack = {f"{u:g}": round((gpu["free_mib"] or 0) - u * gpu["total_mib"]) for u in utils}
        idle_ok = gpu["used_mib"] <= b.gpu_idle_max_mib
        st = OK if (idle_ok or ours) else (BLOCKED if full else WARN)
        detail = f"{gpu['used_mib']:.0f} MiB used of {gpu['total_mib']:.0f}"
        if ours and not idle_ok:
            detail += " (held by a leftover ob-vllm, released before the first start)"
        elif not idle_ok:
            detail += f" (> {b.gpu_idle_max_mib} MiB used by other processes: close GPU applications)"
        else:
            detail += f"; free-VRAM slack per --gpu-memory-utilization: {slack} MiB"
            if min(slack.values()) < 1024:
                st, detail = WARN, detail + " (< 1 GiB: keep the Windows desktop idle during the run)"
        add("B17_gpu_idle", st, detail)
        att = gpu.get("display_attached")
        add("B18_display_gpu", WARN if att else OK,
            "a monitor is attached to the compute GPU: Windows holds 1-1.6 GiB of it and the desktop competes "
            "with vLLM; keep the desktop idle, ideally connect the monitor to the iGPU" if att
            else "no display on the compute GPU")
    return {"blockers": res, "overrides": {"OB_ACCEPT_HOST_RISK": accept}, "host": _host_summary(host, vm, gpu)}


def _commit_needs(models, vram_bytes: float, b, checkpoints: dict | None) -> dict[str, float]:
    """Commit headroom each scheduled server start needs, by weights key (ops/probes.commit_needed)."""
    from originblind.ops import probes
    from originblind.serving import vllm
    from originblind.serving.models import volume_manifest

    mlist = [m for m in models if vllm.servable(m)]
    out: dict[str, float] = {}
    for m in mlist:
        try:
            plan = vllm.serve_plan(m.key, mlist)
        except ValueError:
            continue
        if plan.weights_key in out:
            continue
        wb = checkpoints.get(plan.weights_key) if checkpoints is not None else probes.weights_bytes(
            volume_manifest(plan.weights_key))
        out[plan.weights_key] = probes.commit_needed(vllm.gpu_utilization(vllm.docker_run_argv(plan)), vram_bytes,
                                                     wb, b.host_commit_margin_gb)
    return out


def _host_summary(host: dict | None, vm: dict | None, gpu: dict | None) -> dict | None:
    if host is None and vm is None and gpu is None:
        return None
    h = dict(host or {})
    h.pop("procs", None)
    if isinstance(h.get("wslconfig"), dict):
        h["wslconfig"] = h["wslconfig"].get("raw")
    return {"windows": h or None, "vm": vm, "gpu": gpu}


# ---------------------------------------------------------------- namespace (B22-B26)

def lab_volume_experiments(namespace: str, volume: str = "ob-results", timeout: float = 120) -> set[str] | None:
    """Experiments of `namespace` that hold an episodes.jsonl in the lab results volume, or None when
    the volume cannot be listed (a sentinel line tells "nothing there" from a failed docker run)."""
    script = 'for d in /r/"$0"/*/; do [ -f "$d/episodes.jsonl" ] && basename "$d"; done; echo __ob_listed__'
    r = _run(["docker", "run", "--rm", "-v", f"{volume}:/r:ro", "busybox:1.37.0@sha256:bdf57e528e45e4433820e045b29b4597825a1c9e38353532d90a01445013f82e", "sh", "-c", script, namespace],
             timeout)
    lines = r.stdout.split()
    if r.returncode != 0 or not lines or lines[-1] != "__ob_listed__":
        return None
    return set(lines[:-1])


def namespace_checks(cfgs: list[Path], namespace: str, results: Path, *, full: bool,
                     probe: dict | None = None) -> dict[str, dict]:
    """B22 STOP alarm, B23 resume identity, B24 store integrity, B25 volume-only lab results, B26
    serving stack. `probe` injects {"lab_volume": set | None, "vllm_digest", "gpu_string"} (tests)."""
    res: dict[str, dict] = {}
    nsdir = Path(results) / namespace

    def add(code: str, status: str, detail: str) -> None:
        res[code] = {"status": status, "detail": detail}

    stop = nsdir / "STOP"
    if stop.exists():
        try:
            why = json.loads(stop.read_text()).get("reason", "")
        except (OSError, ValueError, AttributeError):
            why = "(unreadable)"
        add("B22_stop_alarm", BLOCKED, f"{stop} exists: {why}. Fix the cause, then move it aside: "
                                       f"mv {stop} {nsdir}/logs/STOP_<utc>.json")
    else:
        add("B22_stop_alarm", OK, "no STOP alarm pending")

    from originblind.config import load_spec
    from originblind.run.orchestrator import ResumeMismatch, guard_resume

    mism, checked = [], 0
    for c in cfgs:
        try:
            spec = load_spec(c)
        except Exception:  # noqa: BLE001 - B07 reports it
            continue
        out = nsdir / spec.name / "episodes.jsonl"
        if not out.exists():
            continue
        checked += 1
        try:
            guard_resume(spec, out)
        except ResumeMismatch as e:
            mism.append(f"{spec.name}: {str(e)[:160]}")
        except (ValueError, OSError) as e:  # a torn manifest.json (JSONDecodeError, UnicodeDecodeError)
            mism.append(f"{spec.name}: manifest.json unreadable ({e!r})"[:200])
    add("B23_resume_identity", BLOCKED if mism else OK,
        "; ".join(mism) if mism else f"{checked} existing experiment(s) resume under the current identity")

    from originblind.ops.store import check_namespace

    st = check_namespace(nsdir)
    bad = {k: v["status"] for k, v in st["experiments"].items() if v["status"] != "ok"}
    if any(v == "repair" for v in bad.values()):
        add("B24_store_integrity", BLOCKED, f"stores need a manual repair: {bad} "
            f"(`originblind ops store check {nsdir}`, then `ops store repair <dir>`; README.md)")
    elif bad:
        add("B24_store_integrity", WARN, f"unterminated last line in {sorted(bad)} (newline-terminated "
                                         "automatically before the next append)")
    else:
        add("B24_store_integrity", OK, f"{len(st['experiments'])} store(s) intact")

    # B25: scripts/lab_run.sh resumes from the lab volume when only the volume holds an experiment
    # (it then exports that copy to the host), so a volume copy the host lacks -- e.g. an attempt that
    # was archived on the host only -- would silently continue under this namespace
    lab_exps = []
    for c in cfgs:
        try:
            if load_spec(c).driver == "chromium-headful":
                lab_exps.append(Path(c).stem)  # lab_run.sh stores an experiment under its file stem
        except Exception:  # noqa: BLE001 - B07 reports it
            continue
    if not lab_exps:
        add("B25_lab_volume", OK, "no lab (screenshot) config selected")
    else:
        vol = probe.get("lab_volume") if probe and "lab_volume" in probe else lab_volume_experiments(namespace, volume=os.environ.get("OB_RESULTS_VOLUME", "ob-results"))
        if vol is None:
            add("B25_lab_volume", BLOCKED if full else WARN,
                "cannot list the lab results volume ob-results (docker); a volume-only copy cannot be ruled out")
        else:
            orphans = sorted(e for e in lab_exps if e in vol and not (nsdir / e / "episodes.jsonl").exists())
            if orphans:
                add("B25_lab_volume", BLOCKED if full else WARN,
                    f"the lab volume ob-results holds {namespace}/{{{','.join(orphans)}}} but the host does not: "
                    "scripts/lab_run.sh would resume that copy. If it belongs to an archived attempt, move it "
                    f"aside inside the volume (docker run --rm -v ob-results:/r busybox:1.37.0@sha256:bdf57e528e45e4433820e045b29b4597825a1c9e38353532d90a01445013f82e mv /r/{namespace} "
                    f"/r/{namespace}.attempt<N>-<date>); if the host copy was lost, restore it first "
                    "(README.md)")
            else:
                add("B25_lab_volume", OK, f"no volume-only lab results ({len(lab_exps)} lab config(s) checked)")

    seen_img, seen_gpu = set(), set()
    for m in nsdir.glob("*/manifests/*.json"):
        try:
            d = json.loads(m.read_text())
        except (OSError, ValueError):
            continue
        img = (d.get("docker_images") or {}).get("vllm/vllm-openai:v0.31.0")
        if img:
            seen_img.add(img)
        if d.get("gpu"):
            seen_gpu.add(str(d["gpu"]).strip())
    probe = probe or {}
    if not (seen_img or seen_gpu):
        add("B26_serving_stack", OK, "no earlier invocation recorded a serving stack")
        return res
    cur_img = probe["vllm_digest"] if "vllm_digest" in probe else _run(
        ["docker", "inspect", "--format", "{{index .RepoDigests 0}}", "vllm/vllm-openai:v0.31.0"], 20).stdout.strip()
    smi = __import__("originblind.ops.probes", fromlist=["x"]).nvidia_smi_bin() or "nvidia-smi"
    cur_gpu = probe["gpu_string"] if "gpu_string" in probe else _run(
        [smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], 20).stdout.strip()
    changed = []
    if seen_img and cur_img and cur_img not in seen_img:
        changed.append(f"vLLM image {sorted(seen_img)} -> {cur_img}")
    if seen_gpu and cur_gpu and cur_gpu not in seen_gpu:
        changed.append(f"GPU/driver {sorted(seen_gpu)} -> {cur_gpu}")
    if changed:
        add("B26_serving_stack", BLOCKED if full else WARN,
            "the serving stack changed since earlier invocations of this namespace: " + "; ".join(changed)
            + " (restore it, or record the change in README.md and re-run preflight)")
    else:
        add("B26_serving_stack", OK, f"unchanged: {sorted(seen_img)[:1]} {sorted(seen_gpu)[:1]}")
    return res


def main(repo_root: Path | None = None, configs=None, smoke_only: bool = False, full: bool = False,
         namespace: str | None = None, results: str | None = None, host_checks: bool = False) -> int:
    root = repo_root or Path.cwd()
    res = check(root, configs, smoke_only=smoke_only, full=full, namespace=namespace,
                results=Path(results) if results else None, host_checks=host_checks)
    print(json.dumps(res, indent=2))
    return 0 if res["ready"] else 1
