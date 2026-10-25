"""The pre-serve gate: a vLLM server starts only on a quiet GPU, a settled host, and with headroom.

Both host hangs of 2026-10-14 began during the weight load of a server started about 2 s after
the previous one stopped (README.md). Before every `docker run` of a server the gate:

  1. quiesce  polls nvidia-smi until the VRAM used by others is <= gpu_idle_max_mib and the free
              VRAM covers what vLLM will request (its --gpu-memory-utilization x total), steadily
              for gpu_quiesce_stable_s; it fails closed when the GPU cannot be measured;
  2. settle   waits until settle_s have passed since the previous server stopped
              (<ops dir>/last_server_stop.json, written by serving.vllm._remove);
  3. drops the VM page cache (best effort; one kernel serves every WSL2 distro and Docker Desktop);
  4. requires VM MemAvailable and Windows available memory above budget, and Windows *commit*
     headroom for the server: every VRAM allocation of a WSL CUDA process is charged to the Windows
     commit limit, next to the VM's own memory, so a server needs about
     util x VRAM + its weights + host_commit_margin_gb (measured on 2026-10-15: a fresh 4B start
     took 46.9 GB and left 2 GB of a 73.8 GB limit). The host checks are skipped, with a recorded
     note, when the Windows probe is unavailable, unless require_host_probe is set (run_full.sh sets
     it for `full`): then a missing probe is waited for like any other headroom and finally refused.

A STOP file (the monitor's critical alarm) or any refusal ends the gate without starting anything.
Every side effect is injectable for tests.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from originblind.ops import probes
from originblind.ops.budget import GIB, Budget

STOP_STATE = "last_server_stop.json"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def drop_page_cache(timeout: float = 180) -> tuple[bool, str]:
    """`sync; echo 1 > drop_caches` in a privileged busybox (sudo needs a password on this host)."""
    try:
        r = subprocess.run([probes.docker_bin(), "run", "--rm", "--privileged", "busybox:1.37.0@sha256:bdf57e528e45e4433820e045b29b4597825a1c9e38353532d90a01445013f82e", "sh", "-c",
                            "sync && echo 1 > /proc/sys/vm/drop_caches"], capture_output=True, text=True,
                           timeout=timeout, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, repr(e)[:200]
    return r.returncode == 0, (r.stderr or "").strip()[:200]


def write_stop_state(state_dir: Path, *, key: str, container: str, epoch: float | None = None) -> None:
    """Record when a server stopped (the settle reference). Atomic; never raises."""
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        epoch = time.time() if epoch is None else epoch
        tmp = state_dir / (STOP_STATE + ".tmp")
        tmp.write_text(json.dumps({"epoch": epoch, "utc": datetime.fromtimestamp(epoch, timezone.utc)
                                   .strftime("%Y-%m-%dT%H:%M:%SZ"), "key": key, "container": container}))
        tmp.replace(state_dir / STOP_STATE)
    except OSError:
        pass


def last_stop_epoch(state_dir: Path) -> float | None:
    try:
        return float(json.loads((state_dir / STOP_STATE).read_text())["epoch"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


@dataclass
class GateResult:
    ok: bool
    reason: str
    record: dict = field(default_factory=dict)


class Gate:
    def __init__(self, budget: Budget, *, state_dir: Path, stop_file: Path | None = None,
                 gpu: Callable[[], dict | None] = probes.gpu_query,
                 vm: Callable[[], dict | None] = probes.vm_memory,
                 host: Callable[[], dict | None] = probes.host_probe,
                 drop: Callable[[], tuple[bool, str]] = drop_page_cache,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic,
                 now: Callable[[], float] = time.time,
                 log: Callable[[str], None] = lambda m: print(m, file=sys.stderr, flush=True)):
        self.b, self.state_dir, self.stop_file = budget, state_dir, stop_file
        self.gpu, self.vm, self.host, self.drop = gpu, vm, host, drop
        self.sleep, self.clock, self.now, self.log = sleep, clock, now, log

    # ------------------------------------------------------------ helpers
    def _stopped(self) -> str | None:
        if self.stop_file is not None and self.stop_file.exists():
            try:
                why = json.loads(self.stop_file.read_text()).get("reason", "")
            except (OSError, ValueError, AttributeError):
                why = ""
            return f"STOP file present ({self.stop_file}): {why}".rstrip(": ")
        return None

    def _wait(self, seconds: float) -> str | None:
        """Sleep in poll-sized chunks; returns a STOP reason if one appears."""
        end = self.clock() + seconds
        while True:
            stop = self._stopped()
            if stop:
                return stop
            left = end - self.clock()
            if left <= 0:
                return None
            self.sleep(min(left, self.b.gpu_poll_s))

    # ------------------------------------------------------------ the gate
    def run(self, *, util: float, key: str, weights_bytes: int | None = None) -> GateResult:
        b = self.b
        rec: dict = {"utc": _utc(), "key": key, "gpu_memory_utilization": util, "budget": asdict(b), "steps": {}}

        def done(ok: bool, reason: str) -> GateResult:
            rec["ok"], rec["reason"], rec["finished_utc"] = ok, reason, _utc()
            return GateResult(ok, reason, rec)

        stop = self._stopped()
        if stop:
            return done(False, stop)

        # 1 quiesce
        t0, quiet_since, misses, last = self.clock(), None, 0, None
        self.log(f"gate: waiting for a quiet GPU (used <= {b.gpu_idle_max_mib} MiB, free >= {util:g} x total, "
                 f"{b.gpu_quiesce_stable_s:.0f} s steady)")
        while True:
            g = self.gpu()
            if g is None:
                misses += 1
                if misses >= 3:
                    rec["steps"]["quiesce"] = {"last": last, "waited_s": round(self.clock() - t0, 1)}
                    return done(False, "the GPU cannot be measured (nvidia-smi failed 3 times); refusing to start")
            else:
                misses, last = 0, g
                need = util * g["total_mib"]
                quiet = g["used_mib"] <= b.gpu_idle_max_mib and (g["free_mib"] or 0) >= need
                if quiet:
                    quiet_since = self.clock() if quiet_since is None else quiet_since
                    if self.clock() - quiet_since >= b.gpu_quiesce_stable_s:
                        break
                else:
                    quiet_since = None
            if self.clock() - t0 > b.gpu_quiesce_timeout_s:
                rec["steps"]["quiesce"] = {"last": last, "waited_s": round(self.clock() - t0, 1)}
                return done(False, f"the GPU did not become quiet within {b.gpu_quiesce_timeout_s:.0f} s "
                                   f"(last: {last}); refusing to start a server on a busy GPU")
            stop = self._wait(b.gpu_poll_s)
            if stop:
                return done(False, stop)
        rec["steps"]["quiesce"] = {"gpu": last, "waited_s": round(self.clock() - t0, 1),
                                   "slack_mib": round((last["free_mib"] or 0) - util * last["total_mib"])}

        # 2 settle
        stopped = last_stop_epoch(self.state_dir)
        since = None if stopped is None else self.now() - stopped
        remain = 0.0 if since is None else max(0.0, b.settle_s - since)
        rec["steps"]["settle"] = {"since_last_stop_s": None if since is None else round(since, 1),
                                  "waited_s": round(remain, 1)}
        if remain > 0:
            self.log(f"gate: settling {remain:.0f} s (the previous server stopped {since:.0f} s ago; "
                     f"settle_s={b.settle_s:.0f})")
            stop = self._wait(remain)
            if stop:
                return done(False, stop)

        # 3 page cache
        before = self.vm()
        if b.drop_caches:
            ok, err = self.drop()
            after = self.vm()
            rec["steps"]["drop_caches"] = {"ok": ok, "error": err or None,
                                           "available_before": (before or {}).get("available"),
                                           "available_after": (after or {}).get("available"),
                                           "cached_before": (before or {}).get("cached"),
                                           "cached_after": (after or {}).get("cached")}
            if not ok:
                self.log(f"gate: dropping the page cache failed ({err}); continuing")
        else:
            rec["steps"]["drop_caches"] = {"skipped": True}

        # 4 memory headroom (VM always; the Windows host when the probe works)
        vram = (last.get("total_mib") or 0) * 1024 * 1024
        need_commit = probes.commit_needed(util, vram, weights_bytes, b.host_commit_margin_gb)
        t1 = self.clock()
        while True:
            vm, host = self.vm(), self.host()
            vm_ok = vm is not None and vm["available"] >= b.vm_min_available_gib * GIB
            host_avail = (host or {}).get("available")
            host_ok = host_avail is None or host_avail >= b.host_min_available_gb * 1e9
            head = probes.commit_headroom(host)
            commit_ok = head is None or head >= need_commit
            # a probe that answers without the commit counters cannot vouch for the headroom either
            probe_ok = not b.require_host_probe or head is not None
            rec["steps"]["memory"] = {"vm_available": (vm or {}).get("available"), "host_available": host_avail,
                                      "host_gpu_shared": (host or {}).get("gpu_shared"),
                                      "commit_headroom": head, "commit_needed": need_commit,
                                      "weights_bytes": weights_bytes, "host_probe": host is not None,
                                      "host_probe_required": b.require_host_probe}
            if vm_ok and host_ok and commit_ok and probe_ok:
                break
            if self.clock() - t1 > b.gpu_quiesce_timeout_s:
                if not probe_ok:
                    return done(False, "the Windows host probe gave no commit counters, so the commit headroom "
                                       "cannot be checked; refusing to start (require_host_probe; check "
                                       "powershell.exe interop, README.md)")
                if not commit_ok:
                    return done(False, f"Windows commit headroom {head / 1e9:.1f} GB < {need_commit / 1e9:.1f} GB this "
                                       f"server needs ({util:g} x {vram / 1e9:.1f} GB VRAM + "
                                       f"{(weights_bytes or 25e9) / 1e9:.1f} GB weights + {b.host_commit_margin_gb:g} GB): "
                                       "a larger pagefile is required (README.md)")
                return done(False, "not enough memory headroom for a weight load "
                                   f"(VM available {(vm or {}).get('available', 0) / GIB:.1f} GiB, need "
                                   f"{b.vm_min_available_gib:g}; host available "
                                   f"{'n/a' if host_avail is None else f'{host_avail / 1e9:.1f} GB'}, need "
                                   f"{b.host_min_available_gb:g} GB)")
            self.log("gate: waiting for the Windows host probe" if not probe_ok else "gate: waiting for memory headroom")
            stop = self._wait(10.0)
            if stop:
                return done(False, stop)
        if host is None:
            self.log("gate: Windows host probe unavailable; host memory not checked")
        return done(True, "ok")
