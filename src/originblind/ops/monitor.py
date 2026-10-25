"""Flight recorder and guard of an unattended run. A sensor: it never stops anything itself.

Every `monitor_vm_s` seconds it appends one JSON line to <rundir>/logs/monitor_<stamp>.jsonl
(flushed and fsynced, so the samples before a host hang survive it): VM memory, swap and PSI,
the kernel's OOM-kill counter, free space under the results directory, the latest GPU sample (one
long-lived `nvidia-smi -lms`), and the latest Windows sample (a bounded PowerShell `Get-Counter`
stream, relaunched when it ends: host available memory, commit, pagefile, the NVIDIA adapter's
dedicated and *shared* memory, i.e. VRAM spilling into host RAM, and free space per drive).

The guard (pure, unit-tested) turns samples into warnings (logged) and criticals. A critical rule
must hold for `guard_samples` consecutive samples; any new kernel OOM kill is critical at once
(a killed Chromium renderer silently changes what an episode observes). On a critical the monitor
writes <rundir>/STOP (reason and sample) and sends SIGUSR1 to scripts/run_full.sh, which stops the
run cleanly (exit 6). It exits when run_full exits.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from originblind.ops import probes
from originblind.ops.budget import GIB, Budget, load_budget

MON_GPU_FIELDS = ("memory.used", "memory.free", "memory.total", "temperature.gpu", "utilization.gpu",
                  "power.draw")
HOST_FRESH_S = 30.0


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Guard:
    """Thresholds of the budget applied to samples; stateful only for the consecutive counters."""

    def __init__(self, b: Budget, oom_base: int | None = None):
        self.b = b
        self.oom_base = oom_base
        self.streak: dict[str, int] = {}

    def _crit(self, name: str, cond: bool) -> bool:
        self.streak[name] = self.streak.get(name, 0) + 1 if cond else 0
        return self.streak[name] >= self.b.guard_samples

    def evaluate(self, s: dict) -> tuple[list[str], str | None]:
        b = self.b
        vm, psi, gpu = s.get("vm") or {}, s.get("psi") or {}, s.get("gpu") or {}
        host = s.get("host") if (s.get("host_age_s") is not None and s["host_age_s"] <= HOST_FRESH_S) else None
        mem_psi = psi.get("memory") or {}
        warn: list[str] = []
        crit: list[str] = []

        oom = s.get("oom_kill")
        if oom is not None:
            if self.oom_base is None:
                self.oom_base = oom
            elif oom > self.oom_base:
                crit.append(f"kernel OOM killer fired {oom - self.oom_base} time(s) during the run")

        avail = vm.get("available")
        if avail is not None:
            if self._crit("vm_available", avail < b.guard_vm_available_gib * GIB):
                crit.append(f"VM MemAvailable {avail / GIB:.1f} GiB < {b.guard_vm_available_gib:g} GiB")
            if avail < b.warn_vm_available_gib * GIB:
                warn.append(f"VM MemAvailable {avail / GIB:.1f} GiB")
        full = (mem_psi.get("full") or {}).get("avg10")
        if full is not None and self._crit("psi_full", full > b.guard_psi_full_avg10):
            crit.append(f"memory pressure (PSI full avg10) {full:.0f} > {b.guard_psi_full_avg10:g}")
        some = (mem_psi.get("some") or {}).get("avg10")
        if some is not None and some > b.warn_psi_some_avg10:
            warn.append(f"memory pressure (PSI some avg10) {some:.0f}")
        swap = vm.get("swap_used")
        if swap is not None and swap > b.warn_swap_used_gib * GIB:
            warn.append(f"VM swap in use {swap / GIB:.1f} GiB")
        rfree = s.get("results_free")
        if rfree is not None and self._crit("results_free", rfree < b.guard_results_free_gb * 1e9):
            crit.append(f"results disk {rfree / 1e9:.1f} GB free < {b.guard_results_free_gb:g} GB")
        temp = gpu.get("temp_c")
        if temp is not None and temp > b.warn_gpu_temp_c:
            warn.append(f"GPU {temp:.0f} °C")

        if host is not None:
            ha = host.get("available")
            if ha is not None:
                if self._crit("host_available", ha < b.guard_host_available_gb * 1e9):
                    crit.append(f"Windows available memory {ha / 1e9:.1f} GB < {b.guard_host_available_gb:g} GB")
                if ha < b.warn_host_available_gb * 1e9:
                    warn.append(f"Windows available memory {ha / 1e9:.1f} GB")
            shared = host.get("gpu_shared")
            if shared is not None:
                mib = shared / (1024 * 1024)
                if b.guard_gpu_shared_mib and self._crit("gpu_shared", mib > b.guard_gpu_shared_mib):
                    crit.append(f"GPU shared (host RAM) usage {mib:.0f} MiB > {b.guard_gpu_shared_mib} MiB")
                if mib > b.warn_gpu_shared_mib:
                    warn.append(f"GPU shared (host RAM) usage {mib:.0f} MiB")
            head = probes.commit_headroom(host)
            if head is not None:
                if self._crit("commit_headroom", head < b.guard_commit_headroom_gb * 1e9):
                    crit.append(f"Windows commit headroom {head / 1e9:.1f} GB < {b.guard_commit_headroom_gb:g} GB")
                if head < b.warn_commit_headroom_gb * 1e9:
                    warn.append(f"Windows commit headroom {head / 1e9:.1f} GB")
            drives = host.get("drives_free") or {}
            low = {d: f for d, f in drives.items() if f is not None}
            worst = min(low.items(), key=lambda kv: kv[1]) if low else None
            if worst is not None:
                if self._crit("drive_free", worst[1] < b.guard_windows_drive_free_gb * 1e9):
                    crit.append(f"Windows drive {worst[0]} {worst[1] / 1e9:.1f} GB free "
                                f"< {b.guard_windows_drive_free_gb:g} GB")
                if worst[1] < b.warn_windows_drive_free_gb * 1e9:
                    warn.append(f"Windows drive {worst[0]} {worst[1] / 1e9:.1f} GB free")
        return warn, ("; ".join(crit) if crit else None)


class StreamReader:
    """Keeps the latest parsed line of a long-running command; relaunches it when it ends."""

    def __init__(self, argv: list[str] | None, parse, *, cwd: str | None = None, backoff_s: float = 2.0):
        self.argv, self.parse, self.cwd, self.backoff_s = argv, parse, cwd, backoff_s
        self.latest = None
        self.at: float | None = None
        self.proc: subprocess.Popen | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> "StreamReader":
        if self.argv:
            self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.proc = subprocess.Popen(self.argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                             stdin=subprocess.DEVNULL, cwd=self.cwd)
                for raw in self.proc.stdout:
                    if self._stop.is_set():
                        break
                    v = self.parse(raw.decode("utf-8", errors="replace").strip())
                    if v is not None:
                        self.latest, self.at = v, time.monotonic()
                self.proc.wait(timeout=5)
            except Exception:  # noqa: BLE001 - a sensor must never take the run down
                pass
            self._stop.wait(self.backoff_s)

    def sample(self) -> tuple[dict | None, float | None]:
        if self.latest is None or self.at is None:
            return None, None
        return self.latest, round(time.monotonic() - self.at, 1)

    def stop(self) -> None:
        self._stop.set()
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                try:
                    self.proc.kill()
                except Exception:  # noqa: BLE001
                    pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def write_stop(stop_file: Path, reason: str, sample: dict | None, source: str = "monitor") -> None:
    tmp = stop_file.with_name(stop_file.name + ".tmp")
    tmp.write_text(json.dumps({"reason": reason, "utc": _utc(), "source": source, "sample": sample}, indent=2))
    tmp.replace(stop_file)


def vm_sample(results_dir: Path) -> dict:
    return {"vm": probes.vm_memory(), "psi": probes.vm_psi(), "load": probes.loadavg(),
            "oom_kill": probes.oom_kills(), "results_free": probes.disk_free(results_dir)}


def run(rundir: Path, pid: int, out: Path, b: Budget, *, stop_file: Path | None = None,
        max_samples: int | None = None) -> int:
    rundir = Path(rundir)
    stop_file = stop_file or rundir / "STOP"
    smi = probes.nvidia_smi_bin()
    gpu = StreamReader([smi, "--query-gpu=" + ",".join(MON_GPU_FIELDS), "--format=csv,noheader,nounits",
                        "-lms", str(int(b.monitor_gpu_s * 1000))] if smi else None,
                       lambda line: probes.parse_gpu_csv(line, MON_GPU_FIELDS), backoff_s=5.0).start()
    host = StreamReader(probes.powershell_argv(probes.HOST_COUNTERS_PS), probes.parse_counter_line,
                        cwd=probes.ps_cwd(), backoff_s=2.0).start()
    guard = Guard(b)
    active: set[str] = set()
    n, signalled = 0, False
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"monitor: pid {os.getpid()} watching {pid}; samples -> {out}", file=sys.stderr, flush=True)
    try:
        with open(out, "a") as fh:
            while _alive(pid) and (max_samples is None or n < max_samples):
                s: dict = {"t": _utc(), **vm_sample(rundir.parent)}
                s["gpu"], s["gpu_age_s"] = gpu.sample()
                s["host"], s["host_age_s"] = host.sample()
                warn, crit = guard.evaluate(s)
                if warn:
                    s["warn"] = warn
                for w in sorted(set(warn) - active):
                    print(f"monitor: WARNING {w}", file=sys.stderr, flush=True)
                active = set(warn)
                if crit:
                    s["critical"] = crit
                    if not signalled:
                        print(f"monitor: CRITICAL {crit} -> {stop_file}", file=sys.stderr, flush=True)
                        write_stop(stop_file, crit, s)
                        try:
                            os.kill(pid, signal.SIGUSR1)
                        except OSError:
                            pass
                        signalled = True
                fh.write(json.dumps(s, separators=(",", ":")) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                n += 1
                time.sleep(b.monitor_vm_s)
    finally:
        gpu.stop()
        host.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="originblind ops monitor", description=__doc__.split("\n\n")[0])
    ap.add_argument("--rundir", required=True, help="results/<ns>; STOP is written here")
    ap.add_argument("--pid", type=int, required=True, help="the run_full.sh process to watch and signal")
    ap.add_argument("--out", required=True, help="JSONL flight record")
    ap.add_argument("--max-samples", type=int, default=None)
    a = ap.parse_args(argv)
    try:
        b = load_budget()
    except ValueError as e:
        print(f"monitor: invalid resource budget: {e}", file=sys.stderr)
        return 5
    return run(Path(a.rundir), a.pid, Path(a.out), b, max_samples=a.max_samples)
