"""Read-only probes of the VM, the GPU and the Windows host.

Pure parsers (unit-tested on fixtures) plus thin side-effect wrappers. Every wrapper has a timeout,
takes its binary from an environment seam (OB_NVIDIA_SMI, OB_POWERSHELL, OB_DOCKER) with an
absolute-path fallback (systemd units and the tests run with a minimal PATH), and returns None
instead of raising when the probe is unavailable. Callers decide whether "cannot measure" is fatal:
the pre-serve gate fails closed on the GPU, the host checks fail open with a recorded warning.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
from pathlib import Path

from originblind.ops.budget import parse_size

NVIDIA_SMI_FALLBACK = "/usr/lib/wsl/lib/nvidia-smi"
POWERSHELL_FALLBACK = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
GPU_FIELDS = ("name", "driver_version", "memory.used", "memory.free", "memory.total", "display_attached",
              "display_active", "temperature.gpu", "utilization.gpu", "power.draw")


# ---------------------------------------------------------------- binaries

def _bin(env: str, name: str, fallback: str | None) -> str | None:
    if os.environ.get(env):
        return os.environ[env]
    found = shutil.which(name)
    if found:
        return found
    return fallback if fallback and Path(fallback).exists() else None


def nvidia_smi_bin() -> str | None:
    return _bin("OB_NVIDIA_SMI", "nvidia-smi", NVIDIA_SMI_FALLBACK)


def powershell_bin() -> str | None:
    return _bin("OB_POWERSHELL", "powershell.exe", POWERSHELL_FALLBACK)


def docker_bin() -> str:
    return os.environ.get("OB_DOCKER") or "docker"


# ---------------------------------------------------------------- VM (/proc)

def parse_meminfo(text: str) -> dict[str, int]:
    """/proc/meminfo -> {field: bytes}."""
    out: dict[str, int] = {}
    for line in text.splitlines():
        k, _, rest = line.partition(":")
        parts = rest.split()
        if not parts:
            continue
        try:
            v = int(parts[0])
        except ValueError:
            continue
        out[k.strip()] = v * 1024 if len(parts) > 1 and parts[1] == "kB" else v
    return out


def parse_psi(text: str) -> dict[str, dict[str, float]]:
    """/proc/pressure/<x> -> {"some": {"avg10": .., "avg60": .., "avg300": .., "total": ..}, "full": ...}."""
    out: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        kv = {}
        for p in parts[1:]:
            k, _, v = p.partition("=")
            try:
                kv[k] = float(v)
            except ValueError:
                continue
        out[parts[0]] = kv
    return out


def parse_vmstat(text: str, key: str = "oom_kill") -> int | None:
    for line in text.splitlines():
        k, _, v = line.partition(" ")
        if k == key:
            try:
                return int(v)
            except ValueError:
                return None
    return None


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text()
    except OSError:
        return None


def vm_memory() -> dict[str, int] | None:
    t = _read("/proc/meminfo")
    if t is None:
        return None
    m = parse_meminfo(t)
    return {"total": m.get("MemTotal", 0), "available": m.get("MemAvailable", 0), "cached": m.get("Cached", 0),
            "dirty": m.get("Dirty", 0), "swap_total": m.get("SwapTotal", 0),
            "swap_used": m.get("SwapTotal", 0) - m.get("SwapFree", 0)}


def vm_psi() -> dict[str, dict] | None:
    out = {}
    for kind in ("memory", "io", "cpu"):
        t = _read(f"/proc/pressure/{kind}")
        if t is not None:
            out[kind] = parse_psi(t)
    return out or None


def oom_kills() -> int | None:
    t = _read("/proc/vmstat")
    return parse_vmstat(t) if t is not None else None


def loadavg() -> list[float] | None:
    t = _read("/proc/loadavg")
    try:
        return [float(x) for x in t.split()[:3]] if t else None
    except ValueError:
        return None


def disk_free(path: str | Path) -> int | None:
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return None


# ---------------------------------------------------------------- GPU (nvidia-smi)

def _num(s: str) -> float | None:
    try:
        return float(s.strip())
    except ValueError:
        return None


def parse_gpu_csv(line: str, fields: tuple[str, ...] = GPU_FIELDS) -> dict | None:
    """One `nvidia-smi --query-gpu=... --format=csv,noheader,nounits` line -> dict (MiB, °C, W)."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < len(fields):
        return None
    d = dict(zip(fields, parts))
    out = {"name": d.get("name"), "driver": d.get("driver_version"),
           "used_mib": _num(d.get("memory.used", "")), "free_mib": _num(d.get("memory.free", "")),
           "total_mib": _num(d.get("memory.total", "")),
           # attached = a monitor is cabled to this GPU (stable); active = the screen is on right now
           "display_attached": (d.get("display_attached") or "").lower() == "yes" if "display_attached" in d else None,
           "display_active": (d.get("display_active") or "").lower() == "enabled" if "display_active" in d else None,
           "temp_c": _num(d.get("temperature.gpu", "")), "util_pct": _num(d.get("utilization.gpu", "")),
           "power_w": _num(d.get("power.draw", ""))}
    if out["used_mib"] is None or out["total_mib"] is None:
        return None
    return out


def gpu_query(timeout: float = 20) -> dict | None:
    """The first GPU, or None when nvidia-smi is missing or fails."""
    smi = nvidia_smi_bin()
    if not smi:
        return None
    try:
        r = subprocess.run([smi, "--query-gpu=" + ",".join(GPU_FIELDS), "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    return parse_gpu_csv(r.stdout.strip().splitlines()[0])


# ---------------------------------------------------------------- Windows host (powershell.exe)

_PS_UTF8 = "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; $ErrorActionPreference = 'SilentlyContinue'\n"
_PS_GPU = (
    "$ded = $s | Where-Object { $_.Path -like '*\\dedicated usage' } | Sort-Object CookedValue -Descending"
    " | Select-Object -First 1\n"
    "$sh = $s | Where-Object { $_.Path -like '*\\shared usage' -and $_.InstanceName -eq $ded.InstanceName }"
    " | Select-Object -First 1\n")

HOST_PROBE_PS = _PS_UTF8 + (
    "$cs = Get-CimInstance Win32_ComputerSystem\n"
    "$cv = Get-ItemProperty 'HKLM:\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion'\n"
    "$u = Get-ItemProperty 'HKLM:\\SOFTWARE\\Microsoft\\WindowsUpdate\\UX\\Settings'\n"
    "$s = (Get-Counter -Counter '\\Memory\\Available MBytes','\\Memory\\Committed Bytes','\\Memory\\Commit Limit',"
    "'\\GPU Adapter Memory(*)\\Dedicated Usage','\\GPU Adapter Memory(*)\\Shared Usage').CounterSamples\n"
    + _PS_GPU +
    "$wsl = [string](Get-Content -Raw (Join-Path $env:USERPROFILE '.wslconfig'))\n"
    "$drives = @(Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | ForEach-Object {"
    " [pscustomobject]@{ id = $_.DeviceID; free = [int64]$_.FreeSpace; size = [int64]$_.Size } })\n"
    "$procs = @(Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 10 | ForEach-Object {"
    " [pscustomobject]@{ name = $_.ProcessName; ws = [int64]$_.WorkingSet64 } })\n"
    # by name: an idle Ollama is far too small for the top-10 list, yet can load a model onto the GPU
    "$llm = @(Get-Process -Name 'ollama*','LM Studio','lms','llama-server' -ErrorAction SilentlyContinue"
    " | ForEach-Object { [pscustomobject]@{ name = $_.ProcessName; ws = [int64]$_.WorkingSet64 } })\n"
    "$pf = @(Get-CimInstance Win32_PageFileUsage | ForEach-Object {"
    " [pscustomobject]@{ name = $_.Name; alloc_mb = [int64]$_.AllocatedBaseSize } })\n"
    "[pscustomobject]@{\n"
    " total_phys = [int64]$cs.TotalPhysicalMemory\n"
    " available_mb = [double](($s | Where-Object { $_.Path -like '*\\memory\\available mbytes' }).CookedValue)\n"
    " committed = [double](($s | Where-Object { $_.Path -like '*\\memory\\committed bytes' }).CookedValue)\n"
    " commit_limit = [double](($s | Where-Object { $_.Path -like '*\\memory\\commit limit' }).CookedValue)\n"
    " gpu_dedicated = [double]$ded.CookedValue\n"
    " gpu_shared = [double]$sh.CookedValue\n"
    " build = \"$($cv.CurrentBuild).$($cv.UBR)\"\n"
    " display_version = $cv.DisplayVersion\n"
    " pause_updates_expiry = $u.PauseUpdatesExpiryTime\n"
    " drives = $drives\n"
    " procs = $procs\n"
    " llm = $llm\n"
    " pagefiles = $pf\n"
    " pagefile_auto = [bool]$cs.AutomaticManagedPagefile\n"
    " wslconfig = $wsl\n"
    "} | ConvertTo-Json -Compress -Depth 4\n")

# bounded (15 samples x 2 s): the monitor relaunches it, so a dead relay never leaves an endless
# Windows process behind
HOST_COUNTERS_PS = _PS_UTF8 + (
    "$paths = '\\Memory\\Available MBytes','\\Memory\\Committed Bytes','\\Memory\\Commit Limit',"
    "'\\Paging File(_Total)\\% Usage','\\GPU Adapter Memory(*)\\Dedicated Usage',"
    "'\\GPU Adapter Memory(*)\\Shared Usage','\\LogicalDisk(*)\\Free Megabytes'\n"
    "Get-Counter -Counter $paths -SampleInterval 2 -MaxSamples 15 | ForEach-Object {\n"
    " $s = $_.CounterSamples\n"
    + _PS_GPU +
    " $drv = @{}\n"
    " $s | Where-Object { $_.Path -like '*\\free megabytes' -and $_.InstanceName -match '^[a-z]:$' }"
    " | ForEach-Object { $drv[$_.InstanceName.ToUpper()] = [double]$_.CookedValue }\n"
    " [pscustomobject]@{\n"
    "  t = $_.Timestamp.ToUniversalTime().ToString('o')\n"
    "  available_mb = [double](($s | Where-Object { $_.Path -like '*\\memory\\available mbytes' }).CookedValue)\n"
    "  committed = [double](($s | Where-Object { $_.Path -like '*\\memory\\committed bytes' }).CookedValue)\n"
    "  commit_limit = [double](($s | Where-Object { $_.Path -like '*\\memory\\commit limit' }).CookedValue)\n"
    "  pagefile_pct = [double](($s | Where-Object { $_.Path -like '*\\paging file(_total)\\% usage' }).CookedValue)\n"
    "  gpu_dedicated = [double]$ded.CookedValue\n"
    "  gpu_shared = [double]$sh.CookedValue\n"
    "  drives_free_mb = $drv\n"
    " } | ConvertTo-Json -Compress\n"
    "}\n")


def powershell_argv(script: str) -> list[str] | None:
    ps = powershell_bin()
    if not ps:
        return None
    enc = base64.b64encode(script.encode("utf-16-le")).decode()
    return [ps, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-EncodedCommand", enc]


def ps_cwd() -> str | None:
    """A Windows-side working directory (a WSL cwd makes powershell.exe warn about UNC paths)."""
    return "/mnt/c" if Path("/mnt/c").is_dir() else None


def parse_wslconfig(text) -> dict:
    """The [wsl2] section of .wslconfig: raw values plus memory/swap in bytes."""
    out: dict = {"raw": {}, "memory": None, "swap": None}
    if isinstance(text, dict):  # a PowerShell string serialised with its note properties
        text = text.get("value")
    if not isinstance(text, str) or not text:
        return out
    section = ""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().lower()
            continue
        if section == "wsl2" and "=" in line:
            k, _, v = line.partition("=")
            out["raw"][k.strip().lower()] = v.strip()
    out["memory"] = parse_size(out["raw"].get("memory"))
    out["swap"] = parse_size(out["raw"].get("swap"))
    return out


def parse_host_json(text: str) -> dict | None:
    """Normalise the host probe's JSON (bytes everywhere)."""
    try:
        d = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(d, dict):
        return None

    def f(k):
        try:
            return float(d.get(k)) if d.get(k) is not None else None
        except (TypeError, ValueError):
            return None

    drives = {}
    for dr in d.get("drives") or []:
        if isinstance(dr, dict) and dr.get("id"):
            drives[str(dr["id"]).upper()] = {"free": int(dr.get("free") or 0), "size": int(dr.get("size") or 0)}
    procs = [(str(p.get("name")), int(p.get("ws") or 0)) for p in d.get("procs") or [] if isinstance(p, dict)]
    llm = d.get("llm") or []
    llm = [llm] if isinstance(llm, dict) else llm
    llm_servers = [(str(p.get("name")), int(p.get("ws") or 0)) for p in llm if isinstance(p, dict)]
    pagefiles = [(str(p.get("name")), int(p.get("alloc_mb") or 0) * 1024 * 1024)
                 for p in d.get("pagefiles") or [] if isinstance(p, dict)]
    avail_mb = f("available_mb")
    return {"total_phys": int(f("total_phys") or 0) or None,
            "available": int(avail_mb * 1024 * 1024) if avail_mb is not None else None,
            "committed": f("committed"), "commit_limit": f("commit_limit"),
            "gpu_dedicated": f("gpu_dedicated"), "gpu_shared": f("gpu_shared"),
            "build": d.get("build"), "display_version": d.get("display_version"),
            "pause_updates_expiry": d.get("pause_updates_expiry") or None,
            "drives": drives, "procs": procs, "llm_servers": llm_servers, "pagefiles": pagefiles,
            "pagefile_auto": d.get("pagefile_auto"), "wslconfig": parse_wslconfig(d.get("wslconfig"))}


def commit_headroom(host: dict | None) -> float | None:
    """Windows commit limit minus committed bytes (None when unknown). Every VRAM allocation of a
    WSL CUDA process is charged here, next to the VM's own memory (README.md)."""
    if not host or host.get("commit_limit") is None or host.get("committed") is None:
        return None
    return float(host["commit_limit"]) - float(host["committed"])


WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")


def weights_bytes(manifest: dict | None) -> int | None:
    """Bytes of weight files in a materialized model's .ob_manifest.json (a conservative estimate of
    the page cache a load touches); None when unknown."""
    files = (manifest or {}).get("files") or {}
    if not isinstance(files, dict) or not files:
        return None
    sizes = {n: (v or {}).get("size") or 0 for n, v in files.items() if isinstance(v, dict)}
    w = sum(sz for n, sz in sizes.items() if n.endswith(WEIGHT_SUFFIXES))
    return w or sum(sizes.values()) or None


def commit_needed(util: float, vram_total_bytes: float, weights: int | None, margin_gb: float) -> float:
    """Commit headroom a server start needs: its VRAM share + its weights' page cache + a margin
    (weights unknown: assume 25 GB)."""
    return util * vram_total_bytes + (weights if weights is not None else 25e9) + margin_gb * 1e9


def host_probe(timeout: float = 60) -> dict | None:
    """One snapshot of the Windows host, or None without interop (never raises)."""
    argv = powershell_argv(HOST_PROBE_PS)
    if not argv:
        return None
    try:
        r = subprocess.run(argv, capture_output=True, timeout=timeout, cwd=ps_cwd(), stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    out = r.stdout.decode("utf-8", errors="replace").strip().splitlines()
    return parse_host_json(out[-1]) if out else None


def parse_counter_line(line: str) -> dict | None:
    """One HOST_COUNTERS_PS line -> {available, gpu_shared, gpu_dedicated, drives_free (bytes), ...}."""
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict):
        return None
    mib = 1024 * 1024

    def f(k):
        try:
            return float(d[k]) if d.get(k) is not None else None
        except (TypeError, ValueError):
            return None

    drives = {}
    for k, v in (d.get("drives_free_mb") or {}).items():
        try:
            drives[str(k).upper()] = int(float(v) * mib)
        except (TypeError, ValueError):
            continue
    avail = f("available_mb")
    return {"t": d.get("t"), "available": int(avail * mib) if avail is not None else None,
            "committed": f("committed"), "commit_limit": f("commit_limit"), "pagefile_pct": f("pagefile_pct"),
            "gpu_dedicated": f("gpu_dedicated"), "gpu_shared": f("gpu_shared"), "drives_free": drives}
