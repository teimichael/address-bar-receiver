"""The resource budget of an unattended run: documented defaults, `OB_*` environment overrides.

Every field can be overridden by `OB_<FIELD IN UPPER CASE>` (e.g. `OB_SETTLE_S=600`,
`OB_DROP_CACHES=0`, `OB_VLLM_MEMORY=24g`). Invalid values raise ValueError naming the variable, so
a typo fails the run up front instead of silently using a default. The effective budget is
recorded in RUN_STATUS.json and in every gate record. Why each default (README.md):

  settle_s 420       the only server start after the 2026-10-14 Windows update that followed a
                     stopped server and did not hang the host came >= 7 min after the stop
  commit headroom    every VRAM allocation is charged to the Windows commit limit: a server needs
                     about util x VRAM + its weights + 8 GB of commit headroom (measured);
                     require_host_probe makes a missing Windows probe a refusal, not a skip
  gpu_idle_max_mib   Windows/WSL holds 1.0-1.6 GiB of the RTX 5090 when no server runs
  container memory   "" = VM MemTotal - container_reserve_gib: a runaway backstop that cannot bind
                     before the whole VM would run out (a binding limit would silently change
                     what episodes observe; README.md)
  guard_*            critical thresholds: STOP + abort (exit 6), safe to resume after the fix
  warn_*             logged only; the GPU shared-memory guard stays a warning until calibrated
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, fields
from typing import Mapping

GIB = 1024 ** 3


@dataclass(frozen=True)
class Budget:
    # ------------------------------------------------ pre-serve gate (serving/vllm.py, ops/gate.py)
    gpu_idle_max_mib: int = 3072          # VRAM used by others for the GPU to count as quiet
    gpu_quiesce_stable_s: float = 20.0    # quiet for this long before a server may start
    gpu_quiesce_timeout_s: float = 600.0  # give up (exit 6) if the GPU never becomes quiet
    gpu_poll_s: float = 2.0
    settle_s: float = 420.0               # minimum gap after the previous server stopped
    drop_caches: bool = True              # drop the VM page cache before each weight load
    vm_min_available_gib: float = 16.0    # VM MemAvailable needed before a weight load
    host_min_available_gb: float = 16.0   # Windows available memory needed before a weight load
    # Windows commit headroom needed: VRAM x utilization + weights + this margin (measured 2026-10-15:
    # a 4B server took 46.9 GB of commit = 40.1 GB predicted + 6.8 GB)
    host_commit_margin_gb: float = 8.0
    # without the Windows probe the gate cannot see the commit headroom; set (run_full.sh does for
    # `full`) the gate waits for the probe and then refuses instead of skipping the host checks
    require_host_probe: bool = False
    # ------------------------------------------------ container backstops (vLLM and lab)
    container_reserve_gib: float = 6.0
    pids_limit: int = 32768
    vllm_memory: str = ""                 # "" = auto (MemTotal - reserve); "none" = no limit
    lab_memory: str = ""
    # ------------------------------------------------ monitor (sensor) and guard
    monitor_vm_s: float = 2.0
    monitor_gpu_s: float = 5.0
    guard_samples: int = 3                # consecutive samples before a critical rule fires
    guard_host_available_gb: float = 2.0
    guard_vm_available_gib: float = 1.5
    guard_psi_full_avg10: float = 50.0
    guard_results_free_gb: float = 2.0
    guard_windows_drive_free_gb: float = 5.0
    guard_gpu_shared_mib: int = 0         # 0 = warning only (uncalibrated)
    guard_commit_headroom_gb: float = 1.5  # Windows commit limit minus committed
    warn_host_available_gb: float = 8.0
    warn_gpu_shared_mib: int = 1024
    warn_vm_available_gib: float = 4.0
    warn_swap_used_gib: float = 1.0
    warn_psi_some_avg10: float = 20.0
    warn_windows_drive_free_gb: float = 30.0
    warn_gpu_temp_c: float = 85.0
    warn_commit_headroom_gb: float = 10.0
    # ------------------------------------------------ run_full.sh
    poll_s: float = 5.0
    stall_warn_s: float = 1200.0
    stall_timeout_s: float = 3600.0       # no growth in the experiment dir: kill the pair, retry once
    lab_max_run_s: float = 10800.0
    breaker_groups: int = 2               # consecutive failed groups before the loop stops
    # ------------------------------------------------ host policy (preflight; full mode blocks)
    max_wsl_memory_fraction: float = 0.55
    windows_drive_block_gb: float = 20.0
    windows_drive_warn_gb: float = 60.0
    other_vm_warn_gb: float = 4.0
    update_pause_min_h: float = 24.0      # Windows Update must stay paused this long from now


def env_name(field: str) -> str:
    return "OB_" + field.upper()


def _parse(name: str, typ, raw: str):
    try:
        if typ is bool:
            if raw.strip().lower() in ("1", "true", "yes", "on"):
                return True
            if raw.strip().lower() in ("0", "false", "no", "off"):
                return False
            raise ValueError
        if typ is int:
            v = int(raw)
        elif typ is float:
            v = float(raw)
        else:
            return raw.strip()
        if v < 0:
            raise ValueError
        return v
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a valid {typ.__name__}") from None


def load_budget(env: Mapping[str, str] | None = None) -> Budget:
    """Defaults overridden by OB_* variables (raises ValueError on an invalid value)."""
    env = os.environ if env is None else env
    hints = {"int": int, "float": float, "bool": bool, "str": str}
    kw = {}
    for f in fields(Budget):
        raw = env.get(env_name(f.name))
        if raw is None or raw == "":
            continue
        kw[f.name] = _parse(env_name(f.name), hints[f.type] if isinstance(f.type, str) else f.type, raw)
    b = Budget(**kw)
    for f in ("vllm_memory", "lab_memory"):
        v = getattr(b, f)
        if v and v != "none" and parse_size(v) is None:
            raise ValueError(f"{env_name(f)}={v!r} is not a size like 24g or 24GB")
    return b


def as_dict(b: Budget) -> dict:
    return asdict(b)


_SIZE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?)(?:i?b)?\s*$", re.IGNORECASE)


def parse_size(s: str | None) -> int | None:
    """'48GB', '32g', '512MB', '1.5GiB' -> bytes (binary units, as .wslconfig and docker use them)."""
    if not s:
        return None
    m = _SIZE.match(str(s))
    if not m:
        return None
    mult = {"": 1, "k": 1024, "m": 1024 ** 2, "g": 1024 ** 3, "t": 1024 ** 4}[m.group(2).lower()]
    return int(float(m.group(1)) * mult)


def container_memory(setting: str, mem_total_bytes: int | None, reserve_gib: float) -> str | None:
    """The docker --memory value for a container: explicit size, "none", or auto (MemTotal - reserve,
    whole GiB, never below 4 GiB). None means: do not pass --memory."""
    if setting == "none":
        return None
    if setting:
        return setting
    if not mem_total_bytes:
        return None
    gib = int(mem_total_bytes / GIB - reserve_gib)
    return f"{max(gib, 4)}g"
