"""Shared paths of an unattended run and the global run lock.

`scripts/run_full.sh` holds an flock on run_lock_path() for its whole lifetime (the vLLM container
`ob-vllm` and port 8000 are machine-wide singletons). It exports OB_RUN_LOCK_HELD=1 to its
children; any other `serve start`/`serve stop` refuses while the lock is held, so a manual
`scripts/10_serve_model.sh` cannot replace a live run's server.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def ops_dir() -> Path:
    """Run-time state (server-stop stamp, staging, backups): OB_OPS_DIR, else <repo>/results/.ops."""
    return Path(os.environ.get("OB_OPS_DIR") or REPO_ROOT / "results" / ".ops")


def run_lock_path() -> Path:
    return Path(os.environ.get("OB_RUN_LOCK") or ops_dir() / "run_full.lock")


def lock_held_by_other(path: Path | None = None) -> bool:
    """True iff another process holds the run lock (probes with a non-blocking flock)."""
    path = path or run_lock_path()
    if not path.exists():
        return False
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def refuse_if_locked() -> str | None:
    """An error message when a run_full other than our parent holds the lock, else None."""
    if os.environ.get("OB_RUN_LOCK_HELD") == "1":
        return None
    if lock_held_by_other():
        return (f"scripts/run_full.sh is running (lock {run_lock_path()} is held); it owns the vLLM "
                "container. Stop that run first, or wait for it to finish.")
    return None
