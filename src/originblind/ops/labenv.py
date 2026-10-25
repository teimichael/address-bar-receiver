"""One lab environment per run: the lab image's system packages may not change between attempts.

The release pins its base image, Python dependencies, and complete dpkg package versions.
An unavailable recorded Ubuntu version fails the build; there is no version fallback.

`originblind ops lab-env` (run by scripts/run_full.sh after setup whenever a selected config needs
the lab) records the image id, its source label and the sha256 of `dpkg-query -W` the first time a
namespace runs (<ns>/lab_env.json), and exits 1 when a later attempt's packages differ. A rebuild
that only changed the code layers (a new image id, the same packages) passes. Every attempt also
writes its own record with the full package list (logs/lab_env_<utc>.json).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from originblind.ops import probes

LABEL = "org.originblind.src_sha256"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    tmp.replace(path)


def describe(image: str, *, timeout: float = 300) -> dict:
    """The image id, its source label and its dpkg package list (read in a throwaway container with
    no network that never pulls). Raises RuntimeError when the image or docker is unavailable."""
    dk = probes.docker_bin()
    try:
        insp = subprocess.run([dk, "image", "inspect", "-f", "{{.Id}} {{index .Config.Labels \"%s\"}}" % LABEL,
                               image], capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"docker image inspect {image} failed: {e!r}"[:300]) from None
    if insp.returncode != 0 or not insp.stdout.strip():
        raise RuntimeError(f"lab image {image} not found (scripts/00_setup.sh builds it; after a purge, "
                           "`docker load` the saved image: README.md)")
    iid, _, label = insp.stdout.strip().partition(" ")
    try:
        r = subprocess.run([dk, "run", "--rm", "--pull", "never", "--network", "none", "--entrypoint",
                            "dpkg-query", iid, "-W"], capture_output=True, text=True, timeout=timeout,
                           stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"dpkg-query in {iid[:19]} failed: {e!r}"[:300]) from None
    if r.returncode != 0:
        raise RuntimeError(f"dpkg-query in {iid[:19]} failed: {r.stderr.strip()[:200]}")
    pkgs = sorted(ln.strip() for ln in r.stdout.splitlines() if ln.strip())
    return {"image": image, "image_id": iid, "src_sha256": label.strip() or None,
            "dpkg_sha256": hashlib.sha256(("\n".join(pkgs) + "\n").encode()).hexdigest(),
            "n_packages": len(pkgs), "packages": pkgs, "utc": _utc()}


def check(image: str, record: Path, out: Path | None = None, *,
          describe_fn: Callable[[str], dict] = describe) -> tuple[int, dict]:
    """(0, summary) when this is the namespace's first record or the packages match it; (1, summary)
    when they differ or the image cannot be described."""
    try:
        cur = describe_fn(image)
    except RuntimeError as e:
        return 1, {"status": "error", "error": str(e)}
    if out is not None:
        _write(out, cur)
    brief = {k: cur[k] for k in ("image_id", "src_sha256", "dpkg_sha256", "n_packages")}
    if not record.exists():
        _write(record, cur)
        return 0, {"status": "recorded", "record": str(record), "now": brief}
    try:
        prev = json.loads(record.read_text())
    except (OSError, ValueError) as e:
        return 1, {"status": "error", "error": f"unreadable {record}: {e!r}"[:300]}
    first = {k: prev.get(k) for k in ("image_id", "src_sha256", "dpkg_sha256", "n_packages", "utc")}
    if prev.get("dpkg_sha256") == cur["dpkg_sha256"]:
        return 0, {"status": "match", "record": str(record), "first": first, "now": brief,
                   "image_rebuilt": prev.get("image_id") != cur["image_id"]}
    a, b = set(prev.get("packages") or []), set(cur["packages"])
    return 1, {"status": "mismatch", "record": str(record), "first": first, "now": brief,
               "only_first": sorted(a - b)[:40], "only_now": sorted(b - a)[:40],
               "fix": "the lab image's system packages changed since this namespace's first attempt: restore the "
                      "first image (`docker load` the saved copy) or record the change in README.md "
                      "and move the record aside (README.md)"}
