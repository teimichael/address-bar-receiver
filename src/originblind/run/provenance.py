"""Provenance manifest: everything needed to bind a result set to the exact code, models,
configuration, serving stack and environment that produced it."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _git(repo_root: Path) -> dict:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=repo_root).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True, cwd=repo_root)
        dirty = bool(status.strip())
        diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=repo_root) if dirty else b""
        return {"sha": sha, "dirty": dirty, "diff_sha256": hashlib.sha256(diff).hexdigest() if dirty else "",
                "n_changed": len([ln for ln in status.splitlines() if ln.strip()])}
    except Exception:
        return {"sha": "unknown", "dirty": True, "diff_sha256": "", "n_changed": -1}


def git_dirty(repo_root: Path) -> bool:
    return _git(repo_root)["dirty"]


def _sha256_file(p: Path) -> str:
    if not p.exists():
        return ""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _docker_digest(image: str) -> str:
    try:
        return subprocess.check_output(
            ["docker", "inspect", "--format", "{{index .RepoDigests 0}}", image], text=True, timeout=20
        ).strip()
    except Exception:
        return ""


def _gpu() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True, timeout=10).strip()
    except Exception:
        return ""


def _chromium_version() -> str:
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            b = p.chromium.launch()
            v = b.version
            b.close()
        return v
    except Exception:
        return ""


def prompt_template_hash() -> str:
    """Hash of the code that defines what the agent sees and is asked (prompts, serializer,
    pages, defenses, probe)."""
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for rel in ("agent/prompts.py", "agent/observe.py", "agent/htmlitems.py", "lab/pages.py",
                "defense/defenses.py", "agent/probe.py"):
        h.update((root / rel).read_bytes())
    return h.hexdigest()


def build_manifest(repo_root: Path, *, config_hash: str, plan_sha256: str,
                   models: list[dict], images: list[str], chromium: bool = False) -> dict:
    from originblind.oracle.attribute import psl_fingerprint

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": _git(repo_root),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "poetry_lock_sha256": _sha256_file(repo_root / "poetry.lock"),
        "pyproject_sha256": _sha256_file(repo_root / "pyproject.toml"),
        "prereg_sha256": _sha256_file(repo_root / "configs" / "prereg.yaml"),
        "prompt_template_sha256": prompt_template_hash(),
        "psl": psl_fingerprint(),
        "config_hash": config_hash,
        "plan_sha256": plan_sha256,
        "models": models,
        "docker_images": {img: _docker_digest(img) for img in images},
        "gpu": _gpu(),
        "chromium": _chromium_version() if chromium else "",
        "env": {
            "ORIGIN_BLIND_VLLM_URL": os.environ.get("ORIGIN_BLIND_VLLM_URL", ""),
            "cpu_count": os.cpu_count(),
        },
    }


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
