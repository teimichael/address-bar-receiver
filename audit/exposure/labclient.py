"""Stdlib-only client for `originblind lab serve` (imported by scripts running in .venv-bgym/.venv-bu).

The external framework venvs never import originblind: they learn the proxy and control endpoint from
ports.json, reset the twin per condition (POST /episode) and read channel A (GET /log). Loopback
requests bypass any HTTP(S)_PROXY in the environment.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

# The exposure audit's condition grid (flows x origin classes) shared by every dump script: every
# origin class of the design (the pre-registered A1/A2 classes plus the raw-IP floor).
FLOWS = ("login", "otp", "checkout")
CLASSES = ("exact", "typosquat", "combosquat", "idn", "subdomain", "raw_ip", "a2_iframe", "a2_action")

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class Lab:
    def __init__(self, ports_path: str | Path) -> None:
        self.ports = json.loads(Path(ports_path).read_text())
        self.proxy_url: str = self.ports["proxy_url"]
        self.control_url: str = self.ports["control_url"]
        self.ca_pem: str = self.ports["ca_pem"]
        self.spki_sha256: str = self.ports.get("spki_sha256", "")

    def _call(self, method: str, path: str, payload: dict | None = None) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self.control_url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        with _OPENER.open(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))

    def episode(self, origin_class: str, flow: str = "login", **kw) -> dict:
        """Reset the twin to one condition; returns its description (landing_url, canary, ...)."""
        return self._call("POST", "/episode", {"origin_class": origin_class, "flow": flow, **kw})

    def current(self) -> dict | None:
        return self._call("GET", "/episode")

    def log(self) -> dict:
        return self._call("GET", "/log")

    def wait_log(self, settle_s: float = 1.0, timeout_s: float = 10.0) -> dict:
        """Read /log once no new request arrived for `settle_s` (in-flight POSTs finish first)."""
        t0 = time.time()
        last = -1
        while True:
            lg = self.log()
            n = lg.get("n_requests", 0)
            if n == last or time.time() - t0 > timeout_s:
                return lg
            last = n
            time.sleep(settle_s)


# Sidecar keys every dump script writes (read by score.py and, through exposure.json, by
# src/originblind/analysis/audits.py): framework = the producing distribution; framework_family in
# {browsergym, browser_use, playwright, ours}; default_config = the framework's default observation.
SIDECAR_KEYS = ("pipeline", "framework", "framework_family", "default_config", "config", "framework_version",
                "episode", "observed_top_url", "api")


def write_dump(out_root: Path, pipeline: str, flow: str, origin_class: str, text: str, meta: dict) -> Path:
    """results/<ns>/exposure/raw/<pipeline>/<flow>_<class>.txt (+ .json: the condition and API used)."""
    missing = [k for k in SIDECAR_KEYS if k not in meta]
    if missing:
        raise ValueError(f"{pipeline}: sidecar lacks {missing}")
    d = Path(out_root) / pipeline
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{flow}_{origin_class}"
    (d / f"{stem}.error.json").unlink(missing_ok=True)
    p = d / f"{stem}.txt"
    p.write_text(text, encoding="utf-8")
    (d / f"{stem}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def write_error(out_root: Path, pipeline: str, flow: str, origin_class: str, exc: BaseException) -> Path:
    """A failed dump: raw/<pipeline>/<flow>_<class>.error.json and NO .txt, so the scorer can never read
    a missing observation as "absent"."""
    import traceback

    d = Path(out_root) / pipeline
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{flow}_{origin_class}"
    for ext in (".txt", ".json"):
        (d / f"{stem}{ext}").unlink(missing_ok=True)
    p = d / f"{stem}.error.json"
    p.write_text(json.dumps({"pipeline": pipeline, "flow": flow, "origin_class": origin_class,
                             "error": f"{type(exc).__name__}: {exc}",
                             "traceback": "".join(traceback.format_exception(exc))[-4000:]},
                            indent=2, ensure_ascii=False), encoding="utf-8")
    return p
