"""Load episode stores into one analysis frame.

An experiment directory holds `episodes.jsonl` (one line per attempt; append-only), and the
sidecars `probes.jsonl`, `manifest.json` and `manifests/<utc>.json`. The reader is tolerant: a
corrupt line (e.g. a write cut short by a crash) is counted and skipped, never fatal. The
analysis unit is the FIRST `complete` record per episode_id; later complete records for the
same id are counted as duplicates (they can only arise from a broken resume) and ignored, and
non-complete attempts (infra_error / harness_error / excluded) are kept apart for validation.

Schema v1 (the October-2026 pilot) and schema v2 records are normalised into the same columns.
v1 lacks driver/authorized/abstention telemetry: driver is 'inprocess' (the only v1 driver),
r_authorized = r_bound (v1 had no delegates), `abstained` is approximated as a refuse/stop
terminal with no release (v1 did not log secret fills) and fields v1 never measured are NaN.

Probe answers are attached per probe key (model, brand, flow, origin, semantics, text arm,
replicate) from probes.jsonl, falling back to the copy embedded in the episode record.

Stimulus environment. Every row carries `environment` ("lab": the .test infrastructure names and
username; "realistic": no name the agent can read says "test"). Records written after the factor
was introduced carry `factors.environment`; older ones take the experiment's run configuration
(`manifest.json` "config"/"spec" -> "environment", then the per-invocation manifests), and a
store with neither is "lab" (the only environment that existed). The environment is a design
factor like any other: it is part of every cell, two directories of one experiment in different
environments (pv2_X and pv2r_X share their episode_ids by construction) are never deduplicated
against each other, and no estimate pools environments.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from originblind.types import OriginClass

CLASS_ORDER = ("exact", "typosquat", "combosquat", "idn", "subdomain", "raw_ip", "a2_iframe", "a2_action")
ARM_ORDER = ("T0", "T1", "T1u", "T2", "S0", "S1", "S2")
DEFENSE_ORDER = ("D0", "D1", "D2", "D3", "D3info", "D3push", "D5", "D5r", "D6", "D6r", "DBU")
DRIVER_ORDER = ("inprocess", "chromium", "chromium-headful")
FLOW_ORDER = ("login", "otp", "checkout", "profile", "psp", "psp_nocsp")
TERMINALS = ("success", "stop", "refuse", "budget", "parse_fail")
SCREEN_ARMS = ("S0", "S1", "S2")

ENVIRONMENT_ORDER = ("lab", "realistic")
DEFAULT_ENVIRONMENT = "lab"  # every store written before the environment factor existed

FACTOR_COLUMNS = ("model", "brand", "flow", "origin_class", "adversary", "secret", "semantics", "arm",
                  "defense", "driver", "environment", "replicate")
# a "cell" = every factor except the replicate units (brand, replicate)
CELL_COLUMNS = ("model", "flow", "origin_class", "adversary", "secret", "semantics", "arm", "defense", "driver",
                "environment")
BOOL_OUTCOMES = ("r_bound", "r_authorized", "leak", "task_success", "self_corrected", "abstained",
                 "entry_offorigin", "secret_filled", "submitted_after_secret", "channel_agreement")
TELEMETRY = ("steps", "parse_failures", "check_origin_calls", "d3push_shown_steps", "broker_denials",
             "egress_blocks", "fills_performed", "invalid_bid_actions", "nav_errors", "secret_fill_attempts",
             "secret_fills", "steps_to_first_secret_fill", "llm_calls", "llm_latency_ms", "handle_repairs")
PROBE_COLUMNS = ("probe_present", "probe_recognized", "probe_judged_mismatch", "probe_readout_correct",
                 "probe_readout_parse_failures", "probe_disc_yes", "probe_disc_no", "probe_k",
                 "probe_truth_is_official")

_PREFIX_RE = re.compile(r"^(?:pv\d+r?|pilot\d*)_")


def canonical_experiment(name: str) -> str:
    """Map a directory/config name to the pre-registered experiment id: pilot copies of a
    confirmatory config are named `pv2_<exp>` (lab environment) or `pv2r_<exp>` (their realistic-
    environment twins) in configs/pilot/, so the prefix is dropped."""
    return _PREFIX_RE.sub("", name or "")


def manifest_environments(manifest: dict | None, manifests: list[dict] | None = None) -> list[str]:
    """Distinct `environment` values of an experiment's run configuration, manifest.json first,
    then the per-invocation manifests in name order. The configuration is stored under "spec"
    (older drafts: "config"); a configuration without the key ran in the lab environment (the
    field defaults to "lab" and is dropped from the stored spec for identity stability)."""
    out: list[str] = []
    for m in ([manifest] if manifest else []) + list(manifests or []):
        cfg = None
        for key in ("config", "spec"):
            if isinstance((m or {}).get(key), dict):
                cfg = m[key]
                break
        if cfg is None:
            continue
        env = str(cfg.get("environment") or DEFAULT_ENVIRONMENT)
        if env not in out:
            out.append(env)
    return out


def class_group(origin_class: str) -> str:
    """Structural class group: exact | a1 | a2 | floor (raw IP)."""
    if origin_class == "raw_ip":
        return "floor"
    try:
        adv = OriginClass(origin_class).adversary
    except ValueError:
        return "other"
    return {"none": "exact", "A1": "a1", "A2": "a2"}[adv]


def text_arm(arm: str) -> str:
    return "T0" if arm in SCREEN_ARMS else arm


def probe_key_for(model: str, brand: str, flow: str, origin: str, semantics: str, arm: str,
                  replicate: int) -> str:
    """Identical to originblind.agent.probe.probe_key (asserted in tests)."""
    return "::".join([model, brand, flow, origin, semantics, text_arm(arm), str(replicate)])


def template_key_for(model: str, brand: str, flow: str, origin: str, semantics: str, arm: str) -> str:
    """The page template a v1 probe was computed on (v1 probes did not depend on the replicate)."""
    return "::".join([model, brand, flow, origin, semantics, text_arm(arm)])


# ---------------------------------------------------------------- raw readers

def read_jsonl(path: Path) -> tuple[list[dict], int, int]:
    """(records, non-empty lines, corrupt lines). Corrupt = not JSON or not an object."""
    recs: list[dict] = []
    n_lines = corrupt = 0
    if not path.exists():
        return recs, 0, 0
    with path.open("rb") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            n_lines += 1
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                corrupt += 1
                continue
            if not isinstance(obj, dict):
                corrupt += 1
                continue
            recs.append(obj)
    return recs, n_lines, corrupt


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifests(exp_dir: Path) -> tuple[dict | None, list[dict]]:
    """(manifest.json, [manifests/*.json sorted by name]) — unreadable files are skipped."""
    man = None
    p = exp_dir / "manifest.json"
    if p.exists():
        try:
            man = json.loads(p.read_text())
        except json.JSONDecodeError:
            man = None
    inv: list[dict] = []
    d = exp_dir / "manifests"
    if d.is_dir():
        for q in sorted(d.glob("*.json")):
            try:
                m = json.loads(q.read_text())
                m["_file"] = q.name
                inv.append(m)
            except json.JSONDecodeError:
                continue
    return man, inv


def read_probes(path: Path) -> tuple[dict[str, dict], int]:
    """probe_key -> probe record (first occurrence wins), corrupt-line count."""
    recs, _, corrupt = read_jsonl(path)
    out: dict[str, dict] = {}
    for r in recs:
        k = r.get("probe_key")
        if k and k not in out:
            out[k] = r
    return out, corrupt


# ---------------------------------------------------------------- normalisation

def _b(x) -> float:
    """bool-ish -> 1.0 / 0.0, missing -> NaN."""
    if x is None:
        return np.nan
    if isinstance(x, float) and np.isnan(x):
        return np.nan
    return 1.0 if bool(x) else 0.0


def _num(x) -> float:
    try:
        return float(x) if x is not None else np.nan
    except (TypeError, ValueError):
        return np.nan


def _probe_fields(p: dict | None) -> dict:
    if not p:
        return {c: np.nan for c in PROBE_COLUMNS} | {"probe_present": 0.0}
    k = p.get("k")
    out = {
        "probe_present": 1.0,
        "probe_recognized": _b(p.get("recognized")),
        "probe_judged_mismatch": _b(p.get("judged_mismatch")),
        # v2: readout_correct votes; v1: correct_votes
        "probe_readout_correct": _num(p.get("readout_correct", p.get("correct_votes"))),
        "probe_readout_parse_failures": _num(p.get("readout_parse_failures", p.get("parse_failures"))),
        "probe_disc_yes": _num(p.get("disc_yes")),
        "probe_disc_no": _num(p.get("disc_no")),
        "probe_k": _num(k),
        "probe_truth_is_official": _b(p.get("truth_is_official")),
    }
    return out


def normalize_record(rec: dict, *, experiment: str, exp_dir: str,
                     probes: dict[str, dict] | None = None, environment: str | None = None) -> dict:
    """One complete episode record -> one flat row (schema v1 or v2). `environment` is the
    experiment's configured environment, used when the record predates `factors.environment`."""
    f = rec.get("factors") or {}
    o = rec.get("outcome") or {}
    t = rec.get("telemetry") or {}
    sv = int(rec.get("schema_version") or 1)
    oc = str(f.get("origin_class", ""))
    try:
        adversary = f.get("adversary") or OriginClass(oc).adversary
    except ValueError:
        adversary = f.get("adversary") or ""
    model = str(f.get("model_key", f.get("model", "")))
    brand = str(f.get("brand", ""))
    flow = str(f.get("flow") or "login")
    sem = str(f.get("semantics") or "neutral")
    arm = str(f.get("arm", ""))
    rep = int(f.get("replicate") or 0)
    r_bound = _b(o.get("r_bound"))
    leak = _b(o.get("leak"))
    terminal = o.get("terminal_type")
    if sv >= 2:
        abstained = _b(o.get("abstained"))
        r_auth = _b(o.get("r_authorized"))
    else:  # v1 approximation: a refuse/stop terminal with no release
        abstained = float(terminal in ("refuse", "stop") and not (r_bound or leak))
        r_auth = r_bound
    self_corr = o.get("self_corrected")
    if self_corr is None:
        self_corr = bool(r_bound) and adversary == "A1"
    pkey = probe_key_for(model, brand, flow, oc, sem, arm, rep)
    probe = (probes or {}).get(pkey) or rec.get("probe")
    if f.get("environment"):
        env, env_src = str(f["environment"]), "record"
    elif environment:
        env, env_src = str(environment), "manifest"
    else:
        env, env_src = DEFAULT_ENVIRONMENT, "default"
    row = {
        "experiment": experiment, "exp_dir": exp_dir, "namespace": rec.get("namespace"),
        "schema_version": sv, "episode_id": rec.get("episode_id"), "attempt": int(rec.get("attempt") or 0),
        "model": model, "brand": brand, "flow": flow, "origin_class": oc, "adversary": adversary,
        "secret": str(f.get("secret") or ""), "semantics": sem, "arm": arm, "defense": str(f.get("defense", "")),
        "driver": str(f.get("driver") or "inprocess"), "environment": env, "environment_source": env_src,
        "replicate": rep, "pii_mode": f.get("pii_mode"),
        "class_group": class_group(oc), "deceptive": float(oc != "exact"),
        "r_bound": r_bound, "r_authorized": r_auth, "leak": leak,
        "task_success": _b(o.get("task_success")), "self_corrected": _b(self_corr),
        "terminal_type": terminal, "abstained": abstained,
        "entry_offorigin": _b(o.get("entry_offorigin")), "secret_filled": _b(o.get("secret_filled")),
        "submitted_after_secret": _b(o.get("submitted_after_secret")),
        "channel_agreement": _b(o.get("channel_agreement")),
        "n_release_domains": float(len(o.get("release_domains") or [])),
        "wall_s": _num(rec.get("wall_s")),
        "probe_key": pkey, "template_key": template_key_for(model, brand, flow, oc, sem, arm),
    }
    for k in TELEMETRY:
        row[k] = _num(t.get(k))
    ctxs = t.get("secret_fill_contexts")
    row["n_secret_fill_contexts"] = float(len(ctxs)) if isinstance(ctxs, list) else np.nan
    row["broker_deny"] = float((row["broker_denials"] or 0) > 0 or (row["egress_blocks"] or 0) > 0) \
        if not (np.isnan(row["broker_denials"]) and np.isnan(row["egress_blocks"])) else np.nan
    row["check_origin_any"] = float(row["check_origin_calls"] > 0) if not np.isnan(row["check_origin_calls"]) else np.nan
    row["d3push_shown"] = float(row["d3push_shown_steps"] > 0) if not np.isnan(row["d3push_shown_steps"]) else np.nan
    row["parse_fail_any"] = float(row["parse_failures"] > 0) if not np.isnan(row["parse_failures"]) else np.nan
    row["false_refusal"] = 1.0 - row["task_success"] if not np.isnan(row["task_success"]) else np.nan
    row.update(_probe_fields(probe))
    return row


# ---------------------------------------------------------------- experiments

@dataclass
class ExperimentData:
    name: str  # directory name
    experiment: str  # canonical (pre-registered) experiment id
    path: Path
    episodes: pd.DataFrame  # first complete record per episode_id, normalised
    attempts: pd.DataFrame  # one row per parsed record: episode_id, status, attempt
    n_lines: int = 0
    n_corrupt: int = 0
    n_duplicate_complete: int = 0
    manifest: dict | None = None
    manifests: list[dict] = field(default_factory=list)
    probes: dict[str, dict] = field(default_factory=dict)
    n_probe_corrupt: int = 0
    sha256: str = ""
    schema_versions: list[int] = field(default_factory=list)
    environment: str = DEFAULT_ENVIRONMENT  # the run configuration's environment (manifest)
    manifest_environments: list[str] = field(default_factory=list)  # every value the manifests record

    @property
    def spec(self) -> dict | None:
        return (self.manifest or {}).get("spec")

    @property
    def environments(self) -> list[str]:
        """Environments of the complete records (record factor, else the manifest's)."""
        if not len(self.episodes):
            return [self.environment]
        return order_values(set(self.episodes["environment"]), ENVIRONMENT_ORDER)

    @property
    def is_legacy(self) -> bool:
        return bool(self.schema_versions) and max(self.schema_versions) < 2


EPISODE_COLUMNS = (["experiment", "exp_dir", "namespace", "schema_version", "episode_id", "attempt",
                    *FACTOR_COLUMNS, "environment_source", "pii_mode", "class_group", "deceptive", *BOOL_OUTCOMES, "terminal_type",
                    "n_release_domains", "wall_s", "probe_key", "template_key", *TELEMETRY,
                    "n_secret_fill_contexts", "broker_deny", "check_origin_any", "d3push_shown",
                    "parse_fail_any", "false_refusal", *PROBE_COLUMNS])


def empty_episodes() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object) for c in dict.fromkeys(EPISODE_COLUMNS)})


def experiment_name(exp_dir: Path, manifest: dict | None) -> str:
    spec = (manifest or {}).get("spec") or {}
    return str(spec.get("name") or (manifest or {}).get("name") or exp_dir.name)


def order_values(values, order) -> list[str]:
    """Values in a canonical order, unknown ones after it (sorted)."""
    vals = {str(v) for v in values}
    return [v for v in order if v in vals] + sorted(vals - set(order))


def load_experiment(exp_dir: str | Path, *, episodes_path: Path | None = None) -> ExperimentData:
    exp_dir = Path(exp_dir)
    path = episodes_path or exp_dir / "episodes.jsonl"
    manifest, manifests = read_manifests(exp_dir)
    experiment = canonical_experiment(experiment_name(exp_dir, manifest))
    man_envs = manifest_environments(manifest, manifests)
    environment = man_envs[0] if man_envs else DEFAULT_ENVIRONMENT
    probes, probe_corrupt = read_probes(exp_dir / "probes.jsonl")
    recs, n_lines, corrupt = read_jsonl(path)
    rows, attempts = [], []
    seen: set[str] = set()
    dup = 0
    versions: set[int] = set()
    for r in recs:
        eid = r.get("episode_id")
        st = r.get("status")
        attempts.append({"episode_id": eid, "status": st, "attempt": r.get("attempt"),
                         "error": (r.get("error") or "")[:200]})
        if st != "complete" or not eid:
            continue
        if eid in seen:
            dup += 1
            continue
        seen.add(eid)
        versions.add(int(r.get("schema_version") or 1))
        rows.append(normalize_record(r, experiment=experiment, exp_dir=exp_dir.name, probes=probes,
                                     environment=environment))
    eps = pd.DataFrame(rows) if rows else empty_episodes()
    return ExperimentData(
        name=exp_dir.name, experiment=experiment, path=exp_dir, episodes=eps,
        attempts=pd.DataFrame(attempts, columns=["episode_id", "status", "attempt", "error"]),
        n_lines=n_lines, n_corrupt=corrupt, n_duplicate_complete=dup, manifest=manifest,
        manifests=manifests, probes=probes, n_probe_corrupt=probe_corrupt, sha256=sha256_file(path),
        schema_versions=sorted(versions), environment=environment, manifest_environments=man_envs)


def experiment_dirs(ns_dir: str | Path) -> list[Path]:
    """Every direct child directory of a namespace that holds an episodes.jsonl (or the
    directory itself if it is an experiment directory)."""
    ns = Path(ns_dir)
    if (ns / "episodes.jsonl").exists():
        return [ns]
    return sorted(p for p in ns.iterdir() if p.is_dir() and (p / "episodes.jsonl").exists()) if ns.is_dir() else []


def load_namespace(ns_dir: str | Path) -> dict[str, ExperimentData]:
    """Directory name -> ExperimentData, sorted by directory name."""
    return {p.name: load_experiment(p) for p in experiment_dirs(ns_dir)}


def concat_episodes(exps: dict[str, ExperimentData] | list[ExperimentData]) -> pd.DataFrame:
    """One frame over all experiments; if two directories map to the same canonical experiment IN
    THE SAME ENVIRONMENT, an episode_id is kept once (first directory in name order). The lab and
    realistic twins of an experiment share their episode_ids by construction and are both kept."""
    items = list(exps.values()) if isinstance(exps, dict) else list(exps)
    frames = [e.episodes for e in items if len(e.episodes)]
    if not frames:
        return empty_episodes()
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["environment", "experiment", "episode_id"], keep="first").reset_index(drop=True)
    return df


def environments_of(df: pd.DataFrame) -> list[str]:
    """Environments present in an analysis frame, in canonical order."""
    if df is None or df.empty or "environment" not in df:
        return []
    return order_values(set(df["environment"].dropna()), ENVIRONMENT_ORDER)


def order_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows in the canonical factor order (for deterministic tables)."""
    if df.empty:
        return df
    keys = []
    for col, order in (("origin_class", CLASS_ORDER), ("arm", ARM_ORDER), ("defense", DEFENSE_ORDER),
                       ("driver", DRIVER_ORDER), ("flow", FLOW_ORDER), ("environment", ENVIRONMENT_ORDER)):
        if col in df.columns:
            rank = {v: i for i, v in enumerate(order)}
            keys.append((col, df[col].map(lambda v, r=rank: r.get(v, len(r)))))
    tmp = df.copy()
    sort_cols = []
    for col, ranks in keys:
        tmp[f"__r_{col}"] = ranks
    by = []
    for c in ("environment", "experiment", "model", "flow", "arm", "defense", "driver", "semantics", "origin_class"):
        if c in tmp.columns:
            by.append(f"__r_{c}" if f"__r_{c}" in tmp.columns else c)
    for c in tmp.columns:
        if c.startswith("__r_"):
            sort_cols.append(c)
    tmp = tmp.sort_values(by=by, kind="mergesort") if by else tmp
    return tmp.drop(columns=sort_cols).reset_index(drop=True)
