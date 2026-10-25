"""Deterministic experiment plan: the factorial expansion with a seeded, model-blocked run order
and stable identities (plan hash, code hash)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from originblind.config import SCREEN_ARMS, ExperimentSpec
from originblind.lab.registry import BRANDS
from originblind.run.episode import episode_id
from originblind.seeds import derive_seed
from originblind.types import OriginClass, SecretKind


@dataclass(frozen=True)
class PlanItem:
    episode_id: str
    model_key: str
    model_served: str
    brand: str
    flow: str
    origin_class: str
    secret: str
    semantics: str
    arm: str
    defense: str
    pii_mode: str
    replicate: int
    driver: str = "inprocess"


def _legacy_expand(spec: ExperimentSpec) -> list[PlanItem]:
    """The original expansion (brand = brands[r % len(brands)]), kept for the pilot configs."""
    brands = spec.brands or list(BRANDS)
    secrets = spec.secrets or [spec.secret_for(f) for f in spec.flows]
    items: list[PlanItem] = []
    for model in spec.models:
        for oc in spec.origin_classes:
            for sem in spec.semantics:
                for secret in secrets:
                    for flow in spec.flows:
                        for arm in spec.observation_arms:
                            for defense in spec.defenses:
                                for pii in spec.pii_mode:
                                    for r in range(spec.replicates):
                                        brand = brands[r % len(brands)]
                                        eid = episode_id(brand, flow, OriginClass(oc), SecretKind(secret),
                                                         sem, r, model.key, arm, defense) + f"::{pii}"
                                        items.append(PlanItem(
                                            episode_id=eid, model_key=model.key, model_served=model.served_name,
                                            brand=brand, flow=flow, origin_class=str(oc), secret=str(secret),
                                            semantics=sem, arm=arm, defense=defense, pii_mode=pii, replicate=r,
                                            driver=spec.driver,
                                        ))
    return items


def _balanced_expand(spec: ExperimentSpec) -> list[PlanItem]:
    brands = spec.brands or list(BRANDS)
    items: list[PlanItem] = []
    for model in spec.models:
        for oc in spec.origin_classes:
            oc = OriginClass(oc)
            reps = spec.replicates_per_brand * (spec.exact_oversample if oc is OriginClass.EXACT else 1)
            for flow in spec.flows:
                secret = spec.secret_for(flow)
                for arm in spec.observation_arms:
                    if arm in SCREEN_ARMS and not model.vision:
                        continue  # text-only models never get screenshot arms
                    allowed = spec.arm_origins.get(arm)
                    if allowed is not None and oc not in allowed:
                        continue
                    for defense in spec.defenses:
                        for sem in spec.semantics:
                            for brand in brands:
                                for r in range(reps):
                                    eid = "::".join([model.key, arm, defense, brand, flow, oc.value,
                                                     secret.value, sem, str(r), spec.driver])
                                    items.append(PlanItem(
                                        episode_id=eid, model_key=model.key, model_served=model.served_name,
                                        brand=brand, flow=flow, origin_class=oc.value, secret=secret.value,
                                        semantics=sem, arm=arm, defense=defense, pii_mode="plaintext",
                                        replicate=r, driver=spec.driver,
                                    ))
    return items


def expand(spec: ExperimentSpec) -> list[PlanItem]:
    return _balanced_expand(spec) if spec.balanced else _legacy_expand(spec)


def ordered(spec: ExperimentSpec, model: str | None = None) -> list[PlanItem]:
    """Seeded permutation that keeps models contiguous (one vLLM model resident at a time) but
    shuffles within each model so drift/thermal effects are not confounded with any factor."""
    items = expand(spec)
    by_model: dict[str, list[PlanItem]] = {}
    for it in items:
        if model and it.model_key != model:
            continue
        by_model.setdefault(it.model_key, []).append(it)
    out: list[PlanItem] = []
    for mk in sorted(by_model):
        group = by_model[mk]
        group.sort(key=lambda it: hashlib.blake2b(
            (it.episode_id).encode(), digest_size=8,
            key=derive_seed(spec.master_seed, "order", mk).to_bytes(32, "big")[:16],
        ).hexdigest())
        out.extend(group)
    return out


def plan_hash(spec: ExperimentSpec) -> str:
    items = expand(spec)
    rows = []
    for i in sorted(items, key=lambda x: x.episode_id):
        d = asdict(i)
        if not spec.balanced:
            d.pop("driver", None)  # legacy plans hash exactly as before
        rows.append(d)
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


# Code that determines what an episode measures (stimuli, observation, agent loop, defenses,
# oracle, plan/seeds). Analysis, serving, preflight and CLI code are deliberately excluded: they
# can change without invalidating already-measured episodes (they are versioned by git instead).
_MEASUREMENT_CODE = ("agent", "browser", "lab", "defense", "oracle", "llm", "run")
_MEASUREMENT_FILES = ("config.py", "controls.py", "seeds.py", "types.py")
# never executed by an episode: the control server for external frameworks (audits)
_NOT_EPISODE_CODE = ("lab/serve.py",)
# executed only by the screenshot arms in the headful lab
_HEADFUL_ONLY = ("browser/capture.py",)


def code_hash(driver: str = "chromium-headful") -> str:
    """sha256 over the code that can affect an episode run with `driver`: binds results to the
    exact harness/oracle/prompt code (the screenshot-capture module counts only for the lab)."""
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    skip = set(_NOT_EPISODE_CODE) | (set() if driver == "chromium-headful" else set(_HEADFUL_ONLY))
    files = [p for d in _MEASUREMENT_CODE for p in (root / d).rglob("*.py")
             if str(p.relative_to(root)) not in skip]
    files += [root / f for f in _MEASUREMENT_FILES]
    for p in sorted(files):
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def run_identity(spec: ExperimentSpec) -> dict:
    """Everything that must match for two invocations to append to one result file."""
    return {"config_hash": spec.config_hash(), "plan_sha256": plan_hash(spec),
            "code_sha256": code_hash(spec.driver)}
