"""Experiment configuration and deterministic plan identity.

An ExperimentSpec is a factorial over the fields below. Its canonical JSON encoding hashes to a
stable `config_hash`; the realised plan hashes to a `plan_sha256`. Both go in every run manifest
and gate resumption, so episodes produced under different designs can never be mixed.

Two expansion modes:
  - legacy (`replicates`): replicate r uses brand r % len(brands). Kept ONLY so the October-2026
    pilot configs in configs/legacy/ re-expand to the identical plan.
  - balanced (`replicates_per_brand`): every brand appears in every cell the same number of times
    (brand is an explicit factor), exact-origin cells are oversampled by `exact_oversample`
    (they are the denominator of every relative-release estimate), and the secret is determined
    by the flow (login->password, otp->otp, checkout->card, profile->pii, psp*->card).
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from originblind.types import OriginClass, SecretKind

Arm = Literal["T0", "T1", "T1u", "T2", "S0", "S1", "S2"]
DefenseId = Literal["D0", "D1", "D2", "D3", "D3info", "D3push", "D5", "D6", "D5r", "D6r", "DBU"]
Flow = Literal["login", "otp", "checkout", "profile", "psp", "psp_nocsp"]
Semantics = Literal["neutral", "persuasive"]
Driver = Literal["inprocess", "chromium", "chromium-headful"]
Pipeline = Literal["text", "vision", "both"]

SCREEN_ARMS = ("S0", "S1", "S2")
FLOW_SECRET: dict[str, SecretKind] = {
    "login": SecretKind.PASSWORD, "otp": SecretKind.OTP, "checkout": SecretKind.CARD,
    "profile": SecretKind.PII, "psp": SecretKind.CARD, "psp_nocsp": SecretKind.CARD,
}


class ModelSpec(BaseModel):
    key: str  # short id used in paths and episode ids
    hf_repo: str
    served_name: str  # model name presented by the vLLM OpenAI endpoint
    pipeline: Pipeline = "text"  # text-only models never get screenshot arms
    max_model_len: int = 12288
    thinking: bool = False  # Qwen3.x: chat_template_kwargs.enable_thinking
    revision: str | None = None  # HF commit; required for confirmatory namespaces
    family: str = ""
    vllm_args: list[str] = Field(default_factory=list)  # extra vLLM serve arguments
    workers: int | None = None  # concurrent episodes for this model (throughput only, not a factor)

    @property
    def vision(self) -> bool:
        return self.pipeline in ("vision", "both")


class ExperimentSpec(BaseModel):
    name: str
    master_seed_hex: str = "00" * 16
    brands: list[str] = Field(default_factory=list)  # empty => registry default (12 brands)
    models: list[ModelSpec] = Field(default_factory=list)
    origin_classes: list[OriginClass] = Field(default_factory=list)
    semantics: list[Semantics] = Field(default_factory=lambda: ["neutral"])
    secrets: list[SecretKind] = Field(default_factory=list)  # legacy only; balanced mode derives it
    flows: list[Flow] = Field(default_factory=lambda: ["login"])
    observation_arms: list[Arm] = Field(default_factory=lambda: ["T1"])
    defenses: list[DefenseId] = Field(default_factory=lambda: ["D0"])
    pii_mode: list[Literal["plaintext"]] = Field(default_factory=lambda: ["plaintext"])
    replicates: int = 0  # legacy mode
    replicates_per_brand: int = 0  # balanced mode
    exact_oversample: int = 1
    # restrict some arms to some origin classes, e.g. {"T1u": ["exact", "idn"]}
    arm_origins: dict[str, list[OriginClass]] = Field(default_factory=dict)
    driver: Driver = "inprocess"
    # stimulus environment: "lab" (.test infrastructure names) or "realistic" (no "test" cue)
    environment: Literal["lab", "realistic"] = "lab"
    step_budget: int = 12
    reasoning: bool = True  # brief reasoning before the action (framework-like)
    # D3info/D3push: describe the pushed ORIGIN FACTS/CHECK section in the system prompt, exactly as
    # the D3 tool is described (push vs pull then differ only in delivery). False = pilot v2 prompts.
    push_legend: bool = False
    recognition_probe: bool = True
    discrimination_probe: bool = True
    probe_k: int = 3

    @model_validator(mode="after")
    def _validate(self) -> "ExperimentSpec":
        if bool(self.replicates) == bool(self.replicates_per_brand):
            raise ValueError("set exactly one of `replicates` (legacy) or `replicates_per_brand`")
        if self.replicates_per_brand and self.secrets:
            bad = [s for s in self.secrets if s not in {FLOW_SECRET[f] for f in self.flows}]
            if bad:
                raise ValueError(f"secrets {bad} do not match flows {self.flows}; in balanced mode the "
                                 "secret is determined by the flow")
        if self.replicates:  # legacy: refuse incoherent (flow, secret) crossings
            for f in self.flows:
                for s in self.secrets or [FLOW_SECRET[f]]:
                    if FLOW_SECRET[f] is not s:
                        raise ValueError(f"incoherent (flow, secret) pair ({f}, {s.value}): the {f} page "
                                         f"asks for {FLOW_SECRET[f].value}")
        screen = [a for a in self.observation_arms if a in SCREEN_ARMS]
        if screen:
            if self.driver != "chromium-headful":
                raise ValueError(f"screenshot arms {screen} need driver: chromium-headful (the lab)")
            if not any(m.vision for m in self.models):
                raise ValueError("screenshot arms need at least one vision-capable model")
        if self.exact_oversample < 1:
            raise ValueError("exact_oversample must be >= 1")
        return self

    @property
    def master_seed(self) -> bytes:
        return bytes.fromhex(self.master_seed_hex)

    @property
    def balanced(self) -> bool:
        return bool(self.replicates_per_brand)

    def secret_for(self, flow: str) -> SecretKind:
        return FLOW_SECRET[flow]

    def config_hash(self) -> str:
        # throughput-only fields (per-model concurrency) are not part of the design identity
        data = self.model_dump(mode="json", exclude={"models": {"__all__": {"workers"}}})
        if data.get("environment") == "lab":
            data.pop("environment")  # legacy default: pilot identities stay unchanged
        if data.get("push_legend") is False:
            data.pop("push_legend")  # legacy default: pilot identities stay unchanged
        payload = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def load_yaml_spec(path: str | Path) -> dict:
    """Load a config YAML, resolving `roster:` (a file of ModelSpecs) and model keys given as
    strings, relative to the config file."""
    import yaml
    from pathlib import Path as _P

    path = _P(path)
    data = yaml.safe_load(path.read_text())
    roster_ref = data.pop("roster", None)
    if roster_ref:
        roster = yaml.safe_load((path.parent / roster_ref).read_text())
        by_key = {m["key"]: m for m in roster["models"]}
        models = []
        for m in data.get("models", []):
            if isinstance(m, str):
                if m not in by_key:
                    raise ValueError(f"model {m!r} not in roster {roster_ref}")
                models.append(dict(by_key[m]))
            else:
                base = dict(by_key.get(m.get("key"), {}))
                base.update(m)
                models.append(base)
        data["models"] = models
    return data


def load_spec(path: str | Path) -> ExperimentSpec:
    return ExperimentSpec(**load_yaml_spec(path))
