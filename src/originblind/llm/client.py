"""OpenAI-compatible chat client with per-model sampling presets.

Targets a vLLM server's /v1 endpoint. A pluggable `complete` interface lets tests inject a
deterministic backend (see llm/mock.py) without a network or GPU.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Message:
    role: str
    content: object  # str, or a list of content parts for multimodal


@dataclass
class Sampling:
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    max_tokens: int = 512
    seed: int | None = None
    extra_body: dict = field(default_factory=dict)


class Backend(Protocol):
    def complete(self, model: str, messages: list[Message], sampling: Sampling) -> str: ...


class InfraError(RuntimeError):
    """The serving infrastructure failed (unreachable, timeout, 5xx, unknown model)."""


@dataclass
class OpenAIBackend:
    base_url: str = field(default_factory=lambda: os.environ.get("ORIGIN_BLIND_VLLM_URL", "http://localhost:8000/v1"))
    api_key: str = "EMPTY"
    timeout: float = 120.0
    max_retries: int = 1
    _client: object = field(default=None, repr=False)

    def _get(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout,
                                  max_retries=self.max_retries)
        return self._client

    def complete(self, model: str, messages: list[Message], sampling: Sampling) -> str:
        import openai

        client = self._get()
        payload = [{"role": m.role, "content": m.content} for m in messages]
        kw = dict(
            model=model, messages=payload, temperature=sampling.temperature,
            top_p=sampling.top_p, max_tokens=sampling.max_tokens,
        )
        if sampling.seed is not None:
            kw["seed"] = sampling.seed
        extra = {"top_k": sampling.top_k, **sampling.extra_body}
        try:
            resp = client.chat.completions.create(**kw, extra_body=extra)
        except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError,
                openai.NotFoundError, openai.RateLimitError) as e:
            raise InfraError(f"{type(e).__name__}: {str(e)[:200]}") from e
        except openai.BadRequestError as e:
            # e.g. prompt longer than max_model_len: an instrument limit, not agent behaviour
            raise InfraError(f"BadRequest: {str(e)[:200]}") from e
        return resp.choices[0].message.content or ""

    def served_models(self) -> list[dict]:
        """The /v1/models payload (id, root, max_model_len) for identity checks."""
        try:
            data = self._get().models.list()
        except Exception as e:  # noqa: BLE001
            raise InfraError(f"cannot list models: {e!r}") from e
        return [m.model_dump() if hasattr(m, "model_dump") else dict(m) for m in data.data]


def preset_for(model_key: str, thinking: bool = False) -> Sampling:
    if model_key.startswith("qwen3.5") or model_key.startswith("qwen3"):
        mt = 2048 if thinking else 512
        return Sampling(temperature=0.7, top_p=0.8, top_k=20, max_tokens=mt,
                        extra_body={"chat_template_kwargs": {"enable_thinking": bool(thinking)}})
    if model_key.startswith("qwen2.5"):
        return Sampling(temperature=0.7, top_p=0.8, top_k=20)
    if model_key.startswith("gemma"):
        return Sampling(temperature=0.7, top_p=0.9, top_k=64)
    if model_key.startswith("ministral"):
        # model card: "use a temperature below 0.1" for production/agentic use
        return Sampling(temperature=0.05, top_p=1.0, top_k=-1)
    if model_key.startswith("granite"):
        # the Granite 4.1 model card gives no sampling recommendation; its examples decode greedily
        return Sampling(temperature=0.0, top_p=1.0, top_k=-1)
    if model_key.startswith("uitars") or model_key.startswith("ui-tars"):
        return Sampling(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=256)
    return Sampling()
