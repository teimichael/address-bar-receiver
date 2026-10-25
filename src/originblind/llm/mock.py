"""Deterministic scripted LLM backends for tests and scripted-control agents.

A ScriptedBackend returns actions from a policy function keyed by the step's observation,
with no network or GPU. It underpins the ALWAYS-FILL / ORIGIN-CHECKER / NULL controls and
the end-to-end validity tests.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from originblind.llm.client import Message, Sampling


@dataclass
class ScriptedBackend:
    """policy(observation_text, step) -> action string."""

    policy: Callable[[str, int], str]
    calls: list[tuple[str, int]] = field(default_factory=list)
    _step: int = 0

    def complete(self, model: str, messages: list[Message], sampling: Sampling) -> str:
        obs = ""
        for m in reversed(messages):
            if m.role == "user":
                obs = m.content if isinstance(m.content, str) else _text_of(m.content)
                break
        action = self.policy(obs, self._step)
        self.calls.append((model, self._step))
        self._step += 1
        return action


def _text_of(parts) -> str:
    return " ".join(p.get("text", "") for p in parts if isinstance(p, dict))
