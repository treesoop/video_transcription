"""Registry of pluggable CLI agent adapters."""
from __future__ import annotations

from .base import BaseAgent, AgentInvocationError
from .claude import ClaudeAgent
from .codex import CodexAgent
from .gemini import GeminiAgent

_REGISTRY: dict[str, type[BaseAgent]] = {
    "claude": ClaudeAgent,
    "codex": CodexAgent,
    "gemini": GeminiAgent,
}

AGENT_NAMES = tuple(_REGISTRY.keys())


def get_agent(name: str) -> BaseAgent:
    return _REGISTRY[name]()


__all__ = ["BaseAgent", "AgentInvocationError", "get_agent", "AGENT_NAMES"]
