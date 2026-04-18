import pytest

from src.agents import get_agent, AGENT_NAMES
from src.agents.base import BaseAgent


def test_agent_names_covers_three_expected_adapters():
    assert set(AGENT_NAMES) == {"claude", "codex", "gemini"}


def test_get_agent_returns_baseagent_subclass():
    for name in AGENT_NAMES:
        agent = get_agent(name)
        assert isinstance(agent, BaseAgent)


def test_get_agent_unknown_raises():
    with pytest.raises(KeyError):
        get_agent("bogus")
