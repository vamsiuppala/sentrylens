"""
Error triage agent module.

Provides ReAct agent for multi-turn reasoning about errors with tool integration.
"""

from sentrylens.agent.tools import TriageTools
from sentrylens.agent.triage_agent import TriageAgent

__all__ = [
    "TriageAgent",
    "TriageTools",
]
