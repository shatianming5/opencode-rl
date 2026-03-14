from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TurnEvent:
    """Agent turn event for stream display."""
    turn: int
    event_type: str = ""         # step_start, text, tool_running, tool_completed, tool_error, step_finish, finished
    assistant_text: str = ""
    tool_name: str = ""
    tool_input: dict = field(default_factory=dict)
    tool_output: str = ""
    tool_title: str = ""
    tokens: dict = field(default_factory=dict)
    cost: float = 0.0
    finished: bool = False


@dataclass(frozen=True)
class AgentResult:
    assistant_text: str
    tool_trace: list[dict[str, Any]] | None = None
    error: str | None = None
    total_cost: float = 0.0
    total_tokens: dict[str, Any] = field(default_factory=dict)


