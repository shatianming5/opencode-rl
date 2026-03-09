from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class CmdResult:
    cmd: str
    rc: int
    stdout: str
    stderr: str
    timed_out: bool = False


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
    raw: Any | None = None
    tool_trace: list[dict[str, Any]] | None = None
    error: str | None = None


class AgentClient(Protocol):
    def run(self, text: str, *, on_turn: Callable[[TurnEvent], None] | None = None) -> AgentResult: ...
    def close(self) -> None: ...
