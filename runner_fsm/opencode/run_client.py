"""Agent client using `opencode serve` HTTP API + SSE events.

Starts a headless opencode server, creates a session, subscribes to SSE
events, then sends the prompt and collects structured events until the
session becomes idle.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests

from ..dtypes import AgentResult, TurnEvent

# ---------------------------------------------------------------------------
# Global serve-process management
# ---------------------------------------------------------------------------
_serve_proc: subprocess.Popen | None = None
_serve_lock = threading.Lock()
_serve_port: int = 0


def _find_free_port() -> int:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_serve(repo: Path) -> int:
    """Start a shared opencode serve process, return port."""
    global _serve_proc, _serve_port
    with _serve_lock:
        # Already running?
        if _serve_proc is not None and _serve_proc.poll() is None:
            return _serve_port

        if not shutil.which("opencode"):
            raise RuntimeError("`opencode` not found in PATH")

        port = _find_free_port()
        cmd = ["opencode", "serve", "--port", str(port), "--hostname", "127.0.0.1"]
        _serve_proc = subprocess.Popen(
            cmd,
            cwd=str(repo.resolve()),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        _serve_port = port

        # Wait for the server to be ready (up to 15s)
        base = f"http://127.0.0.1:{port}"
        for _ in range(30):
            time.sleep(0.5)
            if _serve_proc.poll() is not None:
                raise RuntimeError("opencode serve exited unexpectedly")
            try:
                r = requests.get(f"{base}/session", timeout=2)
                if r.status_code < 500:
                    return port
            except requests.ConnectionError:
                continue
        raise RuntimeError(f"opencode serve did not become ready on port {port}")


def cleanup_active_procs() -> None:
    """Kill the shared opencode serve process."""
    global _serve_proc
    with _serve_lock:
        if _serve_proc is None:
            return
        try:
            os.killpg(os.getpgid(_serve_proc.pid), signal.SIGTERM)
            _serve_proc.wait(timeout=5)
        except Exception:
            try:
                _serve_proc.kill()
            except Exception:
                pass
        _serve_proc = None


# ---------------------------------------------------------------------------
# RunClient
# ---------------------------------------------------------------------------
@dataclass
class RunClient:
    """Agent client that drives opencode via its HTTP serve API + SSE."""

    repo: Path
    model: str = ""
    timeout_seconds: int = 600

    def run(
        self,
        prompt: str,
        *,
        on_turn: Callable[[TurnEvent], None] | None = None,
    ) -> AgentResult:
        port = _start_serve(self.repo)
        base = f"http://127.0.0.1:{port}"

        # 1. Create session
        resp = requests.post(
            f"{base}/session",
            json={},
            timeout=10,
        )
        resp.raise_for_status()
        session_id = resp.json()["id"]

        # 2. Build model spec
        model_id = self.model or os.environ.get("OPENCODE_MODEL", "")
        provider_id = os.environ.get("OPENCODE_PROVIDER", "openai")

        # 3. Start SSE listener thread BEFORE sending message
        collector = _EventCollector(
            base_url=base,
            session_id=session_id,
            on_turn=on_turn,
            timeout=self.timeout_seconds,
        )
        collector.start()

        # 4. Send message in background thread (the POST blocks until LLM finishes)
        msg_error: list[Exception] = []

        def _send_message():
            try:
                requests.post(
                    f"{base}/session/{session_id}/message",
                    json={
                        "parts": [{"type": "text", "text": prompt}],
                        "model": {"providerID": provider_id, "modelID": model_id},
                    },
                    timeout=self.timeout_seconds,
                )
            except Exception as e:
                msg_error.append(e)

        sender = threading.Thread(target=_send_message, daemon=True)
        sender.start()

        # 5. Wait for completion (SSE collector signals done on session.idle)
        collector.wait()
        sender.join(timeout=5)

        if msg_error and not collector._done.is_set():
            collector.stop()
            raise RuntimeError(f"Failed to send message: {msg_error[0]}")

        return collector.result()

    def close(self) -> None:
        """No-op for API compatibility (serve process is shared)."""
        pass


# ---------------------------------------------------------------------------
# SSE event collector
# ---------------------------------------------------------------------------
class _EventCollector:
    """Subscribes to opencode SSE /event and collects agent events."""

    def __init__(
        self,
        base_url: str,
        session_id: str,
        on_turn: Callable[[TurnEvent], None] | None,
        timeout: int,
    ):
        self._base_url = base_url
        self._session_id = session_id
        self._on_turn = on_turn
        self._timeout = timeout

        self._assistant_texts: list[str] = []
        self._trace: list[dict[str, Any]] = []
        self._current_step = 0
        self._done = threading.Event()
        self._error: str | None = None
        self._thread: threading.Thread | None = None
        self._stop_flag = threading.Event()

        # Track accumulated text per part for delta assembly
        self._text_parts: dict[str, str] = {}
        # Track tool state by partID
        self._tool_parts: dict[str, dict[str, Any]] = {}
        # Track total cost/tokens from the last message.updated
        self._total_cost = 0.0
        self._total_tokens: dict[str, Any] = {}
        # Cache message roles (user vs assistant)
        self._msg_roles: dict[str, str] = {}

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_flag.set()
        self._done.set()

    def wait(self) -> None:
        self._done.wait(timeout=self._timeout + 30)
        if self._thread:
            self._thread.join(timeout=5)

    def result(self) -> AgentResult:
        return AgentResult(
            assistant_text="\n".join(self._assistant_texts),
            tool_trace=self._trace,
            error=self._error,
            total_cost=self._total_cost,
            total_tokens=dict(self._total_tokens),
        )

    def _safe_callback(self, event: TurnEvent) -> None:
        if self._on_turn is None:
            return
        try:
            self._on_turn(event)
        except Exception:
            pass

    def _run(self) -> None:
        deadline = time.time() + self._timeout
        try:
            resp = requests.get(
                f"{self._base_url}/event",
                headers={"Accept": "text/event-stream"},
                stream=True,
                timeout=(5, None),  # connect timeout 5s, no read timeout
            )
            resp.raise_for_status()

            for line in resp.iter_lines(decode_unicode=True):
                if self._stop_flag.is_set():
                    break
                if time.time() > deadline:
                    self._error = f"Timed out after {self._timeout}s"
                    break

                if not line or not line.startswith("data: "):
                    continue

                raw = line[6:]  # strip "data: " prefix
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                self._handle_event(event)

                if self._done.is_set():
                    break

        except Exception as e:
            if not self._stop_flag.is_set():
                self._error = f"SSE error: {e}"
        finally:
            self._done.set()

    def _handle_event(self, event: dict) -> None:
        etype = event.get("type", "")
        props = event.get("properties", {})

        # Filter events to our session
        sid = props.get("sessionID", "")
        if sid and sid != self._session_id:
            return

        if etype == "session.idle":
            if props.get("sessionID") == self._session_id:
                self._safe_callback(TurnEvent(
                    turn=self._current_step,
                    event_type="finished",
                    finished=True,
                ))
                self._done.set()

        elif etype == "session.error":
            if props.get("sessionID") == self._session_id:
                err = props.get("error", {})
                name = err.get("name", "Error")
                data = err.get("data", {})
                msg = data.get("message", "") if isinstance(data, dict) else str(data)
                self._error = f"{name}: {msg}"
                self._done.set()

        elif etype == "message.part.updated":
            self._handle_part_updated(props)

        elif etype == "message.part.delta":
            self._handle_part_delta(props)

        elif etype == "message.updated":
            self._handle_message_updated(props)

    def _handle_part_updated(self, props: dict) -> None:
        part = props.get("part", {})
        part_type = part.get("type", "")
        sid = part.get("sessionID", "")
        if sid != self._session_id:
            return

        msg_info = self._get_msg_role(part.get("messageID", ""))

        if part_type == "step-start":
            self._current_step += 1
            self._safe_callback(TurnEvent(
                turn=self._current_step,
                event_type="step_start",
            ))

        elif part_type == "text":
            text = part.get("text", "")
            part_id = part.get("id", "")
            # This is the final text (may come after deltas, or standalone)
            if text:
                # Only count assistant text, not user text
                if msg_info == "assistant":
                    # Replace any delta-accumulated text with final
                    if part_id in self._text_parts:
                        self._text_parts[part_id] = text
                    else:
                        self._text_parts[part_id] = text
                    self._assistant_texts = list(self._text_parts.values())
                    self._safe_callback(TurnEvent(
                        turn=self._current_step,
                        event_type="text",
                        assistant_text=text,
                    ))

        elif part_type == "tool":
            self._handle_tool_event(part)

        elif part_type == "step-finish":
            tokens = part.get("tokens", {})
            cost = part.get("cost", 0)
            self._safe_callback(TurnEvent(
                turn=self._current_step,
                event_type="step_finish",
                tokens=tokens,
                cost=cost,
            ))

    def _handle_tool_event(self, part: dict) -> None:
        tool_name = part.get("tool", "")
        state = part.get("state", {})
        status = state.get("status", "")
        part_id = part.get("id", "")

        if status == "running":
            # Deduplicate: only fire callback once per tool part
            if part_id in self._tool_parts:
                return
            tool_input = state.get("input", {})
            title = state.get("title", "")
            self._tool_parts[part_id] = {
                "tool": tool_name,
                "input": tool_input,
                "title": title,
            }
            self._safe_callback(TurnEvent(
                turn=self._current_step,
                event_type="tool_running",
                tool_name=tool_name,
                tool_input=tool_input,
                tool_title=title,
            ))

        elif status == "completed":
            tool_input = state.get("input", {})
            tool_output = state.get("output", "")
            title = state.get("title", "")
            timing = state.get("time", {})
            metadata = state.get("metadata", {})

            self._trace.append({
                "turn": self._current_step,
                "tool": tool_name,
                "input": tool_input,
                "output": tool_output,
                "title": title,
                "metadata": metadata,
                "time": timing,
            })
            self._safe_callback(TurnEvent(
                turn=self._current_step,
                event_type="tool_completed",
                tool_name=tool_name,
                tool_input=tool_input,
                tool_output=tool_output,
                tool_title=title,
            ))

        elif status == "error":
            error_msg = state.get("error", "unknown error")
            tool_input = state.get("input", {})
            self._trace.append({
                "turn": self._current_step,
                "tool": tool_name,
                "input": tool_input,
                "error": error_msg,
            })
            self._safe_callback(TurnEvent(
                turn=self._current_step,
                event_type="tool_error",
                tool_name=tool_name,
                tool_input=tool_input,
                tool_output=error_msg,
            ))

    def _handle_part_delta(self, props: dict) -> None:
        """Handle streaming text deltas (we don't display these, just accumulate)."""
        # Deltas are followed by a final message.part.updated with complete text.
        # We just ignore deltas and wait for the final update.
        pass

    def _handle_message_updated(self, props: dict) -> None:
        """Track cost/tokens and cache message roles."""
        info = props.get("info", {})
        if info.get("sessionID") != self._session_id:
            return
        msg_id = info.get("id", "")
        role = info.get("role", "")
        if msg_id and role:
            self._msg_roles[msg_id] = role
        if role != "assistant":
            return
        cost = info.get("cost", 0)
        tokens = info.get("tokens", {})
        if cost:
            self._total_cost = cost
        if tokens:
            self._total_tokens = tokens

    def _get_msg_role(self, msg_id: str) -> str:
        """Determine message role from cached info."""
        return self._msg_roles.get(msg_id, "assistant")
