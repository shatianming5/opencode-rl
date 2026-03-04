"""Rich-based terminal UI for pipeline output.

Provides a shared Console and formatting helpers used across
runner.py, phases.py, and client.py.
"""

from __future__ import annotations

import time

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

console = Console(highlight=False)


# ---------------------------------------------------------------------------
# Colour / style tokens (centralised so every module is consistent)
# ---------------------------------------------------------------------------
STYLE_SUBHEADER = "bold cyan"
STYLE_KEY = "dim"
STYLE_VALUE = "bright_white"
STYLE_SUCCESS = "bold green"
STYLE_DIM = "dim"
STYLE_TOOL = "bold cyan"
STYLE_AGENT = "bright_magenta"
STYLE_PHASE_LABEL = "bold bright_blue"


# ---------------------------------------------------------------------------
# Shared KV table constructor
# ---------------------------------------------------------------------------
def _make_kv_table(key_min_width: int = 18, padding: tuple[int, int] = (0, 2),
                   value_style: str = STYLE_VALUE) -> Table:
    """Create a key-value table with consistent styling."""
    tbl = Table(show_header=False, box=None, padding=padding, show_edge=False)
    tbl.add_column("Key", style=STYLE_KEY, min_width=key_min_width)
    tbl.add_column("Value", style=value_style)
    return tbl


# ---------------------------------------------------------------------------
# Pipeline header & footer
# ---------------------------------------------------------------------------
def print_pipeline_header(task: str, base_model: str, workspace: str,
                          data_path: str, output_dir: str,
                          max_iterations: int, training_timeout: int,
                          max_agent_steps: int, opencode_model: str,
                          resume: bool):
    """Print the main pipeline banner with config table."""
    title = Text("OpenCode RL Pipeline", style="bold bright_white")

    tbl = _make_kv_table()

    tbl.add_row("Task", task)
    tbl.add_row("Base Model", base_model)
    tbl.add_row("Workspace", workspace)
    tbl.add_row("Data Path", data_path)
    tbl.add_row("Output Dir", output_dir)
    tbl.add_row("Max Iterations", str(max_iterations))
    tbl.add_row("Training Timeout", f"{training_timeout}s")
    tbl.add_row("Agent Steps/Iter", str(max_agent_steps))
    tbl.add_row("OpenCode Model", opencode_model or "(env)")
    tbl.add_row("Resume", str(resume))

    console.print(Panel(tbl, title=title, border_style="bright_blue",
                        box=box.DOUBLE_EDGE, padding=(1, 2)))


def print_data_gpu_info(data_count: int, gpu_count: int, gpu_name: str):
    """Print data & GPU summary line."""
    console.print(f"  [dim]Data:[/] [bright_white]{data_count} samples[/]"
                  f"   [dim]GPU:[/] [bright_white]{gpu_count}x {gpu_name}[/]")


def print_pipeline_footer(best_score, best_iteration, total_iters, total_time):
    """Print the final summary panel."""
    tbl = _make_kv_table(key_min_width=16)

    score_str = f"{best_score}" if best_score is not None else "N/A"
    tbl.add_row("Best Score", f"[bold bright_green]{score_str}[/]")
    tbl.add_row("Best Iteration", str(best_iteration) if best_iteration is not None and best_iteration >= 0 else "N/A")
    tbl.add_row("Total Iterations", str(total_iters))
    tbl.add_row("Total Time", f"{total_time:.0f}s")

    console.print(Panel(tbl,
                        title=Text("Pipeline Complete", style="bold bright_green"),
                        border_style="green",
                        box=box.DOUBLE_EDGE,
                        padding=(1, 2)))


# ---------------------------------------------------------------------------
# Iteration header & summary
# ---------------------------------------------------------------------------
def print_iteration_header(iteration: int, max_iterations: int, elapsed: float):
    """Print iteration banner."""
    console.print()
    console.print(Rule(
        f"[bold]ITERATION {iteration}/{max_iterations}[/]  "
        f"[dim](elapsed {elapsed:.0f}s)[/]",
        style="bright_yellow",
    ))


def print_iteration_summary(iteration, score, improvement,
                            best_score, best_iteration, elapsed):
    """Print compact iteration summary."""
    tbl = _make_kv_table(padding=(0, 1), value_style="")

    s = f"[bright_green]{score}[/]" if score is not None else "[dim]N/A[/]"
    imp = f"{improvement}" if improvement is not None else "N/A"
    b = f"[bold bright_green]{best_score}[/] (iter {best_iteration})" if best_score is not None else "N/A"

    tbl.add_row("Score", s)
    tbl.add_row("vs Baseline", imp)
    tbl.add_row("Best So Far", b)
    tbl.add_row("Iteration Time", f"{elapsed:.0f}s")

    console.print(Panel(tbl,
                        title=Text(f"Iteration {iteration} Summary", style="bold"),
                        border_style="bright_yellow", box=box.ROUNDED,
                        padding=(0, 2)))


# ---------------------------------------------------------------------------
# Phase headers
# ---------------------------------------------------------------------------
def print_phase_header(phase_name: str, subtitle: str = ""):
    """Print a phase divider."""
    label = f"[{STYLE_PHASE_LABEL}]{phase_name}[/]"
    if subtitle:
        label += f"  [dim]{subtitle}[/]"
    console.print()
    console.print(Rule(label, style="blue"))


# ---------------------------------------------------------------------------
# Evaluation report
# ---------------------------------------------------------------------------
def print_evaluation_report(score, improvement, best_score, submission_id=None):
    """Print evaluation results panel."""
    tbl = _make_kv_table(key_min_width=20, value_style="")

    tbl.add_row("Source", "[green]Grading Server[/]")
    tbl.add_row("Score",
                f"[bold bright_green]{score}[/]" if score is not None else "N/A")
    tbl.add_row("vs Baseline",
                f"{improvement}" if improvement is not None else "N/A")
    tbl.add_row("Best Score",
                f"[bold bright_green]{best_score}[/]" if best_score is not None else "N/A")
    if submission_id is not None:
        tbl.add_row("Submission ID", str(submission_id))

    console.print(Panel(tbl,
                        title=Text("Evaluation Results", style="bold"),
                        border_style="bright_green", box=box.ROUNDED,
                        padding=(0, 2)))


# ---------------------------------------------------------------------------
# Stream printer (Turn-level output helpers)
# ---------------------------------------------------------------------------
def print_turn_header(label: str, turn: int, elapsed: float):
    """Print a turn header for agent interaction."""
    console.print(
        f"\n  [{STYLE_SUBHEADER}][{label}][/] "
        f"Turn {turn} [dim]({elapsed:.0f}s)[/]"
    )


def print_turn_done(label: str, turn: int, elapsed: float):
    """Print turn completion."""
    console.print(
        f"  [{STYLE_SUCCESS}][{label}] Done[/] "
        f"[dim](turn {turn}, {elapsed:.0f}s)[/]"
    )


def print_turn_waiting(label: str):
    """Print waiting message."""
    console.print(f"  [{STYLE_DIM}][{label}] Waiting for agent response...[/]")


def print_agent_thought(text: str):
    """Print agent reasoning summary."""
    console.print(f"    [{STYLE_AGENT}]Agent:[/] {text}")


def print_tool_call(kind: str, detail: str):
    """Print a tool call (from Turn data, not proxy)."""
    console.print(f"    [{STYLE_TOOL}]> {kind}:[/] {detail}")


def print_tool_result(text: str, ok: bool = True):
    """Print a tool call result."""
    style = "green" if ok else "red"
    console.print(f"    [{style}]< {text}[/]")


# ---------------------------------------------------------------------------
# Stream printer (on_turn callback for agent phases)
# ---------------------------------------------------------------------------
def make_stream_printer(label: str):
    """Return an on_turn callback that prints agent turns in real-time."""
    start = time.time()
    last_turn = [0]
    print_turn_waiting(label)

    def _print(event):
        elapsed = time.time() - start

        if event.finished:
            print_turn_done(label, event.turn, elapsed)
            return

        if event.turn != last_turn[0]:
            last_turn[0] = event.turn
            print_turn_header(label, event.turn, elapsed)

        if event.assistant_text:
            lines = [l.strip() for l in event.assistant_text.strip().splitlines() if l.strip()]
            for line in lines:
                if not line.startswith("<") and not line.startswith("```"):
                    print_agent_thought(line[:200])
                    break

        results = list(event.results) if event.results else []
        for i, call in enumerate(event.calls or []):
            result = results[i] if i < len(results) else None
            payload = call.payload if isinstance(call.payload, dict) else {}

            if call.kind == "bash":
                cmd = str(payload.get("command", ""))
                if len(cmd) > 150:
                    cmd = cmd[:147] + "..."
                print_tool_call("bash", cmd)
                if result:
                    rc = result.detail.get("rc", "?")
                    stdout = str(result.detail.get("stdout") or "").strip()
                    stderr = str(result.detail.get("stderr") or "").strip()
                    if result.ok:
                        out = stdout[:200].replace("\n", " | ") if stdout else ""
                        print_tool_result(f"(rc={rc}) {out}", ok=True)
                    else:
                        err = stderr[:200].replace("\n", " | ") if stderr else stdout[:200].replace("\n", " | ")
                        print_tool_result(f"(rc={rc}) {err}", ok=False)

            elif call.kind == "file":
                file_path = str(payload.get("filePath", ""))
                if result is None:
                    print_tool_call("file", file_path)
                    continue
                kind = result.kind
                ok_str = "ok" if result.ok else str(result.detail.get("error", "failed"))
                if kind == "read":
                    print_tool_call("read", file_path)
                    content = str(result.detail.get("content") or "")
                    print_tool_result(f"ok ({len(content)} chars)" if result.ok else ok_str, ok=result.ok)
                elif kind in ("write", "edit"):
                    print_tool_call(kind, file_path)
                    if result.ok:
                        print_tool_result(f"ok ({result.detail.get('bytes', 0)} bytes, {result.detail.get('mode', kind)})", ok=True)
                    else:
                        print_tool_result(ok_str, ok=False)
                else:
                    print_tool_call(kind, file_path)
                    print_tool_result(ok_str, ok=result.ok)

    return _print
