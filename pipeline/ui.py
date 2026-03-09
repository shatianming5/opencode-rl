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
# Stream printer (on_turn callback for agent phases)
# ---------------------------------------------------------------------------
def make_stream_printer(label: str):
    """Return an on_turn callback that prints agent events in real-time.

    Displays:
    - Tool calls with full input and output (no truncation)
    - Agent text responses
    - Token statistics per step
    - Turn progress
    """
    start = time.time()
    last_turn = [0]
    console.print(f"  [{STYLE_DIM}][{label}] Waiting for agent...[/]")

    def _print(event):
        elapsed = time.time() - start

        if event.finished:
            console.print(
                f"  [{STYLE_SUCCESS}][{label}] Done[/] "
                f"[dim](turn {event.turn}, {elapsed:.0f}s)[/]"
            )
            return

        # New turn header
        if event.turn != last_turn[0] and event.event_type == "step_start":
            last_turn[0] = event.turn
            console.print(
                f"\n  [{STYLE_SUBHEADER}][{label}][/] "
                f"Turn {event.turn} [dim]({elapsed:.0f}s)[/]"
            )
            return

        # Tool running — show what's being called
        if event.event_type == "tool_running":
            tool = event.tool_name
            title = event.tool_title
            inp = event.tool_input

            if tool == "bash":
                cmd = str(inp.get("command", ""))
                console.print(f"    [{STYLE_TOOL}]> bash:[/] {cmd}")
            elif tool in ("read", "glob", "grep"):
                detail = title or str(inp)
                console.print(f"    [{STYLE_TOOL}]> {tool}:[/] {detail}")
            elif tool in ("write", "edit"):
                path = str(inp.get("filePath", inp.get("file_path", "")))
                console.print(f"    [{STYLE_TOOL}]> {tool}:[/] {path}")
            else:
                detail = title or tool
                console.print(f"    [{STYLE_TOOL}]> {detail}[/]")
            return

        # Tool completed — show full output
        if event.event_type == "tool_completed":
            tool = event.tool_name
            output = event.tool_output

            if tool == "bash":
                if output:
                    # Show full output, indented
                    for line in output.splitlines():
                        console.print(f"      [dim]{line}[/]")
                else:
                    console.print(f"      [dim](no output)[/]")
            elif tool in ("read", "glob", "grep"):
                if output:
                    lines = output.splitlines()
                    for line in lines:
                        console.print(f"      [dim]{line}[/]")
                else:
                    console.print(f"      [dim](empty)[/]")
            elif tool in ("write", "edit"):
                console.print(f"      [green]ok[/]")
            else:
                if output:
                    for line in output.splitlines()[:20]:
                        console.print(f"      [dim]{line}[/]")
                    if len(output.splitlines()) > 20:
                        console.print(f"      [dim]... ({len(output.splitlines())} lines total)[/]")
            return

        # Tool error
        if event.event_type == "tool_error":
            console.print(f"      [red]{event.tool_output}[/]")
            return

        # Agent text response
        if event.event_type == "text" and event.assistant_text:
            text = event.assistant_text.strip()
            if text:
                # Show first few lines of the response
                lines = text.splitlines()
                preview = lines[:5]
                for line in preview:
                    console.print(f"    [{STYLE_AGENT}]Agent:[/] {line}")
                if len(lines) > 5:
                    console.print(f"    [{STYLE_DIM}]... ({len(lines)} lines total)[/]")
            return

        # Step finish — token stats
        if event.event_type == "step_finish":
            tokens = event.tokens
            if tokens:
                inp = tokens.get("input", 0)
                out = tokens.get("output", 0)
                reasoning = tokens.get("reasoning", 0)
                cache = tokens.get("cache", {})
                cache_r = cache.get("read", 0) if isinstance(cache, dict) else 0
                parts = [f"in={inp:,}", f"out={out:,}"]
                if reasoning:
                    parts.append(f"reasoning={reasoning:,}")
                if cache_r:
                    parts.append(f"cache_read={cache_r:,}")
                console.print(f"    [dim][tokens] {' | '.join(parts)}[/]")
            if event.cost:
                console.print(f"    [dim][cost] ${event.cost:.4f}[/]")
            return

    return _print
