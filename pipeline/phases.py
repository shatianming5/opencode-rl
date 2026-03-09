"""Pipeline 各阶段执行逻辑。

所有阶段函数统一返回 PhaseResult。
"""

import os
import subprocess
import time
from pathlib import Path

import requests

from runner_fsm.opencode.run_client import RunClient

from .prompts import (
    build_analysis_prompt,
    build_code_prompt,
    build_fix_prompt,
)
from .types import IterationResult, PhaseResult
from .ui import console, make_stream_printer, print_phase_header

# Maximum lines to collect from subprocess output to prevent OOM
_MAX_COLLECTED_LINES = 50000
# Maximum stdout string size in bytes (20MB: 2MB head + 18MB tail)
_MAX_STDOUT_BYTES = 20 * 1024 * 1024
_STDOUT_HEAD_BYTES = 2 * 1024 * 1024


def _resolve_model(opencode_model: str) -> str:
    """Resolve the effective model string (explicit or env fallback)."""
    return opencode_model or os.environ.get("OPENCODE_MODEL", "")


def _make_agent(
    workspace: str,
    opencode_model: str = "",
    stale_timeout: float = 180.0,
    http_timeout: float = 900.0,
) -> RunClient:
    """Create a RunClient for pipeline phases."""
    model = _resolve_model(opencode_model)
    repo_root = Path(workspace).resolve()
    return RunClient(
        repo=repo_root,
        model=model,
        timeout_seconds=int(http_timeout + stale_timeout),
    )


def _log_phase_start(prompt: str, opencode_model: str) -> None:
    """Print shared phase startup info (prompt size, model, starting server)."""
    model = _resolve_model(opencode_model)
    console.print(f"  [dim]Prompt:[/] {len(prompt)} chars  [dim]Model:[/] {model}")


def _save_agent_log(
    log_dir: Path,
    purpose: str,
    iteration: int,
    result: "AgentResult | None",
    error_msg: str = "",
) -> None:
    """Save agent log: assistant text + tool trace + any error."""
    log_parts: list[str] = []
    if error_msg:
        log_parts.append(f"[ERROR] {error_msg}")
    if result:
        if result.assistant_text:
            log_parts.append(result.assistant_text[-20000:])
        if result.tool_trace:
            log_parts.append("\n--- Tool Trace ---")
            for t in result.tool_trace:
                tool = t.get("tool", "?")
                inp = str(t.get("input", {}))[:200]
                out = str(t.get("output", t.get("error", "")))[:500]
                log_parts.append(f"[{tool}] {inp}\n  => {out}")
    if log_parts:
        (log_dir / f"{purpose}_iter{iteration}_result.txt").write_text(
            "\n".join(log_parts), encoding="utf-8",
        )


def _run_agent(
    agent: RunClient,
    prompt: str,
    iteration: int,
    purpose: str,
    label: str,
    phase_name: str,
) -> PhaseResult | str:
    """Run agent with standard try/except/finally pattern.

    Returns PhaseResult on failure, or assistant_text on success.
    """
    log_dir = Path(agent.repo) / "code" / "agent_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    result = None
    try:
        result = agent.run(prompt, on_turn=make_stream_printer(label))
    except Exception as e:
        _save_agent_log(log_dir, purpose, iteration, result, str(e))
        return PhaseResult(
            success=False, phase=phase_name,
            error=f"Agent error: {e}",
        )
    finally:
        try:
            agent.close()
        except Exception:
            pass

    # Agent returned — check for error (e.g. timeout with partial data)
    if result.error:
        _save_agent_log(log_dir, purpose, iteration, result, result.error)
        return PhaseResult(
            success=False, phase=phase_name,
            error=f"Agent error: {result.error}",
        )

    _save_agent_log(log_dir, purpose, iteration, result)
    return result.assistant_text or ""


def _cap_stdout(collected: list[str]) -> str:
    """Join collected lines and cap total size, preserving head and tail.

    Estimates total size before joining to avoid a transient memory spike
    when the collected output is very large.
    """
    # Estimate total size: sum of line lengths + newline separators
    est_size = sum(len(line) for line in collected) + max(0, len(collected) - 1)
    if est_size <= _MAX_STDOUT_BYTES:
        return "\n".join(collected)

    # Over budget — build head and tail separately to limit peak memory
    head_lines: list[str] = []
    head_size = 0
    for line in collected:
        needed = len(line) + (1 if head_lines else 0)
        if head_size + needed > _STDOUT_HEAD_BYTES:
            break
        head_lines.append(line)
        head_size += needed

    tail_budget = _MAX_STDOUT_BYTES - _STDOUT_HEAD_BYTES - 100
    tail_lines: list[str] = []
    tail_size = 0
    for line in reversed(collected):
        needed = len(line) + (1 if tail_lines else 0)
        if tail_size + needed > tail_budget:
            break
        tail_lines.append(line)
        tail_size += needed
    tail_lines.reverse()

    return "\n".join(head_lines) + "\n...[TRUNCATED]...\n" + "\n".join(tail_lines)


def phase_code_generation(
    iteration: int,
    workspace: str,
    base_model: str,
    task_description: str,
    history: list[IterationResult],
    gpu_info: dict | None = None,
    opencode_model: str = "",
    stale_timeout: float = 180.0,
    http_timeout: float = 900.0,
    task_type: str = "math",
    expose_files: tuple[str, ...] = (),
) -> PhaseResult:
    """Phase 1: 代码生成 — 用 agent 探索数据并编写训练代码。"""
    print_phase_header("Code Generation", f"iteration {iteration}")

    prompt = build_code_prompt(
        workspace, base_model, task_description, history,
        gpu_info=gpu_info, task_type=task_type, expose_files=expose_files,
    )
    _log_phase_start(prompt, opencode_model)

    agent = _make_agent(workspace, opencode_model, stale_timeout, http_timeout)
    out = _run_agent(agent, prompt, iteration,
                     "code_generation", f"CodeGen iter{iteration}", "code_gen")
    if isinstance(out, PhaseResult):
        return out

    code_path = Path(workspace) / "code" / "train.py"
    if code_path.exists():
        console.print(f"  [green]Code generated:[/] {code_path} [dim]({code_path.stat().st_size} bytes)[/]")
        return PhaseResult(success=True, phase="code_gen", payload={"code_path": str(code_path)})
    console.print(f"  [bold yellow]WARNING:[/] {code_path} not found after agent finished")
    return PhaseResult(success=False, phase="code_gen", error=f"{code_path} not found",
                       payload={"code_path": str(code_path)})


def phase_training(
    workspace: str,
    code_path: str,
    timeout: int = 3600,
) -> PhaseResult:
    """Phase 2: 训练执行（pipeline 控制，非 agent）

    用 accelerate launch 执行 train.py，自动多卡 DDP。
    """
    print_phase_header("Training Execution")

    if not Path(code_path).exists():
        msg = f"Code file not found: {code_path}"
        console.print(f"  [bold red]ERROR:[/] {msg}")
        return PhaseResult(
            success=False, phase="training", error=msg,
        )

    abs_code_path = str(Path(code_path).resolve())
    start = time.time()

    cmd = ["accelerate", "launch", abs_code_path]
    console.print(f"  [dim]CMD:[/] {' '.join(cmd)}")

    collected: list[str] = []
    exit_code = -1

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(Path(workspace).resolve()),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            start_new_session=True,  # create process group for clean kill
        )
        try:
            deadline = time.time() + timeout
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    # Kill entire process group (parent + all children)
                    import signal
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                        proc.wait(timeout=10)
                    except Exception:
                        try:
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        except Exception:
                            proc.kill()
                        proc.wait()
                    collected.append(f"\n  TIMEOUT after {timeout}s")
                    console.print(f"  [bold red]TIMEOUT after {timeout}s[/]")
                    break
                line = proc.stdout.readline()
                if not line and proc.poll() is not None:
                    break
                if line:
                    line = line.rstrip("\n")
                    if len(collected) < _MAX_COLLECTED_LINES:
                        collected.append(line)
                    elif len(collected) == _MAX_COLLECTED_LINES:
                        collected.append(f"...[TRUNCATED at {_MAX_COLLECTED_LINES} lines]...")
                    console.print(f"  [dim]\\[train][/] {line}")
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass
            proc.wait()
        exit_code = proc.returncode if proc.returncode is not None else -1
    except Exception as e:
        collected.append(f"Exception: {e}")
        console.print(f"  [bold red]Exception:[/] {e}")

    stdout = _cap_stdout(collected)
    elapsed = time.time() - start
    ec_style = "green" if exit_code == 0 else "red"
    console.print(f"  [dim]Exit code:[/] [{ec_style}]{exit_code}[/]  [dim]Time:[/] {elapsed:.1f}s")

    if exit_code != 0:
        tail = "\n".join(stdout.strip().splitlines()[-15:])
        console.print(f"  [dim]Error tail:[/]\n{tail}")

    return PhaseResult(
        success=(exit_code == 0),
        phase="training",
        payload={
            "exit_code": exit_code,
            "stdout": stdout,
            "elapsed": elapsed,
        },
        error="" if exit_code == 0 else f"Training failed with exit code {exit_code}",
    )


def phase_evaluation(
    grading_url: str,
    output_dir: str,
    timeout: int = 600,
) -> PhaseResult:
    """Evaluation 阶段：找到训练产出的模型，POST 到 Grading Server。"""
    print_phase_header("Evaluation", "submit to Grading Server")

    # 1. 找 output_dir 下最新模型目录
    out_path = Path(output_dir)
    if not out_path.exists():
        return PhaseResult(
            success=False, phase="evaluation",
            error=f"Output dir not found: {output_dir}",
        )

    # 检测模型位置：优先使用 output_dir 根目录（有模型文件时），
    # 否则在子目录中找（排除 checkpoint-* 中间产物）
    _MODEL_MARKERS = {"config.json", "adapter_config.json", "model.safetensors",
                      "adapter_model.safetensors", "pytorch_model.bin"}
    model_path = str(out_path)
    root_has_model = bool(_MODEL_MARKERS & {f.name for f in out_path.iterdir() if f.is_file()})
    if not root_has_model:
        try:
            subdirs = [
                d for d in out_path.iterdir()
                if d.is_dir() and not d.name.startswith((".", "checkpoint-"))
            ]
            if subdirs:
                model_path = str(max(subdirs, key=lambda d: d.stat().st_mtime))
        except OSError:
            pass

    console.print(f"  [dim]Model path:[/] {model_path}")

    # 2. POST to Grading Server
    submit_url = f"{grading_url.rstrip('/')}/submit"
    console.print(f"  [dim]Submitting to:[/] {submit_url}")

    try:
        resp = requests.post(
            submit_url,
            json={"model_path": model_path},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError as e:
        return PhaseResult(
            success=False, phase="evaluation",
            error=f"Cannot connect to Grading Server at {grading_url}: {e}",
        )
    except requests.exceptions.Timeout:
        return PhaseResult(
            success=False, phase="evaluation",
            error=f"Grading Server request timed out ({timeout}s)",
        )
    except requests.exceptions.HTTPError as e:
        return PhaseResult(
            success=False, phase="evaluation",
            error=f"Grading Server returned error: {e}",
        )
    except Exception as e:
        return PhaseResult(
            success=False, phase="evaluation",
            error=f"Grading Server error: {e}",
        )

    score = data.get("score")
    improvement = data.get("improvement")
    best = data.get("best", {})
    submission_id = data.get("submission_id")

    console.print(f"  [green]Score:[/] {score}")
    if improvement is not None:
        console.print(f"  [dim]vs Baseline:[/] {improvement}")
    if best:
        console.print(f"  [dim]Best:[/] {best.get('score')}")

    return PhaseResult(
        success=True, phase="evaluation",
        payload={
            "score": score,
            "improvement": improvement,
            "best": best,
            "submission_id": submission_id,
            "model_path": model_path,
        },
    )


def phase_fix_training(
    code_path: str,
    error_log_path: str,
    data_path: str,
    workspace: str,
    iteration: int = 0,
    opencode_model: str = "",
    stale_timeout: float = 180.0,
    http_timeout: float = 900.0,
) -> PhaseResult:
    """训练失败后的修复尝试。"""
    fix_prompt = build_fix_prompt(code_path, error_log_path, data_path)
    _log_phase_start(fix_prompt, opencode_model)

    agent = _make_agent(workspace, opencode_model, stale_timeout, http_timeout)
    out = _run_agent(agent, fix_prompt, iteration,
                     "fix_training", "FixTraining", "fix_training")
    if isinstance(out, PhaseResult):
        return out
    return PhaseResult(success=True, phase="fix_training")


def phase_analysis(
    iteration: int,
    workspace: str,
    code_path: str,
    training_log_path: str,
    score: float | None = None,
    opencode_model: str = "",
    evaluation_summary: str = "",
    stale_timeout: float = 180.0,
    http_timeout: float = 900.0,
) -> PhaseResult:
    """Phase Analysis: 自分析诊断 — 输出 analysis.md 供下一轮参考。"""
    print_phase_header("Analysis", f"self-diagnosis (iteration {iteration})")

    prompt = build_analysis_prompt(
        iteration, workspace, code_path, training_log_path,
        score, evaluation_summary=evaluation_summary,
    )
    _log_phase_start(prompt, opencode_model)

    agent = _make_agent(workspace, opencode_model, stale_timeout, http_timeout)
    out = _run_agent(agent, prompt, iteration,
                     "analysis", f"Analysis iter{iteration}", "analysis")
    if isinstance(out, PhaseResult):
        return out

    # 读取 agent 写的 analysis.md
    analysis_path = Path(workspace).resolve() / "code" / "analysis.md"
    analysis_text = ""
    if analysis_path.exists():
        analysis_text = analysis_path.read_text(encoding="utf-8")
        console.print(f"  [green]Analysis written:[/] {analysis_path} [dim]({len(analysis_text)} chars)[/]")
        preview = analysis_text[:500].strip()
        if preview:
            console.print(f"  [dim]Preview:[/] {preview}...")
    else:
        console.print(f"  [bold yellow]WARNING:[/] analysis.md not found after agent finished")

    return PhaseResult(success=True, phase="analysis", payload={"analysis": analysis_text})
