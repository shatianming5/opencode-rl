"""Pipeline 各阶段执行逻辑。

所有阶段函数统一返回 PhaseResult。
"""

import os
import subprocess
import sys
import time
from pathlib import Path

from runner_fsm.opencode.client import OpenCodeClient

from .prompts import (
    build_analysis_prompt,
    build_code_prompt,
    build_fix_prompt,
    build_verifier_prompt,
)
from .stream import make_stream_printer
from .types import IterationResult, PhaseResult
from .ui import console, print_phase_header, print_phase_status


def phase_verifier_generation(
    workspace: str,
    data_path: str,
    task_description: str,
    max_agent_steps: int = 30,
    opencode_model: str = "",
    opencode_url: str = "",
    stale_timeout: float = 180.0,
    http_timeout: float = 300.0,
) -> PhaseResult:
    """Phase 0: 验证器生成

    让 Agent 分析 benchmark 数据格式，编写 verifier.py。
    仅在第一轮迭代前执行一次。
    """
    model = opencode_model or os.environ.get("OPENCODE_MODEL", "")
    server_url = opencode_url or os.environ.get("OPENCODE_URL", "") or None

    repo_root = Path(workspace).resolve()
    log_dir = repo_root / "code" / "agent_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    prompt = build_verifier_prompt(workspace, data_path, task_description)
    console.print(f"  [dim]Prompt:[/] {len(prompt)} chars  [dim]Model:[/] {model}  [dim]Starting server...[/]")

    agent = OpenCodeClient(
        repo=repo_root,
        plan_rel="PLAN.md",
        pipeline_rel=None,
        model=model,
        base_url=server_url,
        timeout_seconds=http_timeout,
        bash_mode="full",
        scaffold_bash_mode="full",
        unattended="strict",
        max_turns=max_agent_steps,
        server_log_path=log_dir / "opencode_verifier_gen.log",
        session_title="verifier_generation",
        stale_timeout=stale_timeout,
    )

    try:
        result = agent.run(
            prompt,
            fsm_state="S0_VERIFY_GEN",
            iter_idx=0,
            purpose="verifier_generation",
            on_turn=make_stream_printer("VerifierGen"),
        )
        if result.assistant_text:
            (log_dir / "verifier_gen_result.txt").write_text(
                result.assistant_text[-20000:], encoding="utf-8",
            )
    except Exception as e:
        return PhaseResult(
            success=False, phase="verify_gen",
            error=f"Agent error: {e}",
        )
    finally:
        try:
            agent.close()
        except Exception:
            pass

    verifier_path = repo_root / "code" / "verifier.py"
    if verifier_path.exists():
        console.print(f"  [green]Verifier generated:[/] {verifier_path} [dim]({verifier_path.stat().st_size} bytes)[/]")
        return PhaseResult(
            success=True, phase="verify_gen",
            payload={"verifier_path": str(verifier_path)},
        )
    else:
        return PhaseResult(
            success=False, phase="verify_gen",
            error=f"verifier.py not found after agent finished",
        )


def phase_code_generation(
    iteration: int,
    workspace: str,
    base_model: str,
    task_description: str,
    history: list[IterationResult],
    max_agent_steps: int = 30,
    gpu_info: dict | None = None,
    opencode_model: str = "",
    opencode_url: str = "",
    stale_timeout: float = 180.0,
    http_timeout: float = 300.0,
) -> PhaseResult:
    """Phase 1: 代码生成

    用 OpenCodeClient 探索数据并编写训练代码。
    """
    print_phase_header("Code Generation", f"iteration {iteration}")

    model = opencode_model or os.environ.get("OPENCODE_MODEL", "")
    server_url = opencode_url or os.environ.get("OPENCODE_URL", "") or None

    repo_root = Path(workspace).resolve()
    log_dir = repo_root / "code" / "agent_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    prompt = build_code_prompt(
        iteration, workspace, base_model, task_description, history,
        gpu_info=gpu_info,
    )
    console.print(f"  [dim]Prompt:[/] {len(prompt)} chars  [dim]Model:[/] {model}  [dim]Starting server...[/]")

    agent = OpenCodeClient(
        repo=repo_root,
        plan_rel="PLAN.md",
        pipeline_rel=None,
        model=model,
        base_url=server_url,
        timeout_seconds=http_timeout,
        bash_mode="full",
        scaffold_bash_mode="full",
        unattended="strict",
        max_turns=max_agent_steps,
        server_log_path=log_dir / f"opencode_codegen_iter{iteration}.log",
        session_title=f"code_generation_iter{iteration}",
        stale_timeout=stale_timeout,
    )

    try:
        result = agent.run(
            prompt,
            fsm_state="S0_CODE_GEN",
            iter_idx=iteration,
            purpose="code_generation",
            on_turn=make_stream_printer(f"CodeGen iter{iteration}"),
        )
        if result.assistant_text:
            (log_dir / f"codegen_iter{iteration}_result.txt").write_text(
                result.assistant_text[-20000:], encoding="utf-8",
            )
    except Exception as e:
        return PhaseResult(
            success=False, phase="code_gen",
            error=f"Agent error: {e}",
        )
    finally:
        try:
            agent.close()
        except Exception:
            pass

    code_path = Path(workspace) / "code" / "train.py"
    if code_path.exists():
        console.print(f"  [green]Code generated:[/] {code_path} [dim]({code_path.stat().st_size} bytes)[/]")
        return PhaseResult(
            success=True, phase="code_gen",
            payload={"code_path": str(code_path)},
        )
    else:
        console.print(f"  [bold yellow]WARNING:[/] {code_path} not found after agent finished")
        return PhaseResult(
            success=False, phase="code_gen",
            error=f"{code_path} not found",
            payload={"code_path": str(code_path)},
        )


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
        )
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
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
                collected.append(line)
                console.print(f"  [dim]\\[train][/] {line}")
        exit_code = proc.returncode if proc.returncode is not None else -1
    except Exception as e:
        collected.append(f"Exception: {e}")
        console.print(f"  [bold red]Exception:[/] {e}")

    stdout = "\n".join(collected)
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


def phase_eval_generate(
    workspace: str,
    code_path: str,
    timeout: int = 1800,
) -> PhaseResult:
    """Phase Eval Generate: 运行 --eval-only 仅生成 completion。

    用 python（不用 accelerate launch），评测不需要 DDP。
    """
    print_phase_header("Eval Generate", "--eval-only (generate completions)")

    if not Path(code_path).exists():
        msg = f"Code file not found: {code_path}"
        console.print(f"  [bold red]ERROR:[/] {msg}")
        return PhaseResult(success=False, phase="eval_generate", error=msg)

    abs_code_path = str(Path(code_path).resolve())
    start = time.time()

    cmd = [sys.executable, abs_code_path, "--eval-only"]
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
        )
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
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
                collected.append(line)
                console.print(f"  [dim]\\[eval][/] {line}")
        exit_code = proc.returncode if proc.returncode is not None else -1
    except Exception as e:
        collected.append(f"Exception: {e}")
        console.print(f"  [bold red]Exception:[/] {e}")

    stdout = "\n".join(collected)
    elapsed = time.time() - start
    ec_style = "green" if exit_code == 0 else "red"
    console.print(f"  [dim]Exit code:[/] [{ec_style}]{exit_code}[/]  [dim]Time:[/] {elapsed:.1f}s")

    if exit_code != 0:
        tail = "\n".join(stdout.strip().splitlines()[-15:])
        console.print(f"  [dim]Error tail:[/]\n{tail}")

    return PhaseResult(
        success=(exit_code == 0),
        phase="eval_generate",
        payload={
            "exit_code": exit_code,
            "stdout": stdout,
            "elapsed": elapsed,
        },
        error="" if exit_code == 0 else f"Eval failed with exit code {exit_code}",
    )


def phase_fix_training(
    code_path: str,
    error_log_path: str,
    data_path: str,
    workspace: str,
    iteration: int = 0,
    opencode_model: str = "",
    opencode_url: str = "",
    max_agent_steps: int = 30,
    stale_timeout: float = 180.0,
    http_timeout: float = 300.0,
) -> PhaseResult:
    """训练失败后的修复尝试，使用 OpenCodeClient。"""
    model = opencode_model or os.environ.get("OPENCODE_MODEL", "")
    server_url = opencode_url or os.environ.get("OPENCODE_URL", "") or None

    repo_root = Path(workspace).resolve()
    log_dir = repo_root / "code" / "agent_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    fix_prompt = build_fix_prompt(code_path, error_log_path, data_path)
    console.print(f"  [dim]Prompt:[/] {len(fix_prompt)} chars  [dim]Model:[/] {model}  [dim]Starting server...[/]")

    agent = OpenCodeClient(
        repo=repo_root,
        plan_rel="PLAN.md",
        pipeline_rel=None,
        model=model,
        base_url=server_url,
        timeout_seconds=http_timeout,
        bash_mode="full",
        scaffold_bash_mode="full",
        unattended="strict",
        max_turns=max_agent_steps,
        server_log_path=log_dir / f"opencode_fix_iter{iteration}.log",
        session_title=f"fix_training_iter{iteration}",
        stale_timeout=stale_timeout,
    )

    try:
        result = agent.run(
            fix_prompt,
            fsm_state="S0_FIX",
            iter_idx=iteration,
            purpose="fix_training",
            on_turn=make_stream_printer("FixTraining"),
        )
        if result.assistant_text:
            (log_dir / f"fix_iter{iteration}_result.txt").write_text(
                result.assistant_text[-20000:], encoding="utf-8",
            )
        return PhaseResult(success=True, phase="fix_training")
    except Exception as e:
        return PhaseResult(
            success=False, phase="fix_training",
            error=f"Fix agent error: {e}",
        )
    finally:
        try:
            agent.close()
        except Exception:
            pass


def phase_analysis(
    iteration: int,
    workspace: str,
    code_path: str,
    training_log_path: str,
    score: float | None = None,
    samples_path: str = "",
    opencode_model: str = "",
    opencode_url: str = "",
    max_agent_steps: int = 30,
    verification_summary: str = "",
    stale_timeout: float = 180.0,
    http_timeout: float = 300.0,
) -> PhaseResult:
    """Phase Analysis: 自分析诊断

    让 OpenCode 阅读训练日志、rollout 样本、当前代码，
    输出 analysis.md 诊断报告，供下一轮代码生成参考。
    """
    print_phase_header("Analysis", f"self-diagnosis (iteration {iteration})")

    model = opencode_model or os.environ.get("OPENCODE_MODEL", "")
    server_url = opencode_url or os.environ.get("OPENCODE_URL", "") or None

    repo_root = Path(workspace).resolve()
    log_dir = repo_root / "code" / "agent_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    prompt = build_analysis_prompt(
        iteration, workspace, code_path, training_log_path,
        score, samples_path,
        verification_summary=verification_summary,
    )
    console.print(f"  [dim]Prompt:[/] {len(prompt)} chars  [dim]Model:[/] {model}  [dim]Starting server...[/]")

    agent = OpenCodeClient(
        repo=repo_root,
        plan_rel="PLAN.md",
        pipeline_rel=None,
        model=model,
        base_url=server_url,
        timeout_seconds=http_timeout,
        bash_mode="full",
        scaffold_bash_mode="full",
        unattended="strict",
        max_turns=max_agent_steps,
        server_log_path=log_dir / f"opencode_analysis_iter{iteration}.log",
        session_title=f"analysis_iter{iteration}",
        stale_timeout=stale_timeout,
    )

    analysis_text = ""
    try:
        result = agent.run(
            prompt,
            fsm_state="S0_ANALYSIS",
            iter_idx=iteration,
            purpose="analysis",
            on_turn=make_stream_printer(f"Analysis iter{iteration}"),
        )
        if result.assistant_text:
            (log_dir / f"analysis_iter{iteration}_result.txt").write_text(
                result.assistant_text[-20000:], encoding="utf-8",
            )
    except Exception as e:
        return PhaseResult(
            success=False, phase="analysis",
            error=f"Analysis agent error: {e}",
        )
    finally:
        try:
            agent.close()
        except Exception:
            pass

    # 读取 agent 写的 analysis.md
    analysis_path = repo_root / "code" / "analysis.md"
    if analysis_path.exists():
        analysis_text = analysis_path.read_text(encoding="utf-8")
        console.print(f"  [green]Analysis written:[/] {analysis_path} [dim]({len(analysis_text)} chars)[/]")
        preview = analysis_text[:500].strip()
        if preview:
            console.print(f"  [dim]Preview:[/] {preview}...")
    else:
        console.print(f"  [bold yellow]WARNING:[/] analysis.md not found after agent finished")

    return PhaseResult(
        success=True, phase="analysis",
        payload={"analysis": analysis_text},
    )


# ---------------------------------------------------------------------------
# 向后兼容：保留旧的 phase_eval_only 函数签名
# ---------------------------------------------------------------------------
def phase_eval_only(
    workspace: str,
    code_path: str,
    timeout: int = 1800,
) -> tuple[int, str, float]:
    """向后兼容：运行 --eval-only，返回 (exit_code, stdout, elapsed)。"""
    result = phase_eval_generate(workspace, code_path, timeout)
    return (
        result.payload.get("exit_code", -1),
        result.payload.get("stdout", ""),
        result.payload.get("elapsed", 0.0),
    )
