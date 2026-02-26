"""Pipeline 各阶段执行逻辑。"""

import os
import subprocess
import sys
import time
from pathlib import Path

import requests

from runner_fsm.opencode.client import OpenCodeClient

from .prompts import build_analysis_prompt, build_code_prompt, build_fix_prompt
from .stream import make_stream_printer
from .types import IterationResult


def phase_code_generation(
    iteration: int,
    workspace: str,
    base_model: str,
    task_description: str,
    history: list[IterationResult],
    max_agent_steps: int = 25,
    gpu_info: dict | None = None,
    opencode_model: str = "",
    opencode_url: str = "",
) -> str:
    """Phase 1: 代码生成

    用 OpenCodeClient 探索数据并编写训练代码。
    返回代码文件路径。
    """
    print(f"\n{'='*60}")
    print(f"  Phase 1: Code Generation (iteration {iteration})")
    print(f"{'='*60}")

    model = opencode_model or os.environ.get("OPENCODE_MODEL", "")
    # opencode_url 是 OpenCode server 的地址（非 LLM API），为空则自动启动本地 server
    server_url = opencode_url or os.environ.get("OPENCODE_URL", "") or None

    repo_root = Path(workspace).resolve()
    log_dir = repo_root / "code" / "agent_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    prompt = build_code_prompt(
        iteration, workspace, base_model, task_description, history,
        gpu_info=gpu_info,
    )
    print(f"  Prompt: {len(prompt)} chars, model: {model}, starting server...", flush=True)

    agent = OpenCodeClient(
        repo=repo_root,
        plan_rel="PLAN.md",
        pipeline_rel=None,
        model=model,
        base_url=server_url,
        timeout_seconds=900,
        bash_mode="full",
        scaffold_bash_mode="full",
        unattended="strict",
        server_log_path=log_dir / f"opencode_codegen_iter{iteration}.log",
        session_title=f"code_generation_iter{iteration}",
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
    finally:
        try:
            agent.close()
        except Exception:
            pass

    code_path = Path(workspace) / "code" / "train.py"
    if code_path.exists():
        print(f"  Code generated: {code_path} ({code_path.stat().st_size} bytes)")
    else:
        print(f"  WARNING: {code_path} not found after agent finished")

    return str(code_path)


def phase_training(
    workspace: str,
    code_path: str,
    timeout: int = 3600,
) -> tuple[int, str, float]:
    """Phase 2: 训练执行（pipeline 控制，非 agent）

    用 accelerate launch 执行 train.py，自动多卡 DDP。
    """
    print(f"\n{'='*60}")
    print(f"  Phase 2: Training Execution")
    print(f"{'='*60}")

    if not Path(code_path).exists():
        msg = f"Code file not found: {code_path}"
        print(f"  ERROR: {msg}")
        return -1, msg, 0.0

    abs_code_path = str(Path(code_path).resolve())
    start = time.time()

    cmd = ["accelerate", "launch", abs_code_path]
    print(f"  CMD: {' '.join(cmd)}", flush=True)

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
                print(f"  TIMEOUT after {timeout}s", flush=True)
                break
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                line = line.rstrip("\n")
                collected.append(line)
                # 实时输出带前缀，方便 tail -f 识别
                print(f"  [train] {line}", flush=True)
        exit_code = proc.returncode if proc.returncode is not None else -1
    except Exception as e:
        collected.append(f"Exception: {e}")
        print(f"  Exception: {e}", flush=True)

    stdout = "\n".join(collected)
    elapsed = time.time() - start
    print(f"  Exit code: {exit_code}", flush=True)
    print(f"  Time: {elapsed:.1f}s", flush=True)

    if exit_code != 0:
        tail = "\n".join(stdout.strip().splitlines()[-15:])
        print(f"  Error tail:\n{tail}", flush=True)

    return exit_code, stdout, elapsed


def phase_evaluation(
    workspace: str,
    grading_url: str,
    pre_computed_score: float | None = None,
) -> dict | None:
    """Phase 3: 评测提交（pipeline 控制，非 agent）

    找到 $OUTPUT_DIR 下最新模型，POST 到 Grading Server。
    """
    print(f"\n{'='*60}")
    print(f"  Phase 3: Evaluation")
    print(f"{'='*60}")

    output_dir = Path(os.environ.get("OUTPUT_DIR", str(Path(workspace) / "output")))
    if not output_dir.exists() or not any(output_dir.iterdir()):
        print("  No model output, skipping evaluation")
        return None

    subdirs = [d for d in output_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if subdirs:
        model_path = str(max(subdirs, key=lambda d: d.stat().st_mtime))
    else:
        model_path = str(output_dir)

    print(f"  Submitting: {model_path}")

    max_retries = 2
    for attempt in range(1, max_retries + 1):
        try:
            json_body = {"model_path": model_path}
            if pre_computed_score is not None:
                json_body["score"] = pre_computed_score
            resp = requests.post(
                f"{grading_url}/submit",
                json=json_body,
                timeout=1200,
            )
            result = resp.json()
            print(f"  Score: {result.get('score')}")
            print(f"  Improvement: {result.get('improvement')}")
            print(f"  Best: {result.get('best', {}).get('score')}")
            return result
        except Exception as e:
            print(f"  Evaluation attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                print(f"  Retrying...")
                time.sleep(5)
    return None


def phase_fix_training(
    code_path: str,
    error_log_path: str,
    data_path: str,
    workspace: str,
    opencode_model: str = "",
    opencode_url: str = "",
) -> None:
    """训练失败后的修复尝试，使用 OpenCodeClient。"""
    model = opencode_model or os.environ.get("OPENCODE_MODEL", "")
    server_url = opencode_url or os.environ.get("OPENCODE_URL", "") or None

    repo_root = Path(workspace).resolve()
    log_dir = repo_root / "code" / "agent_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    fix_prompt = build_fix_prompt(code_path, error_log_path, data_path, workspace)
    print(f"  Prompt: {len(fix_prompt)} chars, model: {model}, starting server...", flush=True)

    agent = OpenCodeClient(
        repo=repo_root,
        plan_rel="PLAN.md",
        pipeline_rel=None,
        model=model,
        base_url=server_url,
        timeout_seconds=600,
        bash_mode="full",
        scaffold_bash_mode="full",
        unattended="strict",
        server_log_path=log_dir / "opencode_fix.log",
        session_title="fix_training",
    )

    try:
        result = agent.run(
            fix_prompt,
            fsm_state="S0_FIX",
            iter_idx=0,
            purpose="fix_training",
            on_turn=make_stream_printer("FixTraining"),
        )
        if result.assistant_text:
            (log_dir / "fix_result.txt").write_text(
                result.assistant_text[-20000:], encoding="utf-8",
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
    task_description: str = "",
    opencode_model: str = "",
    opencode_url: str = "",
) -> str:
    """Phase 4: 自分析诊断

    让 OpenCode 阅读训练日志、rollout 样本、当前代码，
    输出 analysis.md 诊断报告，供下一轮代码生成参考。
    返回诊断报告文本。
    """
    print(f"\n{'='*60}")
    print(f"  Phase Analysis: Self-diagnosis (iteration {iteration})")
    print(f"{'='*60}")

    model = opencode_model or os.environ.get("OPENCODE_MODEL", "")
    server_url = opencode_url or os.environ.get("OPENCODE_URL", "") or None

    repo_root = Path(workspace).resolve()
    log_dir = repo_root / "code" / "agent_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    prompt = build_analysis_prompt(
        iteration, workspace, code_path, training_log_path,
        score, samples_path, task_description,
    )
    print(f"  Prompt: {len(prompt)} chars, model: {model}, starting server...", flush=True)

    agent = OpenCodeClient(
        repo=repo_root,
        plan_rel="PLAN.md",
        pipeline_rel=None,
        model=model,
        base_url=server_url,
        timeout_seconds=600,
        bash_mode="full",
        scaffold_bash_mode="full",
        unattended="strict",
        server_log_path=log_dir / f"opencode_analysis_iter{iteration}.log",
        session_title=f"analysis_iter{iteration}",
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
    finally:
        try:
            agent.close()
        except Exception:
            pass

    # 读取 agent 写的 analysis.md
    analysis_path = repo_root / "code" / "analysis.md"
    if analysis_path.exists():
        analysis_text = analysis_path.read_text(encoding="utf-8")
        print(f"  Analysis written: {analysis_path} ({len(analysis_text)} chars)")
        # 打印摘要（前 500 字）
        preview = analysis_text[:500].strip()
        if preview:
            print(f"  Preview: {preview}...")
    else:
        print(f"  WARNING: analysis.md not found after agent finished")

    return analysis_text
