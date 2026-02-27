"""Pipeline 主循环：状态机 + 断点续跑 + 两阶段分离验证。

阶段流程：
  Phase 0: VERIFY_GEN（仅第一轮，生成 verifier.py 并锁定）
  每轮迭代:
    CODE_GEN → TRAINING → EVAL_GENERATE → VERIFY → ANALYSIS → COMPLETE
"""

import json
import os
import time
from pathlib import Path

from .phases import (
    phase_analysis,
    phase_code_generation,
    phase_eval_generate,
    phase_fix_training,
    phase_training,
    phase_verifier_generation,
)
from .prompts import build_eval_repair_prompt
from .state import load_checkpoint, save_checkpoint
from .stream import make_stream_printer
from .types import (
    IterationResult,
    IterationState,
    Phase,
    PhaseResult,
    PipelineState,
)
from .ui import (
    console,
    print_data_gpu_info,
    print_iteration_header,
    print_iteration_summary,
    print_phase_header,
    print_phase_status,
    print_pipeline_footer,
    print_pipeline_header,
    print_verification_report,
)
from .utils import count_samples_jsonl, get_data_stats, get_gpu_info
from .verification import (
    check_verifier_integrity,
    compute_sha256,
    run_verification,
    validate_verifier,
)


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
_LOG_TRUNCATE = 20000
_ERR_TRUNCATE = 4000
_MAX_VERIFIER_RETRIES = 2
_MAX_PHASE0_OUTER_RETRIES = 50  # Phase 0 外层最大重试次数，防止永挂


# ---------------------------------------------------------------------------
# 共享工具函数
# ---------------------------------------------------------------------------
def _normalize_score(raw) -> float | None:
    """将 [0,1] 的原始分数归一化为 [0,100] 百分制。"""
    if raw is None or not isinstance(raw, (int, float)):
        return None
    return round(raw * 100, 2) if raw <= 1.05 else raw


def _find_trained_model(output_dir: str) -> str:
    """在 output_dir 下找到最新的模型子目录。"""
    out_path = Path(output_dir)
    if not out_path.exists():
        return str(out_path)
    try:
        subdirs = [d for d in out_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
    except OSError:
        return str(out_path)
    if subdirs:
        return str(max(subdirs, key=lambda d: d.stat().st_mtime))
    return str(out_path)


def _resolve_opencode_config(fsm_config: dict) -> tuple[str, str | None]:
    """统一解析 OpenCode model/url 配置。"""
    model = fsm_config.get("opencode_model", "") or os.environ.get("OPENCODE_MODEL", "")
    url = fsm_config.get("opencode_url", "") or os.environ.get("OPENCODE_URL", "") or None
    return model, url


def _run_opencode_agent(
    fsm_config: dict,
    prompt: str,
    *,
    max_turns: int = 30,
    session_title: str = "",
    log_name: str = "agent",
    fsm_state: str = "S0_EVAL_REPAIR",
    purpose: str = "eval_repair",
    label: str = "Agent",
    stale_timeout: float = 180.0,
    http_timeout: float = 300.0,
) -> bool:
    """通用 OpenCodeClient 执行器。返回 True 如果 agent 正常完成。"""
    try:
        from runner_fsm.opencode.client import OpenCodeClient
    except ImportError:
        console.print(f"  [dim]{label}: runner_fsm not available, skipping[/]")
        return False

    target_repo = fsm_config.get("target_repo", ".")
    repo_root = Path(target_repo).resolve()
    model, base_url = _resolve_opencode_config(fsm_config)

    out_dir = repo_root / "code" / "agent_logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"  [dim]Prompt:[/] {len(prompt)} chars  [dim]Starting server...[/]")
    try:
        agent = OpenCodeClient(
            repo=repo_root,
            plan_rel="PLAN.md",
            pipeline_rel=None,
            model=model,
            base_url=base_url,
            timeout_seconds=http_timeout,
            bash_mode="full",
            scaffold_bash_mode="full",
            unattended="strict",
            max_turns=max_turns,
            server_log_path=out_dir / f"{log_name}.log",
            session_title=session_title,
            stale_timeout=stale_timeout,
        )
    except Exception as e:
        console.print(f"  [bold red]{label}: failed to create OpenCodeClient:[/] {e}")
        return False

    try:
        result = agent.run(
            prompt,
            fsm_state=fsm_state,
            iter_idx=0,
            purpose=purpose,
            on_turn=make_stream_printer(label),
        )
        console.print(f"  [green]{label}: agent completed[/]")
        if result.assistant_text:
            (out_dir / f"{log_name}_result.txt").write_text(
                result.assistant_text[-_LOG_TRUNCATE:], encoding="utf-8",
            )
        return True
    except Exception as e:
        console.print(f"  [bold red]{label}: agent error:[/] {e}")
        (out_dir / f"{log_name}_error.txt").write_text(
            str(e)[:_ERR_TRUNCATE], encoding="utf-8",
        )
        return False
    finally:
        try:
            agent.close()
        except Exception:
            pass


def _repair_eval_zero_reward(
    workspace: str,
    fsm_config: dict,
    code_path: str,
    samples_path: str,
    task_description: str,
    data_path: str,
    pass_count: int,
    total_samples: int,
    repair_attempt: int,
    max_attempts: int,
    eval_log: str = "",
    stale_timeout: float = 180.0,
    http_timeout: float = 300.0,
) -> bool:
    """调用 OpenCode 自主诊断并修复 train.py 的 --eval-only 评测逻辑。"""
    print_phase_header("Eval Repair", f"attempt {repair_attempt}/{max_attempts}")

    prompt = build_eval_repair_prompt(
        code_path=code_path,
        samples_path=samples_path,
        data_path=data_path,
        task_description=task_description,
        pass_count=pass_count,
        total_samples=total_samples,
        repair_attempt=repair_attempt,
        max_attempts=max_attempts,
        eval_log=eval_log,
    )

    return _run_opencode_agent(
        fsm_config, prompt,
        max_turns=30,
        session_title=f"eval_repair_{repair_attempt}",
        log_name=f"opencode_eval_repair_{repair_attempt}",
        label="EvalRepair",
        stale_timeout=stale_timeout,
        http_timeout=http_timeout,
    )


# ---------------------------------------------------------------------------
# Phase 0: 验证器生成与锁定
# ---------------------------------------------------------------------------
def _run_phase_verify_gen(
    state: PipelineState,
    task_description: str,
    opencode_model: str,
    opencode_url: str,
    stale_timeout: float = 180.0,
    max_verifier_retries: int = _MAX_VERIFIER_RETRIES,
    http_timeout: float = 300.0,
) -> PhaseResult:
    """执行 Phase 0：生成 verifier.py，验证，锁定。"""
    workspace = state.workspace
    data_path = state.data_path
    verifier_path = str(Path(workspace) / "code" / "verifier.py")

    for attempt in range(max_verifier_retries + 1):
        if attempt > 0:
            console.print(f"\n  [yellow]Verifier retry {attempt}/{max_verifier_retries}[/]")

        # 生成 verifier.py
        gen_result = phase_verifier_generation(
            workspace=workspace,
            data_path=data_path,
            task_description=task_description,
            max_agent_steps=state.max_agent_steps,
            opencode_model=opencode_model,
            opencode_url=opencode_url,
            stale_timeout=stale_timeout,
            http_timeout=http_timeout,
        )
        if not gen_result.success:
            console.print(f"  [red]Verifier generation failed:[/] {gen_result.error}")
            if attempt < max_verifier_retries:
                continue
            return gen_result

        # 验证 verifier.py 的正确性
        ok, err = validate_verifier(verifier_path, data_path)
        if ok:
            # 锁定：计算 SHA256，备份内容
            sha256 = compute_sha256(verifier_path)
            backup = Path(verifier_path).read_text(encoding="utf-8")
            state.verifier_sha256 = sha256
            state.verifier_backup = backup

            console.print(f"  [green]Verifier validated and locked[/] [dim](SHA256: {sha256[:16]}...)[/]")
            return PhaseResult(
                success=True, phase="verify_gen",
                payload={
                    "verifier_path": verifier_path,
                    "sha256": sha256,
                },
            )
        else:
            console.print(f"  [red]Verifier validation failed:[/] {err}")
            if attempt < max_verifier_retries:
                continue
            return PhaseResult(
                success=False, phase="verify_gen",
                error=f"Verifier validation failed after {max_verifier_retries + 1} attempts: {err}",
            )

    return PhaseResult(success=False, phase="verify_gen", error="Exhausted retries")


# ---------------------------------------------------------------------------
# 迭代内各阶段
# ---------------------------------------------------------------------------
def _run_phase_code_gen(
    state: PipelineState,
    iter_state: IterationState,
    history: list[IterationResult],
    task_description: str,
    gpu_info: dict,
    opencode_model: str,
    opencode_url: str,
    stale_timeout: float = 180.0,
    http_timeout: float = 300.0,
) -> PhaseResult:
    """CODE_GEN 阶段。"""
    result = phase_code_generation(
        iteration=iter_state.iteration,
        workspace=state.workspace,
        base_model=state.base_model,
        task_description=task_description,
        history=history,
        max_agent_steps=state.max_agent_steps,
        gpu_info=gpu_info,
        opencode_model=opencode_model,
        opencode_url=opencode_url,
        stale_timeout=stale_timeout,
        http_timeout=http_timeout,
    )
    if result.success:
        iter_state.code_path = result.payload.get("code_path", "")
    else:
        iter_state.code_path = str(Path(state.workspace) / "code" / "train.py")
    return result


def _run_phase_training(
    state: PipelineState,
    iter_state: IterationState,
    opencode_model: str,
    opencode_url: str,
    stale_timeout: float = 180.0,
    http_timeout: float = 300.0,
) -> PhaseResult:
    """TRAINING 阶段（含重试循环）。"""
    code_path = iter_state.code_path
    workspace = state.workspace

    train_result = phase_training(
        workspace, code_path, timeout=state.training_timeout,
    )

    training_log_path = str(Path(workspace) / "code" / "training_stdout.log")
    stdout = train_result.payload.get("stdout", "")
    Path(training_log_path).write_text(stdout, encoding="utf-8")

    total_time = train_result.payload.get("elapsed", 0.0)

    for retry in range(state.max_fix_retries):
        if train_result.success:
            break
        console.print(f"\n  [yellow]--- Fix retry {retry + 1}/{state.max_fix_retries} ---[/]")

        fix_result = phase_fix_training(
            code_path, training_log_path, state.data_path, workspace,
            iteration=iter_state.iteration,
            opencode_model=opencode_model,
            opencode_url=opencode_url,
            max_agent_steps=state.max_agent_steps,
            stale_timeout=stale_timeout,
            http_timeout=http_timeout,
        )
        if not fix_result.success:
            console.print(f"  [red]Fix agent failed:[/] {fix_result.error}")
            break

        console.print(f"  [dim]Agent fix attempt done, re-running training...[/]")
        train_result = phase_training(
            workspace, code_path, timeout=state.training_timeout,
        )
        extra_time = train_result.payload.get("elapsed", 0.0)
        total_time += extra_time
        stdout = train_result.payload.get("stdout", "")
        Path(training_log_path).write_text(stdout, encoding="utf-8")

    exit_code = train_result.payload.get("exit_code", -1)
    iter_state.exit_code = exit_code
    iter_state.stdout = train_result.payload.get("stdout", "")
    iter_state.training_time = total_time

    return PhaseResult(
        success=train_result.success,
        phase="training",
        payload={
            "exit_code": exit_code,
            "elapsed": total_time,
        },
        error=train_result.error,
    )


def _run_phase_eval_generate(
    state: PipelineState,
    iter_state: IterationState,
    fsm_config: dict,
    task_description: str,
    stale_timeout: float = 180.0,
    http_timeout: float = 300.0,
) -> PhaseResult:
    """EVAL_GENERATE 阶段（含零分修复循环）。"""
    workspace = state.workspace
    code_path = iter_state.code_path
    output_dir = state.output_dir

    for eval_attempt in range(state.max_eval_repair_retries + 1):
        eval_result = phase_eval_generate(workspace, code_path)
        eval_stdout = eval_result.payload.get("stdout", "")

        # 保存评测日志
        eval_log_path = Path(workspace) / "code" / "eval_stdout.log"
        eval_log_path.write_text(eval_stdout, encoding="utf-8")

        if not eval_result.success:
            if eval_attempt < state.max_eval_repair_retries:
                console.print(f"\n  [yellow]Eval crashed, attempting repair...[/]")
                _repair_eval_zero_reward(
                    workspace=workspace,
                    fsm_config=fsm_config,
                    code_path=code_path,
                    samples_path="",
                    task_description=task_description,
                    data_path=state.data_path,
                    pass_count=0,
                    total_samples=0,
                    repair_attempt=eval_attempt + 1,
                    max_attempts=state.max_eval_repair_retries,
                    eval_log=eval_stdout,
                    stale_timeout=stale_timeout,
                    http_timeout=http_timeout,
                )
                continue
            return PhaseResult(
                success=False, phase="eval_generate",
                error="Eval crashed after all retries",
            )

        # 检查 samples.jsonl
        samples_path = str(Path(output_dir) / "samples.jsonl")
        if not Path(samples_path).exists():
            console.print(f"  [bold yellow]WARNING:[/] {samples_path} not found after eval")
            if eval_attempt < state.max_eval_repair_retries:
                _repair_eval_zero_reward(
                    workspace=workspace,
                    fsm_config=fsm_config,
                    code_path=code_path,
                    samples_path="",
                    task_description=task_description,
                    data_path=state.data_path,
                    pass_count=0,
                    total_samples=0,
                    repair_attempt=eval_attempt + 1,
                    max_attempts=state.max_eval_repair_retries,
                    eval_log=eval_stdout,
                    stale_timeout=stale_timeout,
                    http_timeout=http_timeout,
                )
                continue
            return PhaseResult(
                success=False, phase="eval_generate",
                error="samples.jsonl not found after all retries",
            )

        iter_state.samples_path = samples_path

        # 读取 Agent 自报分数（仅用于交叉验证）
        total, pass_count = count_samples_jsonl(samples_path)
        agent_score = pass_count / total if total > 0 else 0.0
        console.print(f"  [dim]\\[samples][/] Agent self-reported: {pass_count}/{total} = {agent_score:.4f}")

        if total == 0 and eval_attempt < state.max_eval_repair_retries:
            console.print(f"\n  [yellow]No samples generated, attempting repair...[/]")
            _repair_eval_zero_reward(
                workspace=workspace,
                fsm_config=fsm_config,
                code_path=code_path,
                samples_path=samples_path,
                task_description=task_description,
                data_path=state.data_path,
                pass_count=0,
                total_samples=0,
                repair_attempt=eval_attempt + 1,
                max_attempts=state.max_eval_repair_retries,
                eval_log=eval_stdout,
                stale_timeout=stale_timeout,
            )
            continue

        return PhaseResult(
            success=True, phase="eval_generate",
            payload={
                "samples_path": samples_path,
                "agent_pass_count": pass_count,
                "agent_total": total,
                "agent_score": agent_score,
            },
        )

    return PhaseResult(
        success=False, phase="eval_generate",
        error="Exhausted all eval repair retries",
    )


def _run_phase_verify(
    state: PipelineState,
    iter_state: IterationState,
) -> PhaseResult:
    """VERIFY 阶段：管线独立验证。不重试。"""
    verifier_path = str(Path(state.workspace) / "code" / "verifier.py")
    samples_path = iter_state.samples_path

    if not state.verifier_sha256:
        console.print(f"  [dim]\\[verify] No verifier locked, skipping independent verification[/]")
        # 回退到 Agent 自报分数
        total, pass_count = count_samples_jsonl(samples_path)
        score = pass_count / total if total > 0 else 0.0
        return PhaseResult(
            success=True, phase="verify",
            payload={"pipeline_score": score, "fallback": True},
        )

    # 检查 verifier 完整性
    integrity_ok = check_verifier_integrity(
        verifier_path, state.verifier_sha256, state.verifier_backup,
    )
    if not integrity_ok:
        console.print(f"  [bold yellow]\\[verify] Verifier integrity check failed, using agent scores[/]")
        total, pass_count = count_samples_jsonl(samples_path)
        score = pass_count / total if total > 0 else 0.0
        return PhaseResult(
            success=True, phase="verify",
            payload={"pipeline_score": score, "fallback": True},
        )

    # 运行独立验证
    vr = run_verification(verifier_path, samples_path, state.data_path)

    iter_state.verification = vr.to_dict()

    return PhaseResult(
        success=True, phase="verify",
        payload={
            "pipeline_score": vr.pipeline_score,
            "agent_score": vr.agent_score,
            "total": vr.total,
            "passed": vr.passed,
            "reward_agreement_rate": vr.reward_agreement_rate,
            "reward_inflation": vr.reward_inflation,
        },
    )


def _run_phase_analysis(
    state: PipelineState,
    iter_state: IterationState,
    opencode_model: str,
    opencode_url: str,
    stale_timeout: float = 180.0,
    http_timeout: float = 300.0,
) -> PhaseResult:
    """ANALYSIS 阶段。"""
    workspace = state.workspace
    training_log_path = str(Path(workspace) / "code" / "training_stdout.log")

    # 构建验证摘要供 analysis prompt 使用
    verification_summary = ""
    if iter_state.verification:
        v = iter_state.verification
        verification_summary = (
            f"总样本数: {v.get('total', 0)}\n"
            f"通过（管线验证）: {v.get('passed', 0)}/{v.get('total', 0)} = {v.get('pipeline_score', 0):.4f}\n"
            f"Agent 自报分数: {v.get('agent_score', 0):.4f}\n"
            f"Reward 一致率: {v.get('reward_agreement_rate', 0) * 100:.2f}%\n"
            f"Reward 注水量: {v.get('reward_inflation', 0):+.4f}"
        )

    result = phase_analysis(
        iteration=iter_state.iteration,
        workspace=workspace,
        code_path=iter_state.code_path,
        training_log_path=training_log_path,
        score=iter_state.score,
        samples_path=iter_state.samples_path,
        opencode_model=opencode_model,
        opencode_url=opencode_url,
        max_agent_steps=state.max_agent_steps,
        verification_summary=verification_summary,
        stale_timeout=stale_timeout,
        http_timeout=http_timeout,
    )

    if result.success:
        iter_state.analysis = result.payload.get("analysis", "")

    return result


# ---------------------------------------------------------------------------
# 主 Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    task: str,
    base_model: str,
    workspace: str,
    data_path: str = "",
    output_dir: str = "",
    max_iterations: int = 5,
    training_timeout: int = 3600,
    max_agent_steps: int = 25,
    max_fix_retries: int = 2,
    max_eval_repair_retries: int = 2,
    fsm_config: dict | None = None,
    resume: bool = False,
    stale_timeout: int = 180,
    max_verifier_retries: int = 2,
    http_timeout: int = 300,
):
    """运行 Pipeline（状态机 + 断点续跑 + 两阶段分离验证）

    阶段流程：
      Phase 0: 验证器生成（仅第一轮前执行一次）
      每轮迭代：
        CODE_GEN → TRAINING → EVAL_GENERATE → VERIFY → ANALYSIS → COMPLETE
    """
    pipeline_start = time.time()
    fsm_config = fsm_config or {}

    if not data_path:
        data_path = os.environ.get("DATA_PATH", "")
    if not output_dir:
        output_dir = os.environ.get("OUTPUT_DIR", str(Path(workspace) / "output"))

    opencode_model = os.environ.get("OPENCODE_MODEL", "")
    opencode_url = os.environ.get("OPENCODE_URL", "")

    # ----- 尝试恢复 checkpoint -----
    state: PipelineState | None = None
    if resume:
        state = load_checkpoint(workspace)
        if state:
            console.print(f"  [green]Resuming from checkpoint:[/] iteration {state.current_iteration}")
        else:
            console.print(f"  [dim]No checkpoint found, starting fresh[/]")

    if state is None:
        state = PipelineState(
            task=task,
            base_model=base_model,
            workspace=workspace,
            data_path=data_path,
            output_dir=output_dir,
            max_iterations=max_iterations,
            training_timeout=training_timeout,
            max_agent_steps=max_agent_steps,
            max_fix_retries=max_fix_retries,
            max_eval_repair_retries=max_eval_repair_retries,
            pipeline_start_time=pipeline_start,
        )

    print_pipeline_header(
        task=task, base_model=base_model, workspace=workspace,
        data_path=data_path, output_dir=output_dir,
        max_iterations=max_iterations, training_timeout=training_timeout,
        max_agent_steps=max_agent_steps, opencode_model=opencode_model,
        resume=resume,
    )

    task_description = ""
    for fname in ["description.md", "instructions.md"]:
        fpath = Path(workspace) / fname
        if fpath.exists():
            task_description += fpath.read_text() + "\n\n"

    if not task_description.strip():
        task_description = f"Benchmark: {task}\nTrain a language model using GRPO reinforcement learning.\n"
        console.print(f"  [bold yellow]WARNING:[/] No description.md found, using fallback: {task}")

    data_stats = get_data_stats(data_path)
    gpu_info = get_gpu_info()
    print_data_gpu_info(data_stats["count"], gpu_info["num_gpus"], gpu_info["gpu_name"])

    # ----- Phase 0: 验证器生成（仅执行一次，必须成功）-----
    if not state.verifier_sha256:
        print_phase_header("Phase 0: Verifier Generation", "generate & lock verifier.py")

        vg_attempt = 0
        while vg_attempt < _MAX_PHASE0_OUTER_RETRIES:
            vg_attempt += 1
            vg_result = _run_phase_verify_gen(
                state, task_description, opencode_model, opencode_url,
                stale_timeout=stale_timeout,
                max_verifier_retries=max_verifier_retries,
                http_timeout=http_timeout,
            )
            if vg_result.success:
                save_checkpoint(state)
                break
            else:
                console.print(f"  [bold yellow]Verifier generation failed (attempt {vg_attempt}/{_MAX_PHASE0_OUTER_RETRIES}):[/] {vg_result.error}")
                console.print(f"  [dim]Retrying... (verifier is required, will not skip)[/]")
        else:
            raise RuntimeError(
                f"Phase 0 failed after {_MAX_PHASE0_OUTER_RETRIES} outer retries. "
                f"Last error: {vg_result.error}"
            )
    else:
        print_phase_status(f"Verifier already locked (SHA256: {state.verifier_sha256[:16]}...)", "green")

    # ----- 构建历史（从 state.iterations 转换）-----
    history: list[IterationResult] = []
    for it in state.iterations:
        if it.current_phase == Phase.COMPLETE.value:
            history.append(IterationResult(
                iteration=it.iteration,
                exit_code=it.exit_code,
                training_time=it.training_time,
                score=it.score,
                agent_score=it.agent_score,
                model_path=it.model_path,
                code_path=it.code_path,
                samples_path=it.samples_path,
                analysis=it.analysis,
            ))

    # ----- 确定起始迭代 -----
    start_iteration = state.current_iteration + 1 if state.current_iteration > 0 else 1

    # 如果 resume，检查是否有未完成的迭代
    resume_phase: str | None = None
    if resume and state.iterations:
        last_iter = state.iterations[-1]
        if last_iter.current_phase != Phase.COMPLETE.value:
            # 从中断的阶段继续
            start_iteration = last_iter.iteration
            resume_phase = last_iter.current_phase
            console.print(f"  [green]Resuming iteration {start_iteration} from phase: {resume_phase}[/]")

    best_score = state.best_score
    best_iteration = state.best_iteration

    for iteration in range(start_iteration, max_iterations + 1):
        iter_start = time.time()
        elapsed_total = iter_start - pipeline_start

        print_iteration_header(iteration, max_iterations, elapsed_total)

        # 获取或创建 IterationState
        if resume_phase and state.iterations and state.iterations[-1].iteration == iteration:
            iter_state = state.iterations[-1]
        else:
            iter_state = IterationState(iteration=iteration)
            state.iterations.append(iter_state)

        state.current_iteration = iteration

        # 确定起始阶段
        phases_order = [
            Phase.CODE_GEN, Phase.TRAINING, Phase.EVAL_GENERATE,
            Phase.VERIFY, Phase.ANALYSIS, Phase.COMPLETE,
        ]

        if resume_phase:
            try:
                start_phase_idx = [p.value for p in phases_order].index(resume_phase)
            except ValueError:
                start_phase_idx = 0
            resume_phase = None  # 仅首轮有效
        else:
            start_phase_idx = 0

        # ---- 状态机循环 ----
        for phase_idx in range(start_phase_idx, len(phases_order)):
            phase = phases_order[phase_idx]
            iter_state.current_phase = phase.value

            if phase == Phase.CODE_GEN:
                result = _run_phase_code_gen(
                    state, iter_state, history, task_description,
                    gpu_info, opencode_model, opencode_url,
                    stale_timeout=stale_timeout,
                    http_timeout=http_timeout,
                )
                iter_state.phase_results["code_gen"] = result.to_dict()
                save_checkpoint(state)

                if not result.success:
                    console.print(f"  [red]Code generation failed:[/] {result.error}")
                    iter_state.current_phase = Phase.COMPLETE.value
                    save_checkpoint(state)
                    break

            elif phase == Phase.TRAINING:
                # 检查 verifier 完整性（被篡改则自动恢复）
                verifier_path = str(Path(workspace) / "code" / "verifier.py")
                if state.verifier_sha256:
                    integrity_ok = check_verifier_integrity(
                        verifier_path, state.verifier_sha256, state.verifier_backup,
                    )
                    if not integrity_ok:
                        console.print(f"  [bold yellow]WARNING:[/] verifier.py was tampered with and has been restored")

                result = _run_phase_training(
                    state, iter_state, opencode_model, opencode_url,
                    stale_timeout=stale_timeout,
                    http_timeout=http_timeout,
                )
                iter_state.phase_results["training"] = result.to_dict()
                save_checkpoint(state)

                if not result.success:
                    console.print(f"  [red]Training failed after all retries[/]")
                    iter_state.current_phase = Phase.COMPLETE.value
                    save_checkpoint(state)
                    break

            elif phase == Phase.EVAL_GENERATE:
                result = _run_phase_eval_generate(
                    state, iter_state, fsm_config, task_description,
                    stale_timeout=stale_timeout,
                    http_timeout=http_timeout,
                )
                iter_state.phase_results["eval_generate"] = result.to_dict()
                save_checkpoint(state)

                if not result.success:
                    console.print(f"  [red]Eval generation failed:[/] {result.error}")
                    # 跳过 VERIFY 和 ANALYSIS
                    iter_state.current_phase = Phase.COMPLETE.value
                    save_checkpoint(state)
                    break

                iter_state.agent_score = _normalize_score(
                    result.payload.get("agent_score")
                )

            elif phase == Phase.VERIFY:
                result = _run_phase_verify(state, iter_state)
                iter_state.phase_results["verify"] = result.to_dict()

                pipeline_score = result.payload.get("pipeline_score", 0.0)
                eval_score = _normalize_score(pipeline_score)

                if eval_score is not None:
                    passed = result.payload.get("passed", 0)
                    total = result.payload.get("total", 0)
                    agent_s = result.payload.get("agent_score", 0.0)
                    agreement = result.payload.get("reward_agreement_rate", 0.0)
                    inflation = result.payload.get("reward_inflation", 0.0)
                    is_fallback = bool(result.payload.get("fallback"))

                    print_verification_report(
                        pipeline_score=pipeline_score,
                        agent_score=agent_s,
                        passed=passed, total=total,
                        agreement_rate=agreement,
                        inflation=inflation,
                        fallback=is_fallback,
                    )

                    iter_state.score = eval_score
                    iter_state.model_path = _find_trained_model(output_dir)
                    if best_score is None or eval_score > best_score:
                        best_score = eval_score
                        best_iteration = iteration
                else:
                    print_phase_status("No score available", "dim")

                state.best_score = best_score
                state.best_iteration = best_iteration
                save_checkpoint(state)

            elif phase == Phase.ANALYSIS:
                if iteration < max_iterations:
                    result = _run_phase_analysis(
                        state, iter_state, opencode_model, opencode_url,
                        stale_timeout=stale_timeout,
                        http_timeout=http_timeout,
                    )
                    iter_state.phase_results["analysis"] = result.to_dict()
                    save_checkpoint(state)

            elif phase == Phase.COMPLETE:
                iter_state.current_phase = Phase.COMPLETE.value
                save_checkpoint(state)

        # ---- 迭代结束 ----
        # 追加到 history
        history.append(IterationResult(
            iteration=iter_state.iteration,
            exit_code=iter_state.exit_code,
            training_time=iter_state.training_time,
            score=iter_state.score,
            agent_score=iter_state.agent_score,
            model_path=iter_state.model_path,
            code_path=iter_state.code_path,
            samples_path=iter_state.samples_path,
            analysis=iter_state.analysis,
        ))

        iter_elapsed = time.time() - iter_start
        print_iteration_summary(
            iteration=iteration,
            score=iter_state.score,
            agent_score=iter_state.agent_score,
            best_score=best_score,
            best_iteration=best_iteration,
            elapsed=iter_elapsed,
        )

    # ----- Pipeline 完成 -----
    total_time = time.time() - pipeline_start
    print_pipeline_footer(best_score, best_iteration,
                          len(state.iterations), total_time)

    summary = {
        "task": task,
        "base_model": base_model,
        "best_score": best_score,
        "best_iteration": best_iteration,
        "total_time": total_time,
        "verifier_sha256": state.verifier_sha256 or None,
        "iterations": [
            {
                "iteration": it.iteration,
                "exit_code": it.exit_code,
                "training_time": it.training_time,
                "score": it.score,
                "agent_score": it.agent_score,
                "samples_path": it.samples_path,
                "verification": it.verification,
            }
            for it in state.iterations
        ],
    }
    summary_path = Path(workspace) / "pipeline_results.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    console.print(f"  [dim]Results saved:[/] {summary_path}")
