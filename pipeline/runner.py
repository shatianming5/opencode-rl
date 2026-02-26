"""Pipeline 主循环：编排 code_gen → training → FSM deploy/rollout/evaluate → 迭代。"""

import json
import os
import threading
import time
from pathlib import Path

from .phases import (
    phase_analysis,
    phase_code_generation,
    phase_fix_training,
    phase_training,
)
from .types import IterationResult
from .prompts import build_scaffold_prompt, build_rollout_repair_prompt
from .stream import make_stream_printer
from .utils import count_samples_jsonl, get_data_stats, get_gpu_info


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
_LOG_TRUNCATE = 20000
_ERR_TRUNCATE = 4000


# ---------------------------------------------------------------------------
# 共享工具函数
# ---------------------------------------------------------------------------
def _normalize_score(raw) -> float | None:
    """将 [0,1] 的原始分数归一化为 [0,100] 百分制。"""
    if raw is None or not isinstance(raw, (int, float)):
        return None
    # 阈值 1.05：容忍浮点误差（如 1.001），但已是百分制的（如 50.0）不再缩放
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
    fsm_state: str = "S0_SCAFFOLD",
    purpose: str = "scaffold_contract",
    label: str = "Agent",
) -> bool:
    """通用 OpenCodeClient 执行器。返回 True 如果 agent 正常完成。"""
    try:
        from runner_fsm.opencode.client import OpenCodeClient
    except ImportError:
        print(f"  {label}: runner_fsm not available, skipping")
        return False

    target_repo = fsm_config.get("target_repo", ".")
    repo_root = Path(target_repo).resolve()
    model, base_url = _resolve_opencode_config(fsm_config)

    out_dir = repo_root / ".opencode_fsm" / "scaffold_logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Prompt: {len(prompt)} chars, starting OpenCode server...", flush=True)
    try:
        agent = OpenCodeClient(
            repo=repo_root,
            plan_rel="PLAN.md",
            pipeline_rel="pipeline.yml",
            model=model,
            base_url=base_url,
            timeout_seconds=600,
            bash_mode="restricted",
            scaffold_bash_mode="full",
            unattended="strict",
            max_turns=max_turns,
            server_log_path=out_dir / f"{log_name}.log",
            session_title=session_title,
        )
    except Exception as e:
        print(f"  {label}: failed to create OpenCodeClient: {e}")
        return False

    try:
        result = agent.run(
            prompt,
            fsm_state=fsm_state,
            iter_idx=0,
            purpose=purpose,
            on_turn=make_stream_printer(label),
        )
        print(f"  {label}: agent completed")
        if result.assistant_text:
            (out_dir / f"{log_name}_result.txt").write_text(
                result.assistant_text[-_LOG_TRUNCATE:], encoding="utf-8",
            )
        return True
    except Exception as e:
        print(f"  {label}: agent error: {e}")
        (out_dir / f"{log_name}_error.txt").write_text(
            str(e)[:_ERR_TRUNCATE], encoding="utf-8",
        )
        return False
    finally:
        try:
            agent.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# FSM 进度监控
# ---------------------------------------------------------------------------
class _FSMProgressMonitor:
    """后台线程，定时扫描 FSM artifacts 目录显示进度。"""

    def __init__(self, label: str, target_repo: str, total_samples: int = 0,
                 interval: float = 15.0, is_evaluate: bool = False):
        self._label = label
        self._target_repo = Path(target_repo).resolve()
        self._total = total_samples
        self._interval = interval
        self._is_evaluate = is_evaluate
        self._stop = threading.Event()
        self._start_time = time.time()
        self._thread: threading.Thread | None = None
        self._last_metrics_mtime: float = 0
        self._last_log_size: int = 0

    def _find_samples_jsonl(self) -> Path | None:
        candidates = [
            self._target_repo / ".opencode_fsm" / "artifacts" / "samples.jsonl",
            self._target_repo / ".opencode_fsm" / "samples.jsonl",
        ]
        rollout_json = self._target_repo / ".opencode_fsm" / "rollout.json"
        if rollout_json.exists():
            try:
                obj = json.loads(rollout_json.read_text())
                p = (obj.get("paths") or {}).get("samples_jsonl") or obj.get("samples_jsonl")
                if p:
                    candidates.insert(0, Path(p))
            except Exception:
                pass
        for c in candidates:
            if c.exists():
                return c
        return None

    def _read_metrics(self) -> dict | None:
        metrics_path = self._target_repo / ".opencode_fsm" / "metrics.json"
        if not metrics_path.exists():
            return None
        try:
            mtime = metrics_path.stat().st_mtime
            if mtime > self._last_metrics_mtime:
                self._last_metrics_mtime = mtime
                return json.loads(metrics_path.read_text())
        except Exception:
            pass
        return None

    def _tail_eval_log(self, max_lines: int = 3) -> str | None:
        log_candidates = [
            self._target_repo / ".opencode_fsm" / "eval.log",
            self._target_repo / ".opencode_fsm" / "evaluation.log",
        ]
        artifacts = self._target_repo / ".opencode_fsm" / "artifacts"
        if artifacts.exists():
            for log_file in sorted(artifacts.rglob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True):
                log_candidates.insert(0, log_file)
                break
        for log_path in log_candidates:
            if not log_path.exists():
                continue
            try:
                size = log_path.stat().st_size
                if size <= self._last_log_size:
                    continue
                self._last_log_size = size
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                tail = [l.rstrip() for l in lines[-max_lines:] if l.strip()]
                if tail:
                    return " | ".join(tail)
            except Exception:
                pass
        return None

    def _tick(self):
        elapsed = time.time() - self._start_time

        if self._is_evaluate:
            metrics = self._read_metrics()
            if metrics:
                score = metrics.get("accuracy", metrics.get("score", "?"))
                passed = metrics.get("pass_count", "?")
                total = metrics.get("total", "?")
                print(f"  [{self._label}] {elapsed:.0f}s | score={score}, passed={passed}/{total}", flush=True)
                return
            log_tail = self._tail_eval_log()
            if log_tail:
                if len(log_tail) > 120:
                    log_tail = log_tail[:117] + "..."
                print(f"  [{self._label}] {elapsed:.0f}s | {log_tail}", flush=True)
                return

        samples_path = self._find_samples_jsonl()
        if samples_path:
            done, passed = count_samples_jsonl(str(samples_path))
            total_str = f"/{self._total}" if self._total else ""
            pct = f" ({done/self._total*100:.0f}%)" if self._total and done else ""
            print(f"  [{self._label}] {elapsed:.0f}s | {done}{total_str} samples{pct}, {passed} passed", flush=True)
        else:
            artifacts = self._target_repo / ".opencode_fsm" / "artifacts"
            runtime = self._target_repo / ".opencode_fsm" / "runtime_env.json"
            if runtime.exists():
                print(f"  [{self._label}] {elapsed:.0f}s | model deployed, generating...", flush=True)
            elif artifacts.exists():
                print(f"  [{self._label}] {elapsed:.0f}s | preparing...", flush=True)
            else:
                print(f"  [{self._label}] {elapsed:.0f}s | waiting...", flush=True)

    def _run(self):
        while not self._stop.wait(self._interval):
            self._tick()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._tick()

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()


# ---------------------------------------------------------------------------
# FSM 核心函数
# ---------------------------------------------------------------------------
def _create_fsm_session(fsm_config: dict):
    """创建 FSM EnvSession。"""
    try:
        from runner_fsm.env import setup
    except ImportError:
        from runner.env import setup

    session = setup(
        target=fsm_config.get("target_repo", "."),
        require_metrics=True,
        audit="warn-only",
        use_cache=fsm_config.get("cache_enabled", True),
        opencode_model=fsm_config.get("opencode_model", ""),
        opencode_url=fsm_config.get("opencode_url", ""),
        unattended="strict",
        strict_opencode=True,
        opencode_timeout_seconds=600,
    )
    session.command_hints = []
    session.hint_anchors = []
    return session


def _build_fsm_env_overrides(
    model_path: str,
    deploy_engine: str,
    data_path: str = "",
    output_dir: str = "",
) -> dict[str, str]:
    """构建传递给 FSM stage 脚本的环境变量。"""
    overrides: dict[str, str] = {"TRAINED_MODEL_PATH": str(model_path), "DEPLOY_ENGINE": deploy_engine}
    if data_path:
        overrides["DATA_PATH"] = data_path
    if output_dir:
        overrides["OUTPUT_DIR"] = output_dir
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE",
                 "MODEL_PATH", "CUDA_VISIBLE_DEVICES"):
        val = os.environ.get(key, "")
        if val:
            overrides[key] = val
    for key in ("DATA_PATH", "OUTPUT_DIR"):
        if key not in overrides:
            val = os.environ.get(key, "")
            if val:
                overrides[key] = val
    return overrides


def _extract_samples_path(rollout_path, target_repo: str) -> str | None:
    """从 rollout.json 中提取 samples.jsonl 路径。"""
    if rollout_path is None:
        return None
    try:
        with open(rollout_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    paths = obj.get("paths")
    if isinstance(paths, dict):
        raw = paths.get("samples_jsonl")
        if isinstance(raw, str) and raw.strip():
            p = Path(raw.strip())
            if p.exists():
                return str(p)
            repo = Path(target_repo).resolve()
            candidate = (repo / p).resolve()
            if candidate.exists():
                return str(candidate)
    return None


def _compute_score_from_samples(samples_path: str) -> dict | None:
    """直接从 rollout 的 samples.jsonl 计算 pass@1 score。"""
    total, pass_count = count_samples_jsonl(samples_path)
    if total == 0:
        return None
    accuracy = pass_count / total
    print(f"  [samples fallback] {pass_count}/{total} = {accuracy:.4f}")
    return {
        "ok": True,
        "score": accuracy,
        "pass_count": pass_count,
        "total": total,
        "source": "samples_direct",
    }


def _read_fsm_metrics(target_repo: str, max_age_seconds: int = 600) -> dict | None:
    """Fallback: 直接读 FSM 产出的 metrics.json。"""
    metrics_path = Path(target_repo) / ".opencode_fsm" / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        age = time.time() - metrics_path.stat().st_mtime
        if age > max_age_seconds:
            print(f"  FSM fallback: metrics.json too old ({age:.0f}s), skipping")
            return None
        data = json.loads(metrics_path.read_text())
        if not isinstance(data, dict):
            return None
        score = data.get("accuracy") if "accuracy" in data else data.get("score")
        if score is not None:
            return {"ok": True, "score": score, "source": "metrics_fallback"}
    except Exception:
        pass
    return None


def _try_fsm_deploy_and_rollout(
    workspace: str,
    model_path: str,
    fsm_config: dict,
    session,
    data_path: str = "",
    output_dir: str = "",
) -> dict | None:
    """通过 FSM-Runner 部署模型并执行 rollout 采样。"""
    print(f"\n{'='*60}")
    print(f"  Phase FSM: Deploy + Rollout")
    print(f"{'='*60}")

    model_dir = Path(model_path).resolve()
    if not model_dir.exists():
        return {"ok": False, "reason": f"model_path_not_found: {model_dir}"}

    env_overrides = _build_fsm_env_overrides(
        model_path, fsm_config.get("deploy_engine", "vllm"), data_path, output_dir,
    )

    # 获取数据总量用于进度显示
    total_samples = 0
    dp = data_path or os.environ.get("DATA_PATH", "")
    if dp:
        p = Path(dp) / "train.jsonl"
        if p.exists():
            with open(p, "r") as f:
                total_samples = sum(1 for _ in f)

    target_repo = fsm_config.get("target_repo", ".")
    try:
        with _FSMProgressMonitor("FSM Rollout", target_repo, total_samples):
            rollout_result = session.rollout(
                llm=model_dir,
                mode=fsm_config.get("mode", "smoke"),
                require_samples=True,
                env_overrides=env_overrides,
                repair_iters=fsm_config.get("repair_iters", 3),
            )
    except Exception as e:
        print(f"  FSM rollout error: {e}")
        return {"ok": False, "reason": f"rollout_exception: {e}"}

    if not rollout_result.ok:
        verify = rollout_result.verify
        failed = getattr(verify, "failed_stage", "unknown") if verify else "unknown"
        print(f"  FSM rollout failed: stage={failed}")
        return {"ok": False, "reason": f"rollout_failed: stage={failed}"}

    samples_path = _extract_samples_path(rollout_result.rollout_path, target_repo)
    print(f"  FSM rollout OK: samples={samples_path}")
    return {
        "ok": True,
        "samples_path": samples_path or "",
        "rollout_path": str(rollout_result.rollout_path or ""),
        "artifacts_dir": str(rollout_result.artifacts_dir),
    }


def _try_fsm_evaluate(
    workspace: str,
    model_path: str,
    fsm_config: dict,
    session,
    data_path: str = "",
    output_dir: str = "",
) -> dict | None:
    """通过 FSM-Runner 执行评测。"""
    print(f"\n{'='*60}")
    print(f"  Phase FSM: Evaluate")
    print(f"{'='*60}")

    env_overrides = _build_fsm_env_overrides(
        model_path, fsm_config.get("deploy_engine", "vllm"), data_path, output_dir,
    )

    target_repo = fsm_config.get("target_repo", ".")
    llm_arg = Path(model_path) if model_path else None
    try:
        with _FSMProgressMonitor("FSM Evaluate", target_repo, is_evaluate=True):
            eval_result = session.evaluate(
                llm=llm_arg,
                mode=fsm_config.get("mode", "smoke"),
                env_overrides=env_overrides or None,
                repair_iters=fsm_config.get("repair_iters", 3),
            )
    except Exception as e:
        print(f"  FSM evaluate error: {e}")
        return {"ok": False, "reason": f"evaluate_exception: {e}"}

    if not eval_result.ok:
        verify = eval_result.verify
        failed = getattr(verify, "failed_stage", "unknown") if verify else "unknown"
        print(f"  FSM evaluate failed: stage={failed}")
        fallback = _read_fsm_metrics(target_repo)
        if fallback:
            print(f"  FSM fallback OK: score={fallback['score']}")
            return fallback
        return {"ok": False, "reason": f"evaluate_failed: stage={failed}"}

    metrics = eval_result.metrics or {}
    score = metrics.get("score")
    print(f"  FSM evaluate OK: score={score}")
    return {
        "ok": True,
        "score": score,
        "metrics": metrics,
        "metrics_path": str(eval_result.metrics_path or ""),
    }


def _scaffold_fsm_evaluation(
    workspace: str,
    task_description: str,
    fsm_config: dict,
    data_path: str = "",
) -> bool:
    """在迭代循环之前，调用 OpenCode 自主探索数据并生成 task-specific 的评测脚本。"""
    target_repo = fsm_config.get("target_repo", ".")
    repo_root = Path(target_repo).resolve()

    print(f"\n{'='*60}")
    print(f"  Phase FSM-Scaffold: Adapting evaluation for task")
    print(f"{'='*60}")

    sample_count = 0
    if data_path:
        p = Path(data_path) / "train.jsonl"
        if p.exists():
            with open(p, "r") as f:
                sample_count = sum(1 for _ in f)

    prompt = build_scaffold_prompt(task_description, data_path, sample_count)

    ok = _run_opencode_agent(
        fsm_config, prompt,
        max_turns=40,
        session_title="scaffold_evaluation",
        log_name="opencode_scaffold",
        fsm_state="S0_SCAFFOLD",
        purpose="scaffold_contract",
        label="Scaffold",
    )

    if not ok:
        return False

    # 验证 scaffold 实际重写了 rollout.sh
    rollout_sh = repo_root / ".opencode_fsm" / "stages" / "rollout.sh"
    if not rollout_sh.exists():
        print("  WARNING: rollout.sh not found after scaffold")
        return False
    content = rollout_sh.read_text(encoding="utf-8")
    if "# SKELETON" in content and "reward = 0.0" in content:
        print("  WARNING: rollout.sh still contains skeleton placeholder")
        return False
    if "reward" not in content.lower():
        print("  WARNING: rollout.sh missing reward logic after scaffold")
        return False
    print(f"  Scaffold verification OK: rollout.sh rewritten ({len(content)} bytes)")
    return True


def _repair_rollout_zero_reward(
    workspace: str,
    fsm_config: dict,
    samples_path: str,
    task_description: str,
    data_path: str,
    pass_count: int,
    total_samples: int,
    repair_attempt: int,
    max_attempts: int,
) -> bool:
    """调用 OpenCode 自主诊断并修复 rollout.sh 中的 reward 计算逻辑。"""
    print(f"\n{'='*60}")
    print(f"  Rollout Repair: attempt {repair_attempt}/{max_attempts}")
    print(f"{'='*60}")

    prompt = build_rollout_repair_prompt(
        samples_path=samples_path,
        data_path=data_path,
        task_description=task_description,
        pass_count=pass_count,
        total_samples=total_samples,
        repair_attempt=repair_attempt,
        max_attempts=max_attempts,
    )

    return _run_opencode_agent(
        fsm_config, prompt,
        max_turns=30,
        session_title=f"rollout_repair_{repair_attempt}",
        log_name=f"opencode_rollout_repair_{repair_attempt}",
        fsm_state="S0_SCAFFOLD",
        purpose="repair_contract",
        label="RolloutRepair",
    )


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
    max_rollout_repair_retries: int = 2,
    fsm_config: dict | None = None,
):
    """运行固定阶段 Pipeline

    每轮迭代：
      Phase 1: Agent 写代码（探索数据 + 编写 train.py）
      Phase 2: Pipeline 执行训练（subprocess）
      Phase FSM: Deploy + Rollout + Evaluate
      Phase Analysis: 记录结果，注入到下轮 prompt
    """
    pipeline_start = time.time()
    fsm_config = fsm_config or {}

    if not data_path:
        data_path = os.environ.get("DATA_PATH", "")
    if not output_dir:
        output_dir = os.environ.get("OUTPUT_DIR", str(Path(workspace) / "output"))

    opencode_model = os.environ.get("OPENCODE_MODEL", "")
    opencode_url = os.environ.get("OPENCODE_URL", "")

    print(f"{'#'*60}")
    print(f"  OpenCode RL Pipeline (Fixed-Stage)")
    print(f"{'#'*60}")
    print(f"  Task: {task}")
    print(f"  Model: {base_model}")
    print(f"  Workspace: {workspace}")
    print(f"  Data path: {data_path}")
    print(f"  Output dir: {output_dir}")
    print(f"  Max iterations: {max_iterations}")
    print(f"  Training timeout: {training_timeout}s")
    print(f"  Agent steps/iter: {max_agent_steps}")
    print(f"  OpenCode model: {opencode_model}")
    print()

    task_description = ""
    for fname in ["description.md", "instructions.md"]:
        fpath = Path(workspace) / fname
        if fpath.exists():
            task_description += fpath.read_text() + "\n\n"

    if not task_description.strip():
        task_description = f"Benchmark: {task}\nTrain a language model using GRPO reinforcement learning.\n"
        print(f"WARNING: No description.md found, using fallback: {task}")

    data_stats = get_data_stats(data_path)
    gpu_info = get_gpu_info()
    print(f"  Data: {data_stats['count']} samples")
    print(f"  GPU: {gpu_info['num_gpus']}x {gpu_info['gpu_name']}")

    history: list[IterationResult] = []
    best_score: float | None = None
    best_iteration = -1

    # Phase FSM-Scaffold: 在迭代开始前让 OpenCode 生成 task-specific 的评测脚本
    if task_description.strip():
        scaffold_ok = _scaffold_fsm_evaluation(workspace, task_description, fsm_config, data_path=data_path)
        if not scaffold_ok:
            print("  WARNING: scaffold failed — FSM evaluation may not work correctly")

    # 每轮迭代创建一个 FSM session，deploy + rollout + evaluate 共享
    for iteration in range(1, max_iterations + 1):
        iter_start = time.time()
        elapsed_total = iter_start - pipeline_start

        print(f"\n{'#'*60}")
        print(f"  ITERATION {iteration}/{max_iterations}")
        print(f"  Elapsed: {elapsed_total:.0f}s")
        print(f"{'#'*60}")

        result = IterationResult(iteration=iteration)

        # Phase 1: Code Generation
        try:
            code_path = phase_code_generation(
                iteration, workspace, base_model,
                task_description, history, max_agent_steps,
                gpu_info=gpu_info,
                opencode_model=opencode_model,
                opencode_url=opencode_url,
            )
            result.code_path = code_path
        except Exception as e:
            print(f"  Code generation failed: {e}")
            result.code_path = str(Path(workspace) / "code" / "train.py")
            result.exit_code = -1
            result.stdout = f"Code generation error: {e}"
            history.append(result)
            continue

        # Phase 2: Training Execution（含重试）
        exit_code, stdout, train_time = phase_training(
            workspace, code_path, timeout=training_timeout,
        )

        training_log_path = str(Path(workspace) / "code" / "training_stdout.log")
        Path(training_log_path).write_text(stdout, encoding="utf-8")

        for retry in range(max_fix_retries):
            if exit_code == 0:
                break
            print(f"\n  --- Fix retry {retry + 1}/{max_fix_retries} ---")
            try:
                phase_fix_training(
                    code_path, training_log_path, data_path, workspace,
                    iteration=iteration,
                    opencode_model=opencode_model,
                    opencode_url=opencode_url,
                    max_agent_steps=max_agent_steps,
                )
                print(f"  Agent fix attempt done, re-running training...")
            except Exception as e:
                print(f"  Fix agent failed: {e}")
                break

            exit_code, stdout, extra_time = phase_training(
                workspace, code_path, timeout=training_timeout,
            )
            train_time += extra_time
            Path(training_log_path).write_text(stdout, encoding="utf-8")

        result.exit_code = exit_code
        result.stdout = stdout
        result.training_time = train_time

        # Phase FSM: Deploy + Rollout + Evaluate（含零分自动修复循环）
        trained_model = ""
        cached_samples_score: dict | None = None

        if exit_code == 0:
            trained_model = _find_trained_model(output_dir)

            # 创建单个 FSM session，本迭代内复用
            try:
                session = _create_fsm_session(fsm_config)
            except ImportError:
                print("  FSM runner not available, skipping FSM stages")
                session = None

            if session:
                for rollout_attempt in range(max_rollout_repair_retries + 1):
                    fsm_result = _try_fsm_deploy_and_rollout(
                        workspace, trained_model, fsm_config, session,
                        data_path=data_path, output_dir=output_dir,
                    )

                    if not fsm_result or not fsm_result.get("ok"):
                        break

                    samples_path = fsm_result.get("samples_path", "")
                    result.samples_path = samples_path

                    # 计算 pass_count（结果缓存供 evaluation 阶段复用）
                    if samples_path:
                        cached_samples_score = _compute_score_from_samples(samples_path)
                    else:
                        cached_samples_score = None

                    pass_count = 0
                    total = 0
                    if cached_samples_score and cached_samples_score.get("ok"):
                        pass_count = cached_samples_score.get("pass_count", 0)
                        total = cached_samples_score.get("total", 0)

                    if pass_count > 0 or rollout_attempt >= max_rollout_repair_retries:
                        break

                    # 全零 reward → 调用 OpenCode 自主修复
                    print(f"\n  All {total} samples got reward=0.0, attempting auto-repair...")
                    repair_ok = _repair_rollout_zero_reward(
                        workspace=workspace,
                        fsm_config=fsm_config,
                        samples_path=samples_path,
                        task_description=task_description,
                        data_path=data_path,
                        pass_count=pass_count,
                        total_samples=total,
                        repair_attempt=rollout_attempt + 1,
                        max_attempts=max_rollout_repair_retries,
                    )
                    if not repair_ok:
                        print(f"  Rollout repair failed, continuing with zero score")
                        break
                    print(f"  Rollout repair completed, re-running rollout...")

        # Phase Evaluation（FSM evaluate → samples fallback）
        if exit_code == 0:
            eval_score: float | None = None
            eval_source = ""

            # FSM evaluate
            if session and trained_model:
                fsm_eval = _try_fsm_evaluate(
                    workspace, trained_model, fsm_config, session,
                    data_path=data_path, output_dir=output_dir,
                )
                if fsm_eval and fsm_eval.get("ok"):
                    eval_score = _normalize_score(fsm_eval.get("score"))
                    eval_source = "FSM evaluate"

            # Fallback：复用 rollout 阶段已计算的 samples score
            if eval_score is None and result.samples_path:
                if not cached_samples_score:
                    cached_samples_score = _compute_score_from_samples(result.samples_path)
                if cached_samples_score and cached_samples_score.get("ok"):
                    eval_score = _normalize_score(cached_samples_score["score"])
                    eval_source = f"samples ({cached_samples_score['pass_count']}/{cached_samples_score['total']})"

            print(f"\n{'='*60}")
            print(f"  Phase Evaluation")
            print(f"{'='*60}")
            if eval_score is not None:
                print(f"  Score: {eval_score}")
                print(f"  Source: {eval_source}")
                result.score = eval_score
                result.model_path = trained_model
                if best_score is None or eval_score > best_score:
                    best_score = eval_score
                    best_iteration = iteration
            else:
                print(f"  No score available (FSM and samples both failed)")

        # Phase Analysis: 让 OpenCode 自分析（非最后一轮时执行）
        if iteration < max_iterations:
            try:
                analysis = phase_analysis(
                    iteration, workspace, result.code_path,
                    training_log_path, result.score,
                    samples_path=result.samples_path,
                    opencode_model=opencode_model,
                    opencode_url=opencode_url,
                    max_agent_steps=max_agent_steps,
                )
                result.analysis = analysis
            except Exception as e:
                print(f"  Analysis phase failed (non-fatal): {e}")

        history.append(result)

        iter_elapsed = time.time() - iter_start
        print(f"\n  --- Iteration {iteration} Summary ---")
        print(f"  Exit code: {result.exit_code}")
        print(f"  Score: {result.score}")
        print(f"  Best so far: {best_score} (iter {best_iteration})")
        print(f"  Iteration time: {iter_elapsed:.0f}s")

    total_time = time.time() - pipeline_start
    print(f"\n{'#'*60}")
    print(f"  Pipeline Complete")
    print(f"  Best score: {best_score}")
    print(f"  Best iteration: {best_iteration}")
    print(f"  Total iterations: {len(history)}")
    print(f"  Total time: {total_time:.0f}s")
    print(f"{'#'*60}")

    summary = {
        "task": task,
        "base_model": base_model,
        "best_score": best_score,
        "best_iteration": best_iteration,
        "total_time": total_time,
        "iterations": [
            {
                "iteration": r.iteration,
                "exit_code": r.exit_code,
                "training_time": r.training_time,
                "score": r.score,
                "samples_path": r.samples_path,
            }
            for r in history
        ],
    }
    summary_path = Path(workspace) / "pipeline_results.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  Results saved: {summary_path}")
