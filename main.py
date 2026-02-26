#!/usr/bin/env python3
"""
OpenCode RL Post-training Pipeline (Fixed-Stage)

每轮迭代：代码生成 → 训练执行 → (FSM 部署/rollout) → 评测提交 → 反馈注入。
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

from benchmarks.registry import get_benchmark, list_benchmarks
from pipeline.runner import run_pipeline


def _init_fsm_workspace(project_root: Path, target: Path) -> None:
    """将项目根目录的 .aider_fsm 模板复制到运行目录，实现每次运行隔离。

    复制内容：
    - stages/ 目录（rollout.sh, evaluation.sh 等模板脚本）
    - pipeline.yml（FSM 合同定义）
    - runner → runner_fsm 的符号链接
    """
    src_fsm = project_root / ".aider_fsm"
    dst_fsm = target / ".aider_fsm"

    if not src_fsm.exists():
        return

    dst_fsm.mkdir(parents=True, exist_ok=True)

    # 复制 stages/ 模板脚本
    src_stages = src_fsm / "stages"
    dst_stages = dst_fsm / "stages"
    if src_stages.is_dir() and not dst_stages.exists():
        shutil.copytree(src_stages, dst_stages)

    # 复制 pipeline.yml
    src_pipeline = project_root / "pipeline.yml"
    dst_pipeline = target / "pipeline.yml"
    if src_pipeline.exists() and not dst_pipeline.exists():
        shutil.copy2(src_pipeline, dst_pipeline)

    # 创建 runner → runner_fsm 的符号链接
    dst_runner = dst_fsm / "runner"
    src_runner_fsm = project_root / "runner_fsm"
    if src_runner_fsm.is_dir() and not dst_runner.exists():
        dst_runner.symlink_to(src_runner_fsm.resolve())


def main():
    parser = argparse.ArgumentParser(description="OpenCode RL Pipeline (Fixed-Stage)")
    parser.add_argument("--benchmark", type=str, default="gsm8k",
                        help="Benchmark 名称（对应 benchmarks/ 下的子目录）")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--run-dir", type=str, default="",
                        help="指定运行目录（默认自动生成 runs/{benchmark}_{timestamp}）")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--training-timeout", type=int, default=3600)
    parser.add_argument("--max-agent-steps", type=int, default=25)

    parser.add_argument("--fsm-enabled", action="store_true",
                        help="启用 FSM-Runner 自动部署/rollout/评测")
    parser.add_argument("--fsm-target-repo", type=str, default="")
    parser.add_argument("--fsm-deploy-engine", type=str, default="vllm",
                        choices=["vllm", "tgi", "local"])
    parser.add_argument("--fsm-repair-iters", type=int, default=3)
    parser.add_argument("--fsm-mode", type=str, default="smoke",
                        choices=["smoke", "full"])

    parser.add_argument("--list-benchmarks", action="store_true",
                        help="列出所有可用 benchmark 并退出")

    args = parser.parse_args()

    if args.list_benchmarks:
        names = list_benchmarks()
        if not names:
            print("No benchmarks found in benchmarks/ directory.")
        else:
            print(f"Available benchmarks ({len(names)}):")
            for n in names:
                b = get_benchmark(n)
                print(f"  {n:<20s} [{b.task_type}]  {b.description}")
        sys.exit(0)

    bench = get_benchmark(args.benchmark)
    data_dir = str(bench.data_dir.resolve())

    if args.run_dir:
        run_dir = str(Path(args.run_dir).resolve())
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = str((Path("runs") / f"{args.benchmark}_{ts}").resolve())

    # 优先使用 run.py 传入的环境变量（已指向 workspace 子目录）
    output_dir = os.environ.get("OUTPUT_DIR") or str(Path(run_dir) / "output")
    data_path = os.environ.get("DATA_PATH") or data_dir
    code_dir = str(Path(run_dir) / "code")
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(code_dir).mkdir(parents=True, exist_ok=True)

    os.environ["DATA_PATH"] = data_path
    os.environ["OUTPUT_DIR"] = output_dir
    os.environ["TRAINING_TIMEOUT"] = str(args.training_timeout)

    # FSM target_repo: 默认使用 run_dir 隔离每次运行，避免交叉污染
    fsm_target = args.fsm_target_repo or os.environ.get("FSM_TARGET_REPO", "")
    if not fsm_target:
        fsm_target = run_dir

    # 将 .aider_fsm 模板从项目根目录复制到运行目录（隔离）
    project_root = Path(__file__).resolve().parent
    _init_fsm_workspace(project_root, Path(fsm_target))

    # 在 run_dir 里创建 data 的 symlink，使 read 工具也能访问数据
    data_link = Path(run_dir) / "data"
    if not data_link.exists():
        data_link.symlink_to(Path(data_path).resolve())

    # 将 benchmark 的 description.md 复制到运行目录，供 pipeline 读取
    for desc_name in ["description.md", "instructions.md"]:
        src_desc = bench.root / desc_name
        dst_desc = Path(run_dir) / desc_name
        if src_desc.exists() and not dst_desc.exists():
            shutil.copy2(src_desc, dst_desc)

    fsm_config = {
        "enabled": args.fsm_enabled or os.environ.get("FSM_ENABLED", "").lower() in ("1", "true", "yes"),
        "target_repo": fsm_target,
        "deploy_engine": args.fsm_deploy_engine or os.environ.get("FSM_DEPLOY_ENGINE", "vllm"),
        "repair_iters": args.fsm_repair_iters,
        "mode": args.fsm_mode,
        "opencode_url": os.environ.get("OPENCODE_URL", ""),
        "opencode_model": os.environ.get("OPENCODE_MODEL", ""),
    }

    run_pipeline(
        task=args.benchmark,
        base_model=args.base_model,
        workspace=run_dir,
        data_path=data_path,
        output_dir=output_dir,
        max_iterations=args.max_iterations,
        training_timeout=args.training_timeout,
        max_agent_steps=args.max_agent_steps,
        fsm_config=fsm_config,
    )


if __name__ == "__main__":
    main()
