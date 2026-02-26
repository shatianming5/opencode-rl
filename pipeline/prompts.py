"""Pipeline prompt 构建逻辑 — 探索式自主 prompt。"""

import os
from pathlib import Path

from .types import IterationResult


def build_code_prompt(
    iteration: int,
    workspace: str,
    base_model: str,
    task_description: str,
    history: list[IterationResult],
    gpu_info: dict | None = None,
) -> str:
    """构建代码生成阶段的 prompt — 探索式，让 agent 自主发现数据格式和设计方案。"""
    model_path = os.environ.get("MODEL_PATH", "")
    data_path = os.environ.get("DATA_PATH", "")
    output_dir = os.environ.get("OUTPUT_DIR", "")
    training_timeout = os.environ.get("TRAINING_TIMEOUT", "3600")

    gpu_section = ""
    if gpu_info:
        gpu_section = f"- 硬件：{gpu_info['num_gpus']}x {gpu_info['gpu_name']}"

    history_section = ""
    if history:
        rows = []
        for h in history:
            score_s = f"{h.score:.2f}" if h.score is not None else "-"
            status = "OK" if h.exit_code == 0 else f"FAIL({h.exit_code})"
            rows.append(f"| {h.iteration} | {status} | {h.training_time:.0f}s | {score_s} |")

        history_section = "\n## 历史记录\n"
        history_section += "| 轮次 | 状态 | 耗时 | 分数 |\n"
        history_section += "|------|------|------|------|\n"
        history_section += "\n".join(rows) + "\n"
        history_section += f"""
- 上一轮代码：{workspace}/code/train.py
- 上一轮训练日志：{workspace}/code/training_stdout.log
- 上一轮诊断报告：{workspace}/code/analysis.md（如果存在）

请先阅读诊断报告和训练日志，理解上一轮的问题，再改进代码。
"""

    return f"""你是 RL 后训练工程师。目标：写一个训练脚本来提升模型在下面任务上的性能。

## 工作空间
- 代码目录：{workspace}/code/（在这里写 train.py）
- 训练数据：{data_path}/train.jsonl
- 基础模型：{model_path}（{base_model}）
- 输出目录：{output_dir}（训练后的模型保存在这里）
- 任务描述：{workspace}/description.md
{gpu_section}

## 你的任务
1. 先探索：用 terminal 查看数据格式、读 description.md、了解任务要求
2. 再设计：选择训练方法、设计 reward 函数、确定超参数
3. 最后写代码：生成 {workspace}/code/train.py

## 输出合约
- 文件：{workspace}/code/train.py
- 执行方式：pipeline 用 `accelerate launch train.py` 运行（自动多卡 DDP）
- 环境变量：MODEL_PATH, DATA_PATH, OUTPUT_DIR 在运行时可用
- 预装库：torch, transformers, trl, datasets, accelerate, peft（禁止 pip install）
- 训练超时：{training_timeout} 秒
- 训练完成后必须把模型保存到 $OUTPUT_DIR

## 任务描述
{task_description}

## 提示
- 可以用 `python3 -c "..."` 快速验证想法
- 可以用 `head -5 {data_path}/train.jsonl` 查看数据
- TRL 的 GRPOTrainer 适合这类 RL 后训练任务
- 你只负责写代码，不要自己执行训练脚本。pipeline 会用 accelerate 自动运行
- 完成后调用 finish 工具结束
{history_section}"""


def build_fix_prompt(
    code_path: str,
    error_log_path: str,
    data_path: str,
    workspace: str,
) -> str:
    """构造训练失败后的修复 prompt — 探索式，让 agent 自主读日志和代码诊断。"""

    return f"""训练脚本执行失败了。请诊断错误并修复代码。

## 工作空间
- 需要修复的代码：{code_path}
- 错误日志：{error_log_path}
- 训练数据：{data_path}（只读）

## 你的任务
1. 读错误日志，理解出了什么问题
2. 读当前代码，找到 bug
3. 如果需要，检查训练数据验证你的理解
4. 可以跑小段测试代码验证修复
5. 修改 {code_path}
6. 不要运行完整训练——pipeline 会执行
7. 完成后调用 finish 工具结束
"""


def build_analysis_prompt(
    iteration: int,
    workspace: str,
    code_path: str,
    training_log_path: str,
    score: float | None,
    samples_path: str = "",
    task_description: str = "",
) -> str:
    """构建自分析 prompt — 探索式，让 agent 自主查阅所有资料写诊断报告。"""

    return f"""第 {iteration} 轮训练和评测已完成。请分析结果并写出诊断报告。

## 结果概览
- 评测分数：{score if score is not None else "无（评测失败或未运行）"}

## 可用资料（请自行查阅）
- 任务描述：{workspace}/description.md
- 训练代码：{code_path}
- 训练日志：{training_log_path}
- Rollout 样本：{samples_path or "（无）"}（JSONL，每行有 prompt/completion/reward 字段）

## 你的任务
分析训练过程：读代码、日志、rollout 样本。理解发生了什么、为什么。

将分析写入 {workspace}/code/analysis.md，包含：
- 做得好的和做得不好的地方
- 性能问题的根因
- 下一轮的具体改进建议（最多3条，按优先级排序）

用日志和样本中的具体数据支撑你的分析（引用 reward 值、loss 趋势、具体样本等）。

完成后调用 finish 工具结束。
"""
