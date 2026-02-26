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

        # 注入上一轮的分析报告，让 agent 直接获得之前的诊断洞察
        last = history[-1]
        if last.analysis and last.analysis.strip():
            history_section += f"\n## 上一轮诊断报告（第 {last.iteration} 轮）\n"
            history_section += last.analysis.strip()[:3000] + "\n"

        # 注入上一轮 rollout 样本统计
        if last.samples_path:
            from .utils import get_rollout_samples_stats
            stats = get_rollout_samples_stats(last.samples_path)
            if stats:
                history_section += f"\n## 上一轮 Rollout 统计\n"
                history_section += f"- 样本数：{stats['total_samples']}\n"
                history_section += f"- 平均 reward：{stats['avg_reward']}\n"
                history_section += f"- reward>0 比例：{stats['reward_positive_ratio']}\n"
                history_section += f"- 平均 completion 长度：{stats['avg_completion_len']} 字符\n"
                history_section += f"- Rollout 样本文件：{last.samples_path}（可用 head 查看具体样本）\n"

        history_section += f"""
## 上一轮文件
- 上一轮代码：{workspace}/code/train.py
- 上一轮训练日志：{workspace}/code/training_stdout.log

请根据上面的诊断报告、Rollout 统计和历史分数，针对性地改进代码。如果需要更多信息，可以自行读取日志、样本和代码。
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


def build_scaffold_prompt(
    task_description: str,
    data_path: str,
    sample_count: int,
) -> str:
    """构建 scaffold prompt — 让 OpenCode 自主探索数据并编写评测脚本。"""
    train_jsonl_path = str(Path(data_path) / "train.jsonl") if data_path else ""

    return f"""你是评测脚本工程师。任务：重写 `.opencode_fsm/stages/rollout.sh` 和 `.opencode_fsm/stages/evaluation.sh`。

## 第一步：自主探索（必须先做，不要跳过）

用 terminal 执行以下命令，理解数据格式和现有脚本：

```bash
# 1. 看数据格式和字段
head -2 {train_jsonl_path}
python3 -c "import json; d=json.loads(open('{train_jsonl_path}').readline()); print({{k: type(v).__name__ for k,v in d.items()}})"
python3 -c "import json; d=json.loads(open('{train_jsonl_path}').readline()); [print(f'  {{k}}: {{repr(v)[:200]}}') for k,v in d.items()]"

# 2. 看现有脚本了解结构
cat .opencode_fsm/stages/deploy_setup.sh
cat .opencode_fsm/stages/rollout.sh
cat .opencode_fsm/stages/evaluation.sh
```

根据探索结果，自己判断：
- 数据有哪些字段？哪个是 prompt？哪个是测试用例？
- 测试用例是什么格式？（字符串？列表？assert 语句？check 函数？）
- 怎么用测试用例验证模型输出的正确性？

## 第二步：任务描述

{task_description.strip()}

## 重要：骨架脚本说明

当前 `rollout.sh` 是一个**通用骨架**，reward 全部写 0.0（占位符）。你**必须**替换为真正的评测逻辑。
当前 `evaluation.sh` 已经是通用的（读 samples.jsonl → 算 reward>=1.0 比例 → 写 metrics.json），通常不需要修改，除非 reward 的含义与 pass/fail 不同。

## 第三步：重写 rollout.sh

**职责**：加载训练后模型 → 对每条数据生成 completion → 用真实测试判定对错 → 写结果

### 输入输出合约
- 输入：`.opencode_fsm/runtime_env.json`（含 `model_path`, `data_path`）
- 输出 1：`$OUTPUT_DIR/samples.jsonl` — 每行 `{{"prompt": "...", "completion": "...", "reward": 0.0或1.0, "task_id": "..."}}`
- 输出 2：`.opencode_fsm/rollout.json` — `{{"paths": {{"samples_jsonl": "<绝对路径>"}}, "num_samples": N}}`
- OUTPUT_DIR 取 `${{ROLLOUT_EVAL_ARTIFACTS_DIR:-${{OPENCODE_FSM_ARTIFACTS_DIR:-.opencode_fsm/artifacts}}}}`

### 硬约束
| 约束 | 要求 |
|------|------|
| 评测范围 | **全量 {sample_count} 条**，禁止截断（不能有 EVAL_LIMIT=5） |
| reward 真实性 | 必须用 exec() 执行真实测试用例，禁止字符串匹配或"非空=1" |
| 超时保护 | 每条样本测试执行最多 **5 秒**（用 multiprocessing.Pool + timeout） |
| GPU 推理 | torch.bfloat16 + device_map="auto"，禁止 CPU float32 |
| Chat 模板 | 训练用 chat prompt，推理也要用 tokenizer.apply_chat_template |
| 总运行时间 | 控制在 **20 分钟**以内 |
| 进度输出 | 每 10 条打印 `[Rollout] done/total, passed, elapsed`，全部 flush=True |

## 第四步：重写 evaluation.sh

- 读 `.opencode_fsm/rollout.json` → 找 samples.jsonl
- 算 pass@1 = (reward >= 1.0 的样本数) / 总数
- 写 `.opencode_fsm/metrics.json`：`{{"ok": true, "score": <0-1>, "accuracy": <0-1>, "pass_count": N, "total": M}}`
- 打印结果

## 第五步：Dry-run 验证（必须做！）

写完脚本后，**必须用 terminal 验证测试逻辑的正确性**：

1. 取 train.jsonl 第 1 条数据的**正确答案**作为 completion，跑测试逻辑 → 应得 reward=1.0
2. 用空字符串作为 completion → 应得 reward=0.0
3. 如果结果不对，修复代码并重新验证

示例验证代码（你需要根据实际数据格式调整）：
```python
import json
data = json.loads(open('{train_jsonl_path}').readline())
# ... 用 data 的正确答案测试你的 reward 逻辑 ...
```

## 约束
- 只能修改 `.opencode_fsm/stages/` 下的文件
- 预装库：torch, transformers, datasets, json, multiprocessing
- 完成后调用 finish 工具
"""


def build_rollout_repair_prompt(
    samples_path: str,
    data_path: str,
    task_description: str,
    pass_count: int,
    total_samples: int,
    repair_attempt: int,
    max_attempts: int,
) -> str:
    """构建 rollout 零分修复 prompt — 让 OpenCode 自主探索诊断并修复 reward 逻辑。"""
    train_jsonl = str(Path(data_path) / "train.jsonl") if data_path else ""

    return f"""Rollout 产生了 {total_samples} 个样本，但只有 {pass_count} 个通过（通过率 {pass_count}/{total_samples}）。
reward 计算逻辑很可能有问题。这是第 {repair_attempt}/{max_attempts} 次修复尝试。

## 任务描述
{task_description.strip()}

## 第一步：自主探索（必须先做，不要跳过）

用 terminal 执行以下命令，理解当前状况：

```bash
# 1. 看模型实际输出了什么
head -3 {samples_path}

# 2. 看数据格式和测试用例结构
head -2 {train_jsonl}

# 3. 看当前 reward 计算逻辑
cat .opencode_fsm/stages/rollout.sh

# 4. 用 python 手动跑一条样本的 reward 逻辑，看具体报错
python3 -c "
import json
sample = json.loads(open('{samples_path}').readline())
data = json.loads(open('{train_jsonl}').readline())
print('=== Sample ===')
for k,v in sample.items():
    print(f'  {{k}}: {{repr(v)[:200]}}')
print('=== Data ===')
for k,v in data.items():
    print(f'  {{k}}: {{repr(v)[:200]}}')
"
```

根据探索结果，自己判断：
- 模型输出的 completion 是什么样的？
- 数据中的测试用例是什么格式？
- rollout.sh 中的 reward 逻辑为什么全部判为 0？是 exec 报错？是字段名不对？是解析方式有问题？

## 第二步：修复 reward 计算逻辑

根据探索结果，修复 `.opencode_fsm/stages/rollout.sh` 中的 reward 计算逻辑，使其能正确判定模型输出的对错。

## 第三步：验证修复（必须做！）

修复后，必须用 terminal 验证：

1. 取 train.jsonl 第 1 条数据的**正确答案**作为 completion，跑 reward 逻辑 → 应得 reward=1.0
2. 用空字符串或错误答案作为 completion → 应得 reward=0.0
3. 如果结果不对，继续修复并重新验证

## 约束
- 只能修改 `.opencode_fsm/stages/` 下的文件
- 不要修改模型加载、推理、数据读取等部分，只修复 reward 计算逻辑
- 完成后调用 finish 工具
"""
