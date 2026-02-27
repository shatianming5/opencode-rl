"""Pipeline prompt 构建逻辑 — 探索式自主 prompt。"""

import os
from pathlib import Path

from .types import IterationResult


def build_verifier_prompt(
    workspace: str,
    data_path: str,
    task_description: str,
) -> str:
    """构建验证器生成阶段的 prompt — Phase 0。

    让 Agent 分析 benchmark 数据格式，编写 verifier.py。
    Agent 不知道后续训练代码的存在，只关注验证逻辑。
    """
    train_jsonl = str(Path(data_path) / "train.jsonl") if Path(data_path).is_dir() else data_path
    verifier_path = f"{workspace}/code/verifier.py"

    return f"""你是一个评测验证器工程师。你的任务是编写一个验证函数，判断模型生成的 completion 是否正确。

## 任务描述
{task_description.strip()}

## 数据路径
- 训练数据：{train_jsonl}

## 第一步：探索数据格式（必须先做）

用 terminal 执行以下命令，理解数据结构：

```bash
# 查看前 3 条数据，理解字段和格式
head -3 {train_jsonl}

# 用 python 分析字段结构
python3 -c "
import json
with open('{train_jsonl}') as f:
    sample = json.loads(f.readline())
for k, v in sample.items():
    print(f'  {{k}} ({{type(v).__name__}}): {{repr(v)[:300]}}')
"
```

## 第二步：编写验证器

根据数据格式，编写 `{verifier_path}`，实现以下接口：

```python
def verify(completion: str, sample: dict) -> dict:
    \"\"\"验证单个 completion 的正确性。

    Args:
        completion: 模型生成的文本
        sample: train.jsonl 中的原始数据（包含 question, answer, test 等字段）

    Returns:
        {{"passed": bool, "reward": float, "reason": str}}
        - passed: 是否通过验证
        - reward: 1.0（通过）或 0.0（不通过）
        - reason: 判定原因的简短描述
    \"\"\"
```

## 验证器编写要求

1. **代码类任务**（如果数据中有 test/entry_point 等字段）：
   - 从 completion 中提取代码（处理 markdown 代码块包裹的情况）
   - 拼接 test 函数后用 exec() 执行
   - 设置超时保护（5 秒），用 signal 或 multiprocessing
   - 捕获所有异常，执行失败 = 不通过

2. **数学类任务**（如果数据中有数值型 answer）：
   - 从 completion 中提取最终数字答案
   - 与标准答案比较（容忍格式差异：逗号、$符号、空格等）

3. **通用要求**：
   - 函数必须是纯函数，不依赖外部状态
   - 必须处理 completion 为空字符串的情况（返回 passed=False）
   - 不能抛出异常，所有错误都应 catch 并返回 passed=False
   - 不要在 verify() 函数外做 import 以外的操作（不要有 main 或 side effects）

## 第三步：自测验证器

编写完成后，必须自测：

```python
python3 -c "
import json, sys
sys.path.insert(0, '{workspace}/code')
from verifier import verify

with open('{train_jsonl}') as f:
    sample = json.loads(f.readline())

# 正确答案应通过
answer = sample.get('answer') or sample.get('canonical_solution') or sample.get('response', '')
result = verify(answer, sample)
print(f'Correct answer test: {{result}}')
assert result['passed'], f'FAIL: correct answer not passed: {{result}}'

# 空字符串应不通过
result_empty = verify('', sample)
print(f'Empty string test: {{result_empty}}')
assert not result_empty['passed'], f'FAIL: empty string passed: {{result_empty}}'

print('All tests passed!')
"
```

如果自测失败，修复并重新测试，直到通过。

## 约束
- 只写 `{verifier_path}`，不要写其他文件
- 完成后调用 finish 工具结束
"""


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
            agent_s = f"{h.agent_score:.2f}" if h.agent_score is not None else "-"
            status = "OK" if h.exit_code == 0 else f"FAIL({h.exit_code})"
            rows.append(f"| {h.iteration} | {status} | {h.training_time:.0f}s | {score_s} | {agent_s} |")

        history_section = "\n## 历史记录\n"
        history_section += "| 轮次 | 状态 | 耗时 | 管线验证分数 | Agent自报分数 |\n"
        history_section += "|------|------|------|-------------|-------------|\n"
        history_section += "\n".join(rows) + "\n"

        # 注入上一轮的分析报告，让 agent 直接获得之前的诊断洞察
        last = history[-1]
        if last.analysis and last.analysis.strip():
            history_section += f"\n## 上一轮诊断报告（第 {last.iteration} 轮）\n"
            history_section += last.analysis.strip()[:3000] + "\n"

        # 注入上一轮评测样本统计
        if last.samples_path:
            from .utils import get_rollout_samples_stats
            stats = get_rollout_samples_stats(last.samples_path)
            if stats:
                history_section += f"\n## 上一轮评测统计\n"
                history_section += f"- 样本数：{stats['total_samples']}\n"
                history_section += f"- 平均评测得分：{stats['avg_eval_score']}\n"
                history_section += f"- 评测通过率：{stats['pass_rate']}\n"
                history_section += f"- 平均 completion 长度：{stats['avg_completion_len']} 字符\n"
                history_section += f"- 评测样本文件：{last.samples_path}（可用 head 查看具体样本）\n"

        history_section += f"""
## 上一轮文件
- 上一轮代码：{workspace}/code/train.py
- 上一轮训练日志：{workspace}/code/training_stdout.log

请根据上面的诊断报告、评测统计和历史分数，针对性地改进代码。如果需要更多信息，可以自行读取日志、样本和代码。
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
2. 再设计：选择训练方法、设计训练 reward 函数、确定超参数
3. 最后写代码：生成 {workspace}/code/train.py

## 输出合约
- 文件：{workspace}/code/train.py
- 执行方式：pipeline 用 `accelerate launch train.py` 运行（自动多卡 DDP）
- 环境变量：MODEL_PATH, DATA_PATH, OUTPUT_DIR 在运行时可用
- 预装库：torch, transformers, trl, datasets, accelerate, peft（禁止 pip install）
- 训练超时：{training_timeout} 秒
- 训练完成后必须把模型保存到 $OUTPUT_DIR

## 评测模式合约（--eval-only）

train.py 必须支持 --eval-only 参数。传入时：
1. 不训练，跳过 GRPOTrainer
2. 从 $OUTPUT_DIR 加载训练后模型（自动检测 LoRA adapter）
3. 对 train.jsonl 全量样本生成 completion（用训练时相同的 prompt 格式 + chat template）
4. 输出 $OUTPUT_DIR/samples.jsonl（每行 JSON：prompt, completion, reward, task_id）
5. 每 10 条打印 [Eval] done/total
6. 结束时打印总结

注意：管线会用独立的验证器对 completion 进行评分。你的 reward 字段仅供参考。
重要：确保 completion 字段包含模型的完整输出，管线验证器需要从中提取答案。

LoRA 加载模式：
  if adapter_config.json 存在 → PeftModel.from_pretrained + merge_and_unload
  else → AutoModelForCausalLM.from_pretrained

## 任务描述
{task_description}

## 提示
- 可以用 `python3 -c "..."` 快速验证想法
- 可以用 `head -5 {data_path}/train.jsonl` 查看数据
- TRL 的 GRPOTrainer 适合这类 RL 后训练任务
- 你只负责写代码，不要自己执行训练脚本。pipeline 会用 accelerate 自动运行
- 不要修改 {workspace}/code/verifier.py（如果存在），那是管线锁定的验证器
- 完成后调用 finish 工具结束

## 重要：文件读取限制
- **禁止用 read 工具读取超过 500 行的文件**，这会导致系统卡死
- 查看大型库文件（如 trl 源码、transformers 源码）时，只用 bash 命令：
  - `python3 -c "import inspect; from trl import GRPOTrainer; print(inspect.signature(GRPOTrainer.__init__))"` 查看函数签名
  - `grep -n 'def method_name' /path/to/file.py` 定位函数位置
  - `sed -n '100,150p' /path/to/file.py` 读取指定行范围
- **绝对不要** read 整个 `grpo_trainer.py`、`modeling_*.py` 等库源码文件
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

## 重要：文件读取限制
- **禁止用 read 工具读取超过 500 行的文件**，会导致系统卡死
- 查看库源码时用 `grep -n`、`sed -n 'N,Mp'`、`python3 -c "import inspect; ..."` 等方式只看关键片段
"""


def build_analysis_prompt(
    iteration: int,
    workspace: str,
    code_path: str,
    training_log_path: str,
    score: float | None,
    samples_path: str = "",
    verification_summary: str = "",
) -> str:
    """构建自分析 prompt — 探索式，让 agent 自主查阅所有资料写诊断报告。"""

    verification_section = ""
    if verification_summary:
        verification_section = f"""
## 管线独立验证结果
{verification_summary}

注意：管线验证分数是由独立验证器计算的权威分数，不受 train.py 中的 reward 函数影响。
如果 Agent 自报分数与管线验证分数差距大，说明 train.py 的 reward 函数有问题。
"""

    return f"""第 {iteration} 轮训练和评测已完成。请分析结果并写出诊断报告。

## 结果概览
- 管线验证分数：{score if score is not None else "无（评测失败或未运行）"}

## 可用资料（请自行查阅）
- 任务描述：{workspace}/description.md
- 训练代码：{code_path}
- 训练日志：{training_log_path}
- 评测样本：{samples_path or "（无）"}（JSONL，每行有 prompt/completion/reward 字段）
{verification_section}
## 你的任务
分析训练过程：读代码、日志、评测样本。理解发生了什么、为什么。

将分析写入 {workspace}/code/analysis.md，包含：
- 做得好的和做得不好的地方
- 性能问题的根因
- 下一轮的具体改进建议（最多3条，按优先级排序）

用日志和评测样本中的具体数据支撑你的分析（引用评测通过率、loss 趋势、具体样本等）。

## 重要：文件读取限制
- **禁止用 read 工具读取超过 500 行的文件**，会导致系统卡死
- 查看大文件用 `head`、`tail`、`grep -n`、`sed -n 'N,Mp'` 只看关键片段

完成后调用 finish 工具结束。
"""


def build_eval_repair_prompt(
    code_path: str,
    samples_path: str,
    data_path: str,
    task_description: str,
    pass_count: int,
    total_samples: int,
    repair_attempt: int,
    max_attempts: int,
    eval_log: str = "",
) -> str:
    """构建 --eval-only 零分修复 prompt — 让 OpenCode 自主诊断并修复 train.py 的评测逻辑。"""
    train_jsonl = str(Path(data_path) / "train.jsonl") if data_path else ""

    eval_log_section = ""
    if eval_log:
        tail = eval_log[-3000:]
        eval_log_section = f"\n## 评测日志（尾部）\n```\n{tail}\n```\n"

    # 区分 crash vs 零分
    if total_samples == 0:
        situation = (
            "train.py --eval-only 执行失败或未产生 samples.jsonl。"
            "可能是 --eval-only 参数未正确处理、LoRA 加载失败、或脚本崩溃。"
        )
    else:
        situation = (
            f"train.py --eval-only 产生了 {total_samples} 个样本，"
            f"但只有 {pass_count} 个通过（通过率 {pass_count}/{total_samples}）。"
            "评测 reward（通过/不通过判定）逻辑很可能有问题。"
        )

    # 构建探索命令——仅在 samples_path 非空时包含样本查看
    if samples_path:
        samples_explore = f"""# 1. 看模型实际输出了什么
head -3 {samples_path}

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
"""
    else:
        samples_explore = "# samples.jsonl 不存在，跳过样本查看。先修复 --eval-only 使其能正常运行并输出 samples.jsonl。\n"

    return f"""{situation}
这是第 {repair_attempt}/{max_attempts} 次修复尝试。

## 任务描述
{task_description.strip()}
{eval_log_section}
## 第一步：自主探索（必须先做，不要跳过）

用 terminal 执行以下命令，理解当前状况：

```bash
{samples_explore}
# 2. 看数据格式和测试用例结构
head -2 {train_jsonl}

# 3. 看当前 train.py 中的 --eval-only 和 reward 逻辑
cat {code_path}
```

根据探索结果，自己判断：
- 模型输出的 completion 是什么样的？
- 数据中的测试用例是什么格式？
- --eval-only 模式的 reward 逻辑为什么判定失败？
- 常见问题：LoRA 加载失败、reward 函数参数格式差异、prompt 格式不一致、代码提取方式有误

## 第二步：修复 train.py 的评测逻辑

根据探索结果，修复 `{code_path}` 中 --eval-only 模式的评测逻辑。

注意：管线会用独立验证器重新评分，你的 reward 字段仅供参考。
重要：确保 samples.jsonl 的 completion 字段包含模型的完整原始输出。

## 第三步：验证修复（必须做！）

修复后，必须用 terminal 验证：

1. 取 train.jsonl 第 1 条数据的**正确答案**作为 completion，跑 reward 函数 → 应得 reward=1.0（通过）
2. 用空字符串或错误答案作为 completion → 应得 reward=0.0（不通过）
3. 如果结果不对，继续修复并重新验证

## 约束
- 只修改 `{code_path}`，不要改变训练逻辑，只修复 --eval-only 模式
- 不要修改 verifier.py（如果存在），那是管线锁定的验证器
- **禁止用 read 工具读取超过 500 行的文件**（会导致系统卡死），用 grep/sed/head 查看关键片段
- 完成后调用 finish 工具
"""
