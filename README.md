# OpenCode RL

使用 OpenCode 驱动 RL 后训练的自动化 Pipeline Agent。通过大模型（如 GPT-5.2）进行代码生成→训练→评测→反馈的迭代循环。

> **GitHub**: https://github.com/shatianming5/opencode-rl

## 功能

- **固定阶段 Pipeline**：Code Gen → Training → Eval → Analysis，每轮迭代自动执行
- **Benchmark 注册表**：自动发现 `benchmarks/` 下的所有 benchmark，支持 Smith 系列和自定义 benchmark
- **Grading Server 集成**：训练后的模型提交到 Grading Server 评测，获取标准化分数
- **断点续跑**：每阶段完成后自动 checkpoint，`--resume` 从中断处继续
- **运行隔离**：每次运行产物存放在 `runs/{benchmark}_{timestamp}/`

## 两种使用方式

### 方式 1：独立使用

直接运行 opencode-rl 进行单个 benchmark 的训练迭代。

```bash
git clone https://github.com/shatianming5/opencode-rl.git
cd opencode-rl
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

配置环境变量后运行：

```bash
export OPENAI_API_KEY="sk-1234"
export OPENAI_API_BASE="http://your-llm-server:port"
export OPENCODE_MODEL="gpt-5.2"

python main.py --benchmark gsm8k --base-model Qwen/Qwen2.5-0.5B-Instruct
```

### 方式 2：作为 RD-Agent 的 Agent 插件

opencode-rl 可以作为 [RD-Agent](https://github.com/microsoft/RD-Agent) AutoRL-Bench 框架的一个 Agent 使用。详见 [RD-Agent OpenCode Agent 文档](https://github.com/microsoft/RD-Agent/tree/rl-posttraining/rdagent/scenarios/rl/autorl_bench/agents/opencode)。

```bash
# 在 RD-Agent 中运行（会自动调用外部 opencode-rl）
python -m rdagent.scenarios.rl.autorl_bench.run \
    --agent opencode --task gsm8k --model Qwen/Qwen2.5-0.5B-Instruct --timeout 41600
```

RD-Agent 的 `start.sh` 默认指向外部 opencode-rl 目录，通过 `OPENCODE_RL_ROOT` 环境变量可自定义路径。

## 安装

```bash
git clone https://github.com/shatianming5/opencode-rl.git
cd opencode-rl
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

此外需要 [OpenCode](https://opencode.ai/) CLI 工具：

```bash
npm install -g opencode    # 需要 Node.js >= 18
```

## 快速开始

### 1. 配置 LLM API

```bash
export OPENAI_API_KEY="sk-1234"
export OPENAI_API_BASE="http://your-llm-server:port"
export OPENCODE_MODEL="gpt-5.2"
```

### 2. 运行

```bash
# 直接运行
python main.py --benchmark humaneval --base-model Qwen/Qwen2.5-0.5B-Instruct

# 指定 GPU
CUDA_VISIBLE_DEVICES=0,1 python main.py --benchmark gsm8k --base-model Qwen/Qwen2.5-0.5B-Instruct
```

### 3. 查看结果

运行结束后查看 `runs/{benchmark}_{timestamp}/pipeline_results.json`。

## Benchmark 管理

### 查看可用 benchmark

```bash
python main.py --list-benchmarks
```

### 已有 benchmark

| 名称 | 类型 | 说明 |
|------|------|------|
| gsm8k | math | 小学数学推理 |
| humaneval | code | Python 函数生成 |
| mbpp | code | Python 编程 |
| alfworld | interactive | 文本交互环境 |
| smith-humaneval | code | Smith 系列 - HumanEval |
| smith-mbpp | code | Smith 系列 - MBPP |
| smith-math_hendrycks | math | Smith 系列 - MATH |
| smith-mmlu | qa | Smith 系列 - MMLU |
| smith-bbh | reasoning | Smith 系列 - BIG-Bench Hard |
| smith-arc_agi | reasoning | Smith 系列 - ARC-AGI |
| smith-pal | code+math | Smith 系列 - PAL |
| smith-zero_shot_cot | reasoning | Smith 系列 - Zero-Shot CoT |

### 新增 Benchmark

1. 创建目录 `benchmarks/my_benchmark/`
2. 添加 `config.yaml`（名称、类型、数据源）
3. 添加 `description.md`（Agent 的任务说明）
4. 准备 `data/train.jsonl`（可手动放置或通过 `download_data.py` 自动下载）

参考 `benchmarks/_template/` 目录中的模板。

## Pipeline 执行流程

```
每轮迭代：

Code Gen → Training → Eval → Analysis → 下一轮
   │          │         │        │
   │          │         │        └─ Agent 总结结果，规划改进方向
   │          │         └─ 提交模型到 Grading Server 评分
   │          └─ accelerate launch train.py（GRPO 训练）
   └─ Agent 生成/修改 train.py
```

失败时自动重试，支持 `--resume` 断点续跑。

## 命令行参数

```bash
python main.py \
    --benchmark {name}              # benchmark 名称（默认 gsm8k）
    --base-model {model}            # 基础模型路径（默认 Qwen/Qwen2.5-0.5B-Instruct）
    --max-iterations {n}            # 最大迭代次数（默认 5）
    --max-retries {n}               # 各阶段失败重试次数（默认 20）
    --training-timeout {seconds}    # 训练超时（默认 7200）
    --stale-timeout {seconds}       # LLM 无活动超时（默认 1800）
    --http-timeout {seconds}        # HTTP 连接超时（默认 600）
    --eval-timeout {seconds}        # 评测超时（默认 7200）
    --max-agent-steps {n}           # Agent 每阶段最大步数（默认 25）
    --resume                        # 从 checkpoint 断点续跑
    --run-dir {path}                # 自定义输出目录
    --list-benchmarks               # 列出可用 benchmark
```

## 环境变量

| 变量 | 用途 |
|------|------|
| `OPENAI_API_KEY` | LLM API 密钥 |
| `OPENCODE_MODEL` | OpenCode 模型名（推荐 gpt-5.2） |
| `OPENAI_API_BASE` | LLM API 地址 |
| `CUDA_VISIBLE_DEVICES` | GPU 选择 |
| `GRADING_SERVER_URL` | Grading Server 地址（RD-Agent 模式下自动设置） |

## 项目结构

```
opencode-rl/
├── main.py                      # 主入口
├── requirements.txt             # Python 依赖
│
├── benchmarks/                  # Benchmark 注册表 + 数据
│   ├── registry.py              #   自动发现
│   ├── download.py              #   数据下载工具
│   ├── _template/               #   新 benchmark 模板
│   ├── gsm8k/                   #   GSM8K 数学推理
│   ├── humaneval/               #   HumanEval 代码生成
│   ├── mbpp/                    #   MBPP 编程
│   ├── alfworld/                #   ALFWorld 交互环境
│   └── smith-*/                 #   Smith 系列 benchmark
│
├── pipeline/                    # Pipeline 核心
│   ├── runner.py                #   状态机主循环 + checkpoint
│   ├── phases.py                #   各阶段实现（code_gen/train/eval/analysis）
│   ├── prompts.py               #   Agent prompt 模板
│   ├── types.py                 #   数据类型（Phase/State/Result）
│   ├── state.py                 #   checkpoint 存取
│   ├── ui.py                    #   终端 UI
│   ├── stream.py                #   流式输出
│   └── utils.py                 #   工具函数
│
├── runner_fsm/                  # OpenCode 客户端 & 工具执行
│   ├── opencode/
│   │   ├── client.py            #   OpenCode server 通信
│   │   ├── llm_proxy.py         #   LLM 请求代理（token 统计）
│   │   └── tool_*.py            #   工具调用解析与执行
│   ├── core/                    #   环境设置 & 执行
│   ├── contract/                #   合约验证 & 修复
│   ├── hints/                   #   提示注入 & 评分
│   └── utils/                   #   安全 & 子进程管理
│
└── runs/                        # 运行产物（自动生成，不入 git）
    └── {benchmark}_{timestamp}/
        ├── code/
        │   └── train.py         #   Agent 生成的训练代码
        ├── output/              #   模型 checkpoint
        ├── checkpoint.json      #   pipeline 断点
        └── pipeline_results.json
```
