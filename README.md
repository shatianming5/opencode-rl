# OpenCode RL

使用 OpenCode 进行 RL 后训练的自动化 Pipeline，集成 FSM-Runner 合同驱动执行框架实现模型部署、Rollout 采样和评测的全流程自动化。

## 功能

- 固定阶段式 Pipeline：代码生成 → 训练执行 → 模型部署 → Rollout 采样 → 评测
- 支持 GRPO/DPO/PPO 训练（基于 trl）
- FSM-Runner 合同驱动：通过 `pipeline.yml` 定义 deploy/rollout/evaluation 各阶段
- 阶段缓存：部署成功后自动缓存，下次跳过重复部署（SHA256 校验 + TTL + Health Check）
- Benchmark 注册表：自动发现 `benchmarks/` 下的所有 benchmark，新增只需一个 `config.yaml`
- 运行隔离：每次运行产物存放在 `runs/{benchmark}_{timestamp}/`
- 结果导出：支持导入到 RD-Agent UI 进行可视化查看

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 快速开始

### 查看可用 Benchmark

```bash
python main.py --list-benchmarks
```

### 基础 Pipeline（不启用 FSM）

```bash
export OPENAI_API_KEY="sk-1234"
export OPENAI_API_BASE="http://10.150.240.117:38888"
export OPENCODE_MODEL="gpt-5.1-codex"

python main.py --benchmark mbpp --base-model Qwen/Qwen2.5-0.5B-Instruct
```

### 启用 FSM Deploy + Rollout + Evaluate

```bash
python main.py \
    --benchmark mbpp \
    --base-model Qwen/Qwen2.5-0.5B-Instruct \
    --fsm-enabled \
    --fsm-deploy-engine local \
    --fsm-mode smoke
```

### 运行脚本（推荐）

项目提供了预配置的运行脚本，自动处理环境变量和 OpenCode 配置：

```bash
# 运行 MBPP benchmark（默认 GPU 1,2,3）
bash run_mbpp.sh

# 运行 HumanEval benchmark（默认 GPU 1,2,3）
bash run_humaneval.sh

# 指定 GPU
CUDA_VISIBLE_DEVICES=0,1 bash run_mbpp.sh
```

## LLM API 配置

### 直接使用 API 端点

在运行脚本中设置三个环境变量即可：

```bash
export OPENAI_API_KEY="sk-1234"
export OPENAI_API_BASE="http://10.150.240.117:38888"
export OPENCODE_MODEL="gpt-5.1-codex"
```

运行脚本会自动生成项目专属的 OpenCode 配置（`.opencode-config/opencode/opencode.json`），通过 `XDG_CONFIG_HOME` 加载，不影响全局 `~/.config/opencode/opencode.json`。

> **关键点**：OpenCode 的 LLM proxy 优先从 `opencode.json` 读取 `baseURL`，而非 `OPENAI_API_BASE` 环境变量。运行脚本已自动处理这一点。

### 使用 CLIProxyAPI + Codex OAuth

通过 [CLIProxyAPI](https://github.com/router-for-me/CLIProxyAPI) 可以使用 ChatGPT Plus/Pro 订阅的 Codex 模型，无需 API Key。

**1. 部署 CLIProxyAPI**

```bash
cd ~/CLIProxyAPI
curl -L -o CLIProxyAPI.tar.gz \
  "https://github.com/router-for-me/CLIProxyAPI/releases/latest/download/CLIProxyAPI_$(curl -s https://api.github.com/repos/router-for-me/CLIProxyAPI/releases/latest | grep tag_name | cut -d'"' -f4 | sed 's/^v//')_linux_amd64.tar.gz"
tar xzf CLIProxyAPI.tar.gz && chmod +x cli-proxy-api

# OAuth 登录（远程服务器需先建 SSH 隧道）
./cli-proxy-api -codex-login -no-browser -config config.yaml

# 启动服务
./cli-proxy-api -config config.yaml &
```

**2. 在运行脚本中配置**

```bash
export OPENAI_API_KEY="sk-cliproxy-local"
export OPENAI_API_BASE="http://127.0.0.1:8317/v1"
export OPENCODE_MODEL="gpt-5.3-codex"
```

## 结果导出到 UI

运行结果可以导入到 RD-Agent UI 进行可视化查看（`http://10.150.240.113:8510/`）。

```bash
# 预览（不写入）
python3 export_to_ui.py --dry-run

# 导入所有 runs/
python3 export_to_ui.py

# 导入指定 run
python3 export_to_ui.py --run-dir runs/mbpp_20260226_153910
```

自动去重，已导入的结果不会重复写入。

## 环境变量

### 核心配置

| 变量 | 用途 |
|------|------|
| `OPENAI_API_KEY` | LLM API 密钥 |
| `OPENCODE_MODEL` | OpenCode 模型名（如 `gpt-5.1-codex`） |
| `OPENAI_API_BASE` | LLM API 地址 |
| `OPENCODE_URL` | OpenCode server 地址（可选，留空自动启动本地 server） |
| `XDG_CONFIG_HOME` | 项目专属 OpenCode 配置目录（运行脚本自动设置） |
| `CUDA_VISIBLE_DEVICES` | GPU 选择 |
| `FSM_ENABLED` | 是否启用 FSM（`1`/`true`） |

### FSM 高级配置

| 变量 | 用途 | 默认值 |
|------|------|--------|
| `OPENCODE_FSM_CACHE_TTL` | 缓存 TTL 秒数 | 3600 |
| `OPENCODE_FSM_CACHE_ENABLED` | 启用阶段缓存 | 1 |
| `OPENCODE_EVAL_MODE` | 评测模式 (`smoke`/`full`) | smoke |
| `OPENCODE_EVAL_LIMIT` | 评测样本上限 | 200 |
| `OPENCODE_FSM_REQUIRE_HINTS` | 要求执行 hints | 0 |
| `OPENCODE_LLM_MODEL` | LLM 模型名（FSM 内部） | - |
| `OPENCODE_LLM_KIND` | 推理类型 (`local_hf`/`remote`) | - |
| `OPENCODE_FSM_PYTHON` | Python 解释器路径 | python3 |

## 项目结构

```
opencode-rl/
├── main.py                     # 主入口
├── config.py                   # 配置管理
├── fsm_bridge.py               # FSM-Runner 桥接层
├── export_to_ui.py             # 结果导出到 RD-Agent UI
├── pipeline.yml                # FSM 合同定义
├── requirements.txt
│
├── benchmarks/                 # Benchmark 注册表 + 数据
│   ├── registry.py             #   自动发现/路径解析
│   ├── gsm8k/
│   │   ├── config.yaml
│   │   └── data/train.jsonl
│   ├── humaneval/
│   │   ├── config.yaml
│   │   └── data/train.jsonl
│   ├── mbpp/
│   │   ├── config.yaml
│   │   └── data/train.jsonl
│   └── _template/              #   新 benchmark 模板
│       └── config.yaml
│
├── pipeline/                   # Pipeline 核心逻辑
│   ├── runner.py               #   Pipeline 编排器
│   ├── phases.py               #   各阶段实现（代码生成/训练/修复/分析）
│   ├── prompts.py              #   Agent prompt 模板
│   ├── types.py                #   数据类型定义
│   ├── stream.py               #   流式输出
│   └── utils.py                #   工具函数
│
├── runner_fsm/                 # FSM-Runner 合同驱动执行框架
│   ├── env.py                  #   EnvSession API
│   ├── core/                   #   核心执行引擎
│   │   ├── env_setup.py        #     环境初始化
│   │   ├── env_execution.py    #     阶段执行
│   │   ├── stage_cache.py      #     阶段缓存（SHA256 + TTL）
│   │   ├── pipeline_spec.py    #     解析 pipeline.yml
│   │   ├── pipeline_verify.py  #     合同验证
│   │   └── bootstrap.py        #     引导初始化
│   ├── contract/               #   合同验证 + 修复
│   │   ├── validation.py       #     验证报告
│   │   ├── repair.py           #     自动修复
│   │   └── provenance.py       #     文件变更追踪
│   ├── opencode/               #   OpenCode 客户端集成
│   │   ├── client.py           #     OpenCode server 通信
│   │   ├── tool_executor.py    #     工具执行 + 沙箱/权限
│   │   ├── tool_parser.py      #     LLM 工具调用解析
│   │   ├── llm_proxy.py        #     LLM 请求代理（含 encrypted_content 剥离）
│   │   └── prompts.py          #     OpenCode 专用 prompts
│   ├── hints/                  #   执行提示系统
│   │   ├── executor.py         #     Hint 执行
│   │   ├── probing.py          #     数据探测
│   │   ├── scoring.py          #     指标归一化
│   │   └── python_env.py       #     Python 环境管理
│   ├── utils/                  #   工具集
│   │   ├── eval_audit.py       #     评测脚本审计
│   │   ├── security.py         #     安全检查
│   │   ├── repo_resolver.py    #     仓库解析
│   │   └── subprocess.py       #     子进程工具
│   ├── generic_evaluation.py   #   通用评测逻辑
│   └── generic_rollout.py      #   通用 Rollout 逻辑
│
├── .opencode_fsm/              # FSM 运行时目录
│   ├── stages/                 #   各阶段 shell 脚本
│   │   ├── deploy_setup.sh     #     模型部署
│   │   ├── deploy_health.sh    #     健康检查
│   │   ├── rollout.sh          #     生成训练样本
│   │   └── evaluation.sh       #     计算指标
│   └── cache/                  #   阶段缓存文件
│
├── .opencode-config/           # 项目专属 OpenCode 配置（运行脚本自动生成）
│   └── opencode/opencode.json
│
├── run_mbpp.sh                 # MBPP 运行脚本
├── run_humaneval.sh            # HumanEval 运行脚本
│
└── runs/                       # 运行产物（自动生成）
    └── {benchmark}_{timestamp}/
        ├── code/               #   生成的训练代码
        │   └── train.py
        ├── output/             #   模型输出/checkpoint
        └── pipeline_results.json
```

## Pipeline 执行流程

```
main.py
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│  迭代循环 (max_iterations)                                │
│                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐           │
│  │ 代码生成  │───▶│ 训练执行  │───▶│ 错误修复  │ (重试)    │
│  │ OpenCode │    │ train.py │    │ OpenCode │           │
│  └──────────┘    └──────────┘    └──────────┘           │
│       │                                                  │
│       ▼ (FSM 启用时)                                      │
│  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────┐│
│  │  deploy  │───▶│ rollout  │───▶│ evaluate │───▶│ 分析 ││
│  │  setup   │    │ 生成样本  │    │ 计算指标  │    │      ││
│  │  health  │    │          │    │          │    │      ││
│  └─────────┘    └──────────┘    └──────────┘    └──────┘│
│       │                              │                   │
│    缓存写入                       metrics.json            │
│    (.opencode_fsm/cache/)       {ok, score, accuracy}    │
└─────────────────────────────────────────────────────────┘
```

## 新增 Benchmark

1. 创建目录 `benchmarks/{name}/`
2. 添加 `config.yaml`：
   ```yaml
   name: my_benchmark
   task_type: math
   description: "My custom benchmark"
   ```
3. 放入数据 `benchmarks/{name}/data/train.jsonl`
4. 运行 `python main.py --benchmark my_benchmark`

## 命令行参数

```
python main.py \
    --benchmark {name}              # benchmark 名称
    --base-model {model}            # 基础模型（HF 或本地路径）
    --max-iterations {n}            # 最大迭代次数（默认 1）
    --max-fix-retries {n}           # 训练失败最大修复次数（默认 10）
    --training-timeout {seconds}    # 训练超时秒数（默认 3600）
    --fsm-enabled                   # 启用 FSM deploy/rollout/evaluate
    --fsm-deploy-engine {engine}    # 部署引擎：vllm/tgi/local
    --fsm-mode {mode}               # 执行模式：smoke/full
    --run-dir {path}                # 自定义输出目录
    --list-benchmarks               # 列出可用 benchmark
```
