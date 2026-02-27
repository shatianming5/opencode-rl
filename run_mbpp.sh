#!/bin/bash
cd "$(dirname "$0")"
source /data/userdata/v-tiansha/venvs/opencode-rl/bin/activate

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"
export PYTHONUNBUFFERED=1
export OPENAI_API_KEY="sk-cliproxy-local"
export OPENAI_API_BASE="http://127.0.0.1:8317/v1"
export OPENCODE_MODEL="gpt-5.3-codex-spark"

# 自动生成项目专属 opencode config，保持与环境变量一致
export XDG_CONFIG_HOME="$(pwd)/.opencode-config"
mkdir -p "$XDG_CONFIG_HOME/opencode"
cat > "$XDG_CONFIG_HOME/opencode/opencode.json" <<EOCFG
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": {
    "openai": {
      "npm": "@ai-sdk/openai",
      "name": "Auto-configured",
      "options": {
        "baseURL": "${OPENAI_API_BASE}",
        "apiKey": "${OPENAI_API_KEY}"
      },
      "models": {
        "${OPENCODE_MODEL}": { "name": "${OPENCODE_MODEL}" }
      }
    }
  }
}
EOCFG

python main.py \
    --benchmark mbpp \
    --base-model "Qwen/Qwen2.5-0.5B-Instruct" \
    --max-iterations 5 \
    --max-fix-retries 20 \
    --training-timeout 7200 \
    --stale-timeout 1800 \
    --max-verifier-retries 5 \
    --http-timeout 600
