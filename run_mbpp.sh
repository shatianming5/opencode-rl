#!/bin/bash
cd "$(dirname "$0")"
source /data/userdata/v-tiansha/venvs/opencode-rl/bin/activate

export PYTHONUNBUFFERED=1
export OPENAI_API_KEY="31822d3f2f3644bea22155e2ffe3c263.LDkMb2wPaKheSyUt"
export OPENAI_API_BASE="https://open.bigmodel.cn/api/coding/paas/v4/"
export OPENCODE_MODEL="glm/glm-4.7"
export MODEL_PATH="/data/userdata/v-tiansha/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
export FSM_ENABLED=true

python main.py \
    --benchmark mbpp \
    --base-model "Qwen/Qwen2.5-0.5B-Instruct" \
    --max-iterations 1 \
    --training-timeout 3600 \
    --fsm-enabled
