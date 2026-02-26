#!/bin/bash
# SKELETON — scaffold 阶段会根据 benchmark 自动重写此文件
# 本骨架只包含输入输出合约，reward 全部为 0.0（占位）。
# 真正的评测逻辑由 _scaffold_fsm_evaluation() 调用 OpenCode 生成。
set -e

RUNTIME_ENV_JSON=".opencode_fsm/runtime_env.json"
MODEL_PATH=$(jq -r '.model_path' "$RUNTIME_ENV_JSON")
DATA_PATH=$(jq -r '.data_path' "$RUNTIME_ENV_JSON")
OUTPUT_DIR="${ROLLOUT_EVAL_ARTIFACTS_DIR:-${OPENCODE_FSM_ARTIFACTS_DIR:-.opencode_fsm/artifacts}}"

echo "Model path: $MODEL_PATH"
echo "Data path: $DATA_PATH"
echo "Output dir: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

export OUTPUT_DIR
${OPENCODE_FSM_PYTHON:-python3} << 'PYTHON_SCRIPT'
import json
import os
import time
from pathlib import Path

RUNTIME_ENV_JSON = os.getenv('RUNTIME_ENV_JSON', '.opencode_fsm/runtime_env.json')
with open(RUNTIME_ENV_JSON, 'r') as f:
    runtime_env = json.load(f)

MODEL_PATH = runtime_env['model_path']
DATA_PATH = runtime_env['data_path']
OUTPUT_DIR = os.getenv('OUTPUT_DIR', runtime_env.get('output_dir', '.opencode_fsm/artifacts'))

print(f"Loading model from {MODEL_PATH}")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
import warnings
warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated")

print("Model loaded successfully")

# Load data — field names are generic; scaffold will rewrite with benchmark-specific logic
data_path = Path(DATA_PATH) / "train.jsonl"
samples = []
with open(data_path, 'r') as f:
    for line in f:
        samples.append(json.loads(line))

print(f"Loaded {len(samples)} samples from {data_path}")

results = []
start_time = time.time()

for idx, sample in enumerate(samples):
    # Use 'question' or 'prompt' field — scaffold will adapt to actual data schema
    prompt_text = sample.get('question') or sample.get('prompt') or sample.get('input') or ''

    if tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt_text}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        input_text = prompt_text

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    completion = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # SKELETON: reward=0.0 placeholder — scaffold must replace with real evaluation logic
    reward = 0.0

    results.append({
        "prompt": prompt_text,
        "completion": completion,
        "reward": reward,
        "task_id": sample.get('task_id', f"{idx}")
    })

    if (idx + 1) % 10 == 0 or idx == len(samples) - 1:
        elapsed = time.time() - start_time
        passed = sum(1 for r in results if r['reward'] >= 1.0)
        print(f"  [Rollout] {idx+1}/{len(samples)}, passed={passed}, elapsed={elapsed:.0f}s", flush=True)

samples_jsonl = Path(OUTPUT_DIR) / "samples.jsonl"
with open(samples_jsonl, 'w') as f:
    for result in results:
        f.write(json.dumps(result) + '\n')

print(f"Written {len(results)} samples to {samples_jsonl}")

rollout_info = {
    "paths": {
        "samples_jsonl": str(samples_jsonl)
    },
    "samples_jsonl": str(samples_jsonl),
    "model_path": MODEL_PATH,
    "num_samples": len(results),
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}

rollout_json = Path(".opencode_fsm") / "rollout.json"
with open(rollout_json, 'w') as f:
    json.dump(rollout_info, f, indent=2)

print(f"Rollout info written to {rollout_json}")
print(f"Total samples: {len(results)}")
PYTHON_SCRIPT

echo "Rollout completed"
