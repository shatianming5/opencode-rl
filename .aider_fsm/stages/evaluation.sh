#!/bin/bash
set -e

ROLLOUT_JSON=".aider_fsm/rollout.json"
if [ ! -f "$ROLLOUT_JSON" ]; then
    echo "Error: rollout.json not found at $ROLLOUT_JSON"
    exit 1
fi

SAMPLES_JSONL=$(jq -r '.samples_jsonl // .paths.samples_jsonl' "$ROLLOUT_JSON")
echo "Evaluating samples from $SAMPLES_JSONL"

${AIDER_FSM_PYTHON:-python3} << 'PYTHON_SCRIPT'
import json
import os

with open('.aider_fsm/rollout.json', 'r') as f:
    rollout_info = json.load(f)

samples_jsonl = rollout_info.get('samples_jsonl') or rollout_info.get('paths', {}).get('samples_jsonl', '')
if not samples_jsonl or not os.path.exists(samples_jsonl):
    print(f"Error: samples file not found: {samples_jsonl}")
    exit(1)

samples = []
with open(samples_jsonl, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            samples.append(json.loads(line))

print(f"Loaded {len(samples)} samples")

pass_count = sum(1 for s in samples if float(s.get('reward', 0.0)) >= 1.0)
accuracy = pass_count / len(samples) if samples else 0.0

print(f"Pass count: {pass_count}/{len(samples)}")
print(f"Accuracy (pass@1): {accuracy:.4f}")

metrics = {
    "ok": True,
    "score": accuracy,
    "accuracy": accuracy,
    "pass_count": pass_count,
    "total": len(samples)
}

with open('.aider_fsm/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("Metrics written to .aider_fsm/metrics.json")
print(json.dumps(metrics, indent=2))
PYTHON_SCRIPT

echo "Evaluation completed"
