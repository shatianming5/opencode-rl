# Display Logic Simplification: opencode run replacement

## Problem
- Current architecture is overly complex: `opencode serve` + HTTP session + LLM proxy + manual tool parsing/execution (~1500 lines)
- Display shows too much noise: LLM thinking status, token log tailing, heartbeats
- Tool output truncated to 200 chars, losing important information
- Thinking process display is not useful

## Solution
Replace `OpenCodeClient` with a lightweight `RunClient` that spawns `opencode run --format json` as a subprocess.

## Architecture

```
Pipeline (phases.py)
  └── RunClient.run(prompt, on_turn=callback)
        ├── subprocess: opencode run --format json --model X --dir Y "prompt"
        ├── Parse JSON events line-by-line from stdout
        │   ├── step_start → new turn signal
        │   ├── tool_use (completed) → on_turn callback with full input/output
        │   ├── text → collect final response
        │   ├── step_finish → token statistics
        │   └── error → raise exception
        └── Watchdog: kill process after idle timeout post-completion
```

## Files to delete
- `runner_fsm/opencode/client.py` (OpenCodeClient, ~900 lines)
- `runner_fsm/opencode/tool_parser.py`
- `runner_fsm/opencode/tool_executor.py`
- `runner_fsm/opencode/llm_proxy.py`
- `runner_fsm/utils/security.py`

## Files to modify
- `runner_fsm/opencode/run_client.py` (new, ~150 lines)
- `runner_fsm/dtypes.py` — simplify TurnEvent
- `pipeline/ui.py` — improve display: full output, no thinking noise
- `pipeline/phases.py` — use RunClient

## Display format
- Full tool output (no truncation)
- No LLM thinking status lines
- Clean turn headers with elapsed time
- Token stats per step_finish
- Agent text response per turn
