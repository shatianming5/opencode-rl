"""流式输出：实时打印 OpenCode 每一轮的动作。"""

import time


def make_stream_printer(label: str):
    """返回一个 on_turn 回调，实时打印 agent 每轮的思考和工具调用。

    输出格式示例：
        [CodeGen iter1] Turn 1 (12s)
          Agent: 让我先看看数据格式...
          > bash: head -5 /data/train.jsonl
          < (rc=0) {"question": "What is 2+3?", ...
          > read: /workspace/description.md
          < ok (2345 bytes)
        [CodeGen iter1] Turn 2 (34s)
          Agent: 数据是数学问答，用 GRPOTrainer...
          > write: /workspace/code/train.py
          < ok (3456 bytes)
        [CodeGen iter1] Done (2 turns, 45s)
    """
    start = time.time()
    print(f"  [{label}] Waiting for agent response...", flush=True)

    def _print(event):
        elapsed = time.time() - start

        if event.finished:
            print(f"  [{label}] Done (turn {event.turn}, {elapsed:.0f}s)", flush=True)
            return

        print(f"\n  [{label}] Turn {event.turn} ({elapsed:.0f}s)", flush=True)

        # agent 思考摘要：取第一个非空行，截取 200 字符
        if event.assistant_text:
            lines = [l.strip() for l in event.assistant_text.strip().splitlines() if l.strip()]
            # 跳过 tool call 标签行，找 agent 自己说的话
            summary = ""
            for line in lines:
                if line.startswith("<") or line.startswith("```"):
                    continue
                summary = line[:200]
                break
            if summary:
                print(f"    Agent: {summary}", flush=True)

        # 打印每个 tool call + 对应结果
        results = list(event.results) if event.results else []
        for i, call in enumerate(event.calls or []):
            result = results[i] if i < len(results) else None
            payload = call.payload if isinstance(call.payload, dict) else {}

            if call.kind == "bash":
                cmd = str(payload.get("command", ""))
                # 截断过长命令
                if len(cmd) > 150:
                    cmd = cmd[:147] + "..."
                print(f"    > bash: {cmd}", flush=True)
                if result:
                    rc = result.detail.get("rc", "?")
                    stdout = str(result.detail.get("stdout") or "").strip()
                    stderr = str(result.detail.get("stderr") or "").strip()
                    if result.ok:
                        # 输出截取前 200 字符，单行化
                        out = stdout[:200].replace("\n", " | ") if stdout else ""
                        print(f"    < (rc={rc}) {out}", flush=True)
                    else:
                        err = stderr[:200].replace("\n", " | ") if stderr else stdout[:200].replace("\n", " | ")
                        print(f"    < (rc={rc}) {err}", flush=True)

            elif call.kind == "file":
                file_path = str(payload.get("filePath", ""))
                if result is None:
                    print(f"    > file: {file_path}", flush=True)
                    continue

                kind = result.kind  # read / write / edit
                ok_str = "ok" if result.ok else str(result.detail.get("error", "failed"))

                if kind == "read":
                    print(f"    > read: {file_path}", flush=True)
                    if result.ok:
                        content = str(result.detail.get("content") or "")
                        print(f"    < ok ({len(content)} chars)", flush=True)
                    else:
                        print(f"    < {ok_str}", flush=True)

                elif kind in ("write", "edit"):
                    size = result.detail.get("bytes", 0)
                    mode = result.detail.get("mode", kind)
                    print(f"    > {kind}: {file_path}", flush=True)
                    if result.ok:
                        print(f"    < ok ({size} bytes, {mode})", flush=True)
                    else:
                        print(f"    < {ok_str}", flush=True)

                else:
                    print(f"    > {kind}: {file_path}", flush=True)
                    print(f"    < {ok_str}", flush=True)

    return _print
