#!/usr/bin/env python3
"""将 OpenCode RL 的运行结果导入到 RD-Agent UI 的 results.csv。

用法:
    python export_to_ui.py                     # 导入所有 runs/
    python export_to_ui.py --run-dir runs/mbpp_20260226_162908  # 导入指定 run
    python export_to_ui.py --dry-run           # 只预览不写入
"""

import argparse
import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path

RESULTS_CSV = Path(os.environ.get(
    "RESULTS_CSV_PATH",
    os.path.expanduser("~/RD-Agent/rdagent/scenarios/rl/autorl_bench/results.csv"),
))
RUNS_DIR = Path(__file__).resolve().parent / "runs"

COLUMNS = [
    "run_id", "timestamp", "task", "agent", "driver_model", "base_model",
    "baseline", "best_score", "improvement", "submissions",
    "duration_s", "success", "workspace",
]


def parse_run_dir(run_path: Path) -> dict | None:
    """解析一个 run 目录，返回 CSV 行 dict。"""
    results_file = run_path / "pipeline_results.json"
    if not results_file.exists():
        return None

    with open(results_file) as f:
        data = json.load(f)

    # 从目录名提取时间戳: task_YYYYMMDD_HHMMSS
    dirname = run_path.name
    m = re.match(r"(.+?)_(\d{8})_(\d{6})", dirname)
    if not m:
        return None

    task = m.group(1)
    date_str = m.group(2)
    time_str = m.group(3)
    ts = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
    run_id = ts.strftime("%Y%m%dT%H%M%S")

    # 检测 driver model: 从 run 目录的环境或用全局 env
    driver_model = _detect_driver_model(run_path)

    # 统计有效提交次数 (exit_code == 0 的迭代)
    iterations = data.get("iterations", [])
    submissions = sum(1 for it in iterations if it.get("exit_code") == 0)

    best_score = data.get("best_score")
    success = data.get("best_iteration", -1) >= 0

    return {
        "run_id": run_id,
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "task": task,
        "agent": "opencode",
        "driver_model": driver_model,
        "base_model": data.get("base_model", ""),
        "baseline": 0.0,
        "best_score": best_score if best_score is not None else "",
        "improvement": best_score if best_score is not None else "",
        "submissions": submissions,
        "duration_s": int(data.get("total_time", 0)),
        "success": success,
        "workspace": str(run_path.resolve()),
    }


def _detect_driver_model(run_path: Path) -> str:
    """尝试从 agent log 或 shell 脚本中检测使用的 LLM 模型。"""
    # 方法1: 从对应的 run_*.sh 中读 OPENCODE_MODEL
    task = run_path.name.split("_")[0]
    script = run_path.parent.parent / f"run_{task}.sh"
    if script.exists():
        text = script.read_text()
        m = re.search(r'OPENCODE_MODEL="([^"]+)"', text)
        if m:
            return m.group(1)

    # 方法2: 环境变量
    return os.environ.get("OPENCODE_MODEL", "unknown")


def get_existing_run_ids(csv_path: Path = RESULTS_CSV) -> set:
    """读取 CSV 中已有的 run_id，避免重复导入。"""
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return {row["run_id"] for row in reader}


def export(run_dirs: list[Path], csv_path: Path, dry_run: bool = False) -> int:
    """导入指定的 run 目录列表，返回新增行数。"""
    existing = get_existing_run_ids(csv_path)
    new_rows = []

    for run_path in sorted(run_dirs):
        row = parse_run_dir(run_path)
        if row is None:
            continue
        if row["run_id"] in existing:
            print(f"  跳过 (已存在): {run_path.name}")
            continue
        new_rows.append(row)

    if not new_rows:
        print("没有新的结果需要导入。")
        return 0

    print(f"\n将导入 {len(new_rows)} 条结果到 {csv_path}:\n")
    for r in new_rows:
        status = "OK" if r["success"] else "FAIL"
        score = r["best_score"] if r["best_score"] != "" else "N/A"
        print(f"  [{status}] {r['task']:10s} {r['timestamp']}  score={score}  model={r['driver_model']}")

    if dry_run:
        print("\n(--dry-run 模式，未实际写入)")
        return len(new_rows)

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        for row in new_rows:
            writer.writerow(row)

    print(f"\n已写入 {len(new_rows)} 条记录。")
    return len(new_rows)


def main():
    parser = argparse.ArgumentParser(description="导出 OpenCode RL 结果到 RD-Agent UI")
    parser.add_argument("--run-dir", type=Path, nargs="*", help="指定 run 目录（默认导入所有 runs/）")
    parser.add_argument("--dry-run", action="store_true", help="只预览，不实际写入")
    parser.add_argument("--csv", type=Path, default=RESULTS_CSV, help="目标 CSV 路径")
    args = parser.parse_args()

    if args.run_dir:
        run_dirs = [Path(d).resolve() for d in args.run_dir]
    else:
        if not RUNS_DIR.exists():
            print(f"runs 目录不存在: {RUNS_DIR}")
            return
        run_dirs = [d for d in RUNS_DIR.iterdir() if d.is_dir()]

    print(f"扫描 {len(run_dirs)} 个 run 目录...")
    export(run_dirs, csv_path=args.csv, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
