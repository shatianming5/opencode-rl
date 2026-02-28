"""验证编排器：加载 Agent 写的 verifier.py、运行验证、计算指标、校验和检查。"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import traceback
from pathlib import Path
from types import ModuleType

from .types import VerificationResult, VerifiedSample

# ---------------------------------------------------------------------------
# SHA256 校验和
# ---------------------------------------------------------------------------

def compute_sha256(file_path: str) -> str:
    """计算文件的 SHA256 校验和。"""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_backup_content(backup: str) -> str:
    """Resolve verifier backup: if it's a file path, read it; otherwise return as-is.

    Backward compatible: old checkpoints store inline content, new ones store file paths.
    """
    if not backup:
        return ""
    # If it looks like a file path and exists, read the content
    backup_path = Path(backup)
    if backup_path.suffix == ".py" and backup_path.exists():
        try:
            return backup_path.read_text(encoding="utf-8")
        except OSError:
            return ""
    # Otherwise treat as inline content (backward compat)
    return backup


def check_verifier_integrity(
    verifier_path: str,
    expected_sha256: str,
    backup_content: str,
) -> bool:
    """检查 verifier.py 是否被篡改。若被篡改则恢复备份。

    Args:
        backup_content: Either inline source code (old format) or a file path (new format).

    Returns:
        True 如果完整性检查通过（或成功恢复），False 如果恢复失败。
    """
    resolved_backup = _resolve_backup_content(backup_content)
    path = Path(verifier_path)
    if not path.exists():
        if resolved_backup:
            print(f"  [verifier] WARNING: {verifier_path} missing, restoring from backup")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(resolved_backup, encoding="utf-8")
            return True
        return False

    actual_sha256 = compute_sha256(verifier_path)
    if actual_sha256 == expected_sha256:
        return True

    # 被篡改
    print(f"  [verifier] WARNING: verifier.py has been tampered with!")
    print(f"    Expected SHA256: {expected_sha256}")
    print(f"    Actual SHA256:   {actual_sha256}")

    if resolved_backup:
        print(f"  [verifier] Restoring from backup...")
        path.write_text(resolved_backup, encoding="utf-8")
        restored_sha = compute_sha256(verifier_path)
        if restored_sha == expected_sha256:
            print(f"  [verifier] Restored successfully")
            return True
        else:
            print(f"  [verifier] WARNING: Restored file SHA mismatch")
            return False

    print(f"  [verifier] No backup available, cannot restore")
    return False


# ---------------------------------------------------------------------------
# 加载 verifier 模块
# ---------------------------------------------------------------------------

def _load_verifier_module(verifier_path: str) -> ModuleType:
    """动态加载 verifier.py 为 Python 模块。"""
    spec = importlib.util.spec_from_file_location("verifier", verifier_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {verifier_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_verifier(verifier_path: str, data_path: str) -> tuple[bool, str]:
    """验证 verifier.py 的正确性。

    用 train.jsonl 第一条数据的正确答案调用 verify()，确认返回 passed=True；
    再用空字符串调用，确认返回 passed=False。

    Returns:
        (success, error_message)
    """
    path = Path(verifier_path)
    if not path.exists():
        return False, f"Verifier not found: {verifier_path}"

    # 加载模块
    try:
        module = _load_verifier_module(verifier_path)
    except Exception as e:
        return False, f"Failed to load verifier: {e}"

    if not hasattr(module, "verify"):
        return False, "verifier.py does not have a verify() function"

    # 读取第一条数据
    jsonl_path = Path(data_path)
    if jsonl_path.is_dir():
        jsonl_path = jsonl_path / "train.jsonl"
    if not jsonl_path.exists():
        return False, f"Data file not found: {jsonl_path}"

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            first_line = ""
            for line in f:
                if line.strip():
                    first_line = line.strip()
                    break
        if not first_line:
            return False, "train.jsonl is empty"
        sample = json.loads(first_line)
    except Exception as e:
        return False, f"Failed to read train.jsonl: {e}"

    # 提取正确答案作为 completion
    correct_answer = (
        sample.get("answer")
        or sample.get("response")
        or sample.get("output")
        or sample.get("canonical_solution")
        or ""
    )
    if not correct_answer:
        return False, "Cannot find correct answer in first sample (tried: answer, response, output, canonical_solution)"

    # 测试 1：正确答案应返回 passed=True
    try:
        result = module.verify(correct_answer, sample)
    except Exception as e:
        return False, f"verify() raised exception with correct answer: {e}\n{traceback.format_exc()}"

    if not isinstance(result, dict):
        return False, f"verify() returned {type(result).__name__}, expected dict"
    if not result.get("passed"):
        reason = result.get("reason", "unknown")
        return False, f"verify() returned passed=False for correct answer. Reason: {reason}"

    # 测试 2：空字符串应返回 passed=False
    try:
        result_empty = module.verify("", sample)
    except Exception as e:
        return False, f"verify() raised exception with empty string: {e}"

    if not isinstance(result_empty, dict):
        return False, f"verify() returned {type(result_empty).__name__} for empty string, expected dict"
    if result_empty.get("passed"):
        return False, "verify() returned passed=True for empty string (should be False)"

    return True, ""


# ---------------------------------------------------------------------------
# 批量验证
# ---------------------------------------------------------------------------

def run_verification(
    verifier_path: str,
    samples_path: str,
    data_path: str,
) -> VerificationResult:
    """用锁定的 verifier.py 对 samples.jsonl 进行独立验证。

    Args:
        verifier_path: 锁定的 verifier.py 路径
        samples_path: Agent 生成的 samples.jsonl
        data_path: 原始 train.jsonl（目录或文件路径）

    Returns:
        VerificationResult 包含管线独立验证的分数和交叉验证指标
    """
    result = VerificationResult()

    # 加载 verifier
    try:
        module = _load_verifier_module(verifier_path)
    except Exception as e:
        print(f"  [verify] ERROR: Failed to load verifier: {e}")
        return result

    # 构建原始数据索引（task_id -> sample）
    jsonl_path = Path(data_path)
    if jsonl_path.is_dir():
        jsonl_path = jsonl_path / "train.jsonl"

    data_index: dict[str, dict] = {}
    data_list: list[dict] = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                tid = str(item.get("task_id", len(data_list)))
                data_index[tid] = item
                data_list.append(item)
    except Exception as e:
        print(f"  [verify] ERROR: Failed to read data: {e}")
        return result

    # 读取 samples.jsonl
    samples: list[dict] = []
    try:
        with open(samples_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                samples.append(json.loads(line))
    except Exception as e:
        print(f"  [verify] ERROR: Failed to read samples: {e}")
        return result

    if not samples:
        print(f"  [verify] No samples found in {samples_path}")
        return result

    # 逐样本验证
    verified: list[VerifiedSample] = []
    agent_rewards: list[float] = []
    agree_count = 0

    for i, sample_entry in enumerate(samples):
        completion = sample_entry.get("completion", "")
        agent_reward = float(sample_entry.get("reward", 0.0))
        task_id = str(sample_entry.get("task_id", i))
        agent_rewards.append(agent_reward)

        # 查找对应的原始数据
        original = data_index.get(task_id)
        if original is None and i < len(data_list):
            original = data_list[i]
        if original is None:
            original = {}

        # 调用 verifier
        vs = VerifiedSample(task_id=task_id, agent_reward=agent_reward)
        try:
            vr = module.verify(completion, original)
            if isinstance(vr, dict):
                vs.passed = bool(vr.get("passed", False))
                vs.reward = float(vr.get("reward", 1.0 if vs.passed else 0.0))
                vs.reason = str(vr.get("reason", ""))
            else:
                vs.passed = False
                vs.reason = f"verify() returned {type(vr).__name__}"
        except Exception as e:
            vs.passed = False
            vs.reason = f"verify() error: {e}"

        # 交叉验证：管线判定与 Agent 自报是否一致
        pipeline_pass = vs.passed
        agent_pass = agent_reward >= 1.0
        if pipeline_pass == agent_pass:
            agree_count += 1

        verified.append(vs)

    # 计算汇总指标
    total = len(verified)
    passed = sum(1 for v in verified if v.passed)
    pipeline_score = passed / total if total > 0 else 0.0
    agent_score = sum(1 for r in agent_rewards if r >= 1.0) / total if total > 0 else 0.0
    agreement_rate = agree_count / total if total > 0 else 0.0
    inflation = agent_score - pipeline_score

    result = VerificationResult(
        total=total,
        passed=passed,
        pipeline_score=pipeline_score,
        agent_score=agent_score,
        reward_agreement_rate=agreement_rate,
        reward_inflation=inflation,
        samples=verified,
    )

    # 打印交叉验证报告
    print(f"\n  管线独立验证结果:")
    print(f"    总样本数: {total}")
    print(f"    通过（管线验证）: {passed}/{total} = {pipeline_score:.4f}")
    print(f"    Agent 自报分数: {agent_score:.4f}")
    print(f"    Reward 一致率: {agreement_rate * 100:.2f}%")
    print(f"    Reward 注水量: {inflation:+.4f}")
    if inflation > 0.05:
        print(f"    警告: Agent reward 注水 {inflation:.4f}")

    return result
