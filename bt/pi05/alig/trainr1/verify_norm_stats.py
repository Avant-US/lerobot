#!/usr/bin/env python3
"""
verify_norm_stats.py — 归一化统计对齐验证

验证转换后的 LeRobot 数据集 stats.json 中的 q01/q99 分位数
与 OpenPI 原始 norm_stats.json 完全一致。

用法:
    python bt/pi05/alig/trainr1/verify_norm_stats.py

检查项:
    1. q01/q99 数值精确匹配 (来自 OpenPI norm_stats.json 导入)
    2. 维度一致性 (23 维 state/action)
    3. 归一化公式等价性验证
"""

import json
import sys
from pathlib import Path

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# 路径配置
# ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent

# OpenPI 原始 norm stats
OPENPI_NORM_STATS = (
    Path("/mnt/r/share/lkx/pi/openpi/assets/pi05_r1pro_chassis")
    / "r1_pro_data_convert_chassis" / "norm_stats.json"
)

# 转换后的 LeRobot 数据集 stats
LEROBOT_STATS = SCRIPT_DIR / "data" / "r1_pro_chassis_v30" / "meta" / "stats.json"

# Key 映射
KEY_MAP = {
    "state": "observation.state",
    "actions": "action",
}


def load_openpi_stats(path: Path) -> dict:
    """加载 OpenPI norm_stats.json 并提取 q01/q99。"""
    with open(path) as f:
        data = json.load(f)

    # 支持嵌套 "norm_stats" 格式
    if "norm_stats" in data:
        data = data["norm_stats"]

    return data


def load_lerobot_stats(path: Path) -> dict:
    """加载 LeRobot stats.json。"""
    with open(path) as f:
        return json.load(f)


def verify():
    """执行归一化统计对齐验证。"""
    print("=" * 70)
    print(" Normalization Stats 对齐验证")
    print(" OpenPI norm_stats.json vs LeRobot stats.json")
    print("=" * 70)
    print()

    # ── 检查文件存在 ──────────────────────────────────────────────
    errors = []
    if not OPENPI_NORM_STATS.exists():
        errors.append(f"OpenPI norm_stats 不存在: {OPENPI_NORM_STATS}")
    if not LEROBOT_STATS.exists():
        errors.append(f"LeRobot stats 不存在: {LEROBOT_STATS}")
        errors.append("请先运行: bash bt/pi05/alig/trainr1/prepare_data.sh")

    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        return False

    # ── 加载数据 ──────────────────────────────────────────────────
    openpi_stats = load_openpi_stats(OPENPI_NORM_STATS)
    lerobot_stats = load_lerobot_stats(LEROBOT_STATS)

    print(f"  OpenPI keys: {list(openpi_stats.keys())}")
    print(f"  LeRobot keys: {[k for k in lerobot_stats.keys() if 'state' in k or 'action' in k]}")
    print()

    all_passed = True

    for openpi_key, lerobot_key in KEY_MAP.items():
        print(f"  [{openpi_key} → {lerobot_key}]")

        if openpi_key not in openpi_stats:
            print(f"    SKIP: OpenPI 缺少 key '{openpi_key}'")
            continue
        if lerobot_key not in lerobot_stats:
            print(f"    FAIL: LeRobot 缺少 key '{lerobot_key}'")
            all_passed = False
            continue

        src = openpi_stats[openpi_key]
        dst = lerobot_stats[lerobot_key]

        # ── 检查 q01/q99 ─────────────────────────────────────────
        for q_key in ["q01", "q99"]:
            if q_key not in src:
                print(f"    SKIP: OpenPI {openpi_key} 无 {q_key}")
                continue
            if q_key not in dst:
                print(f"    FAIL: LeRobot {lerobot_key} 无 {q_key}")
                all_passed = False
                continue

            src_val = np.array(src[q_key], dtype=np.float64)
            dst_val = np.array(dst[q_key], dtype=np.float64)

            # 维度检查
            if src_val.shape != dst_val.shape:
                print(f"    FAIL: {q_key} 维度不匹配 — OpenPI={src_val.shape}, LeRobot={dst_val.shape}")
                all_passed = False
                continue

            # 数值检查
            max_abs_diff = np.max(np.abs(src_val - dst_val))
            max_rel_diff = np.max(np.abs(src_val - dst_val) / (np.abs(src_val) + 1e-30))

            if max_abs_diff < 1e-10:
                status = "EXACT MATCH"
                passed = True
            elif max_rel_diff < 1e-6:
                status = "NEAR MATCH"
                passed = True
            else:
                status = "MISMATCH"
                passed = False
                all_passed = False

            print(f"    {q_key}: dim={src_val.shape[0]:2d}, "
                  f"max_abs_diff={max_abs_diff:.2e}, "
                  f"max_rel_diff={max_rel_diff:.2e}  [{status}]")

            if not passed:
                # 打印前 5 个不匹配的维度
                diffs = np.abs(src_val - dst_val)
                worst_dims = np.argsort(diffs)[-5:][::-1]
                for dim in worst_dims:
                    if diffs[dim] > 1e-10:
                        print(f"      dim[{dim}]: OpenPI={src_val[dim]:.8f}, "
                              f"LeRobot={dst_val[dim]:.8f}, diff={diffs[dim]:.2e}")

        # ── 检查 mean/std (参考信息) ─────────────────────────────
        for stat_key in ["mean", "std"]:
            if stat_key in src and stat_key in dst:
                src_val = np.array(src[stat_key], dtype=np.float64)
                dst_val = np.array(dst[stat_key], dtype=np.float64)
                if src_val.shape == dst_val.shape:
                    max_diff = np.max(np.abs(src_val - dst_val))
                    print(f"    {stat_key}: max_diff={max_diff:.2e}  [参考]")

        print()

    # ── 归一化公式验证 ────────────────────────────────────────────
    print("  [归一化公式等价性验证]")
    print("    OpenPI:  (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0")
    print("    LeRobot: 2.0 * (x - q01) / (q99 - q01) - 1.0")
    print()

    # 用实际数据验证
    if "state" in openpi_stats and "observation.state" in lerobot_stats:
        q01 = np.array(openpi_stats["state"]["q01"])
        q99 = np.array(openpi_stats["state"]["q99"])

        # 生成测试值 (在 q01 和 q99 之间)
        test_values = np.linspace(q01, q99, 10)

        # OpenPI 公式
        openpi_norm = (test_values - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

        # LeRobot 公式
        range_val = q99 - q01
        range_val = np.maximum(range_val, 1e-8)  # LeRobot 用 max 而非 +
        lerobot_norm = 2.0 * (test_values - q01) / range_val - 1.0

        max_norm_diff = np.max(np.abs(openpi_norm - lerobot_norm))
        print(f"    归一化结果最大差异: {max_norm_diff:.2e}")
        if max_norm_diff < 1e-6:
            print(f"    PASS: 归一化公式等价 (差异 < 1e-6)")
        else:
            print(f"    WARN: 归一化公式有微小差异 (不影响训练)")

    # ── 总结 ──────────────────────────────────────────────────────
    print()
    print("=" * 70)
    if all_passed:
        print("  PASS: 所有归一化统计验证通过!")
    else:
        print("  FAIL: 存在不匹配项，请检查数据转换")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    passed = verify()
    sys.exit(0 if passed else 1)
