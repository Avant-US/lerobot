#!/usr/bin/env python3
"""
compare_openpi_lerobot.py — 端到端对齐验证工具

整合所有验证步骤，一次性检查全部对齐项:
    1. LR Schedule 对齐
    2. Normalization Stats 对齐
    3. 训练配置参数对齐
    4. 数据集完整性

用法:
    python bt/pi05/alig/trainr1/compare_openpi_lerobot.py

    # 含 LR 曲线图
    python bt/pi05/alig/trainr1/compare_openpi_lerobot.py --plot
"""

import argparse
import json
import sys
from pathlib import Path

# 将 lerobot 加入路径
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))


def check_section(name: str):
    """打印检查节标题。"""
    print()
    print(f"  {'─' * 60}")
    print(f"  {name}")
    print(f"  {'─' * 60}")


def verify_config_alignment():
    """验证训练配置参数与 OpenPI 对齐。"""
    check_section("1. 训练配置参数对齐检查")

    # 导入 config
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import (
        ALIGNMENT_CONFIG,
        LR_SCHEDULE_CONFIG,
        OPTIMIZER_CONFIG,
        TRAINING_CONFIG,
    )

    checks = [
        ("optimizer.weight_decay", OPTIMIZER_CONFIG["weight_decay"], 1e-10, "P0"),
        ("optimizer.lr", OPTIMIZER_CONFIG["lr"], 2.5e-5, ""),
        ("optimizer.betas", OPTIMIZER_CONFIG["betas"], (0.9, 0.95), ""),
        ("optimizer.eps", OPTIMIZER_CONFIG["eps"], 1e-8, ""),
        ("optimizer.grad_clip", OPTIMIZER_CONFIG["grad_clip_norm"], 1.0, ""),
        ("lr_schedule.phase_mode", LR_SCHEDULE_CONFIG["phase_mode"], "post_warmup", "P0"),
        ("lr_schedule.warmup_steps", LR_SCHEDULE_CONFIG["warmup_steps"], 1000, ""),
        ("lr_schedule.decay_steps", LR_SCHEDULE_CONFIG["decay_steps"], 30000, ""),
        ("lr_schedule.peak_lr", LR_SCHEDULE_CONFIG["peak_lr"], 2.5e-5, ""),
        ("lr_schedule.decay_lr", LR_SCHEDULE_CONFIG["decay_lr"], 2.5e-6, ""),
        ("training.batch_size", TRAINING_CONFIG["batch_size"], 256, "CLI覆盖"),
        ("training.steps", TRAINING_CONFIG["steps"], 100000, "CLI覆盖"),
        ("training.save_freq", TRAINING_CONFIG["save_freq"], 500, "CLI覆盖"),
        ("training.keep_period", TRAINING_CONFIG["keep_period"], 2500, "CLI覆盖"),
        ("training.num_workers", TRAINING_CONFIG["num_workers"], 2, "CLI覆盖"),
        ("training.seed", TRAINING_CONFIG["seed"], 42, "@#2"),
        ("alignment.ema_decay", ALIGNMENT_CONFIG["ema_decay"], 0.99, "P1"),
        ("alignment.loss_include_padding", ALIGNMENT_CONFIG["loss_include_padding"], True, "P0"),
        ("alignment.augmentation_enabled", ALIGNMENT_CONFIG["augmentation_enabled"], True, "P1"),
    ]

    all_ok = True
    for name, actual, expected, tag in checks:
        ok = actual == expected
        status = "OK" if ok else "MISMATCH"
        tag_str = f" [{tag}]" if tag else ""
        print(f"    {name:40s} = {str(actual):15s}  expected={str(expected):15s}  [{status}]{tag_str}")
        if not ok:
            all_ok = False

    return all_ok


def verify_lr_schedule():
    """验证 LR Schedule 对齐。"""
    check_section("2. LR Schedule 对齐检查")

    from verify_lr_schedule import openpi_lr_schedule, lerobot_lr_schedule

    key_steps = [0, 1000, 5000, 10000, 15000, 20000, 25000, 30000, 50000, 100000]
    max_diff = 0.0

    for step in key_steps:
        openpi = openpi_lr_schedule(step)
        lerobot = lerobot_lr_schedule(step, "post_warmup")
        rel_diff = abs(openpi - lerobot) / max(openpi, 1e-30) * 100
        max_diff = max(max_diff, rel_diff)

    passed = max_diff < 0.1
    status = "PASS" if passed else "FAIL"
    print(f"    最大相对差异: {max_diff:.4f}%  [{status}]")
    return passed


def verify_norm_stats():
    """验证 Normalization Stats 对齐。"""
    check_section("3. Normalization Stats 对齐检查")

    from verify_norm_stats import LEROBOT_STATS, OPENPI_NORM_STATS

    if not OPENPI_NORM_STATS.exists():
        print(f"    SKIP: OpenPI norm_stats 不存在")
        return True

    if not LEROBOT_STATS.exists():
        print(f"    FAIL: LeRobot stats 不存在 (运行 prepare_data.sh)")
        return False

    import numpy as np

    with open(OPENPI_NORM_STATS) as f:
        openpi = json.load(f)
    if "norm_stats" in openpi:
        openpi = openpi["norm_stats"]

    with open(LEROBOT_STATS) as f:
        lerobot = json.load(f)

    key_map = {"state": "observation.state", "actions": "action"}
    all_ok = True

    for src_key, dst_key in key_map.items():
        if src_key not in openpi or dst_key not in lerobot:
            print(f"    SKIP: {src_key} → {dst_key}")
            continue

        for q_key in ["q01", "q99"]:
            if q_key not in openpi[src_key] or q_key not in lerobot[dst_key]:
                continue

            src_val = np.array(openpi[src_key][q_key])
            dst_val = np.array(lerobot[dst_key][q_key])
            max_diff = np.max(np.abs(src_val - dst_val))

            ok = max_diff < 1e-6
            status = "OK" if ok else "MISMATCH"
            print(f"    {dst_key}.{q_key}: max_diff={max_diff:.2e}  [{status}]")
            if not ok:
                all_ok = False

    return all_ok


def verify_dataset():
    """验证数据集完整性。"""
    check_section("4. 数据集完整性检查")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import CONVERTED_DATA_DIR

    if not CONVERTED_DATA_DIR.exists():
        print(f"    FAIL: 数据集目录不存在: {CONVERTED_DATA_DIR}")
        print(f"    请运行: bash bt/pi05/alig/trainr1/prepare_data.sh")
        return False

    info_path = CONVERTED_DATA_DIR / "meta" / "info.json"
    if not info_path.exists():
        print(f"    FAIL: info.json 不存在")
        return False

    with open(info_path) as f:
        info = json.load(f)

    checks = [
        ("codebase_version", info.get("codebase_version"), "v3.0"),
        ("total_episodes", info.get("total_episodes"), 64),
        ("fps", info.get("fps"), 14),
    ]

    all_ok = True
    for name, actual, expected in checks:
        ok = actual == expected
        status = "OK" if ok else "WARN"
        print(f"    {name:20s} = {str(actual):10s}  expected={str(expected):10s}  [{status}]")
        if name == "codebase_version" and not ok:
            all_ok = False

    # 检查必需的 features
    required_features = [
        "observation.images.head_rgb",
        "observation.images.left_wrist_rgb",
        "observation.images.right_wrist_rgb",
        "observation.state",
        "action",
    ]
    features = info.get("features", {})
    for feat in required_features:
        ok = feat in features
        status = "OK" if ok else "MISSING"
        print(f"    feature: {feat:40s}  [{status}]")
        if not ok:
            all_ok = False

    # 检查 stats.json 存在且有分位数
    stats_path = CONVERTED_DATA_DIR / "meta" / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        for key in ["observation.state", "action"]:
            has_q = "q01" in stats.get(key, {}) and "q99" in stats.get(key, {})
            status = "OK" if has_q else "MISSING"
            print(f"    {key:40s} q01/q99  [{status}]")
            if not has_q:
                all_ok = False
    else:
        print(f"    FAIL: stats.json 不存在")
        all_ok = False

    return all_ok


def verify_pretrained_model():
    """检查预训练模型可访问性。"""
    check_section("5. 预训练模型检查")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import PRETRAINED_MODEL_PATH

    print(f"    Model path: {PRETRAINED_MODEL_PATH}")

    try:
        from huggingface_hub import model_info
        info = model_info(PRETRAINED_MODEL_PATH)
        print(f"    HuggingFace Hub: {info.modelId}  [OK]")
        return True
    except Exception as e:
        print(f"    HuggingFace Hub: 无法访问 ({e})")
        print(f"    如果使用本地模型, 请修改 config.py PRETRAINED_MODEL_PATH")
        return False  # non-fatal


def main():
    parser = argparse.ArgumentParser(description="端到端对齐验证")
    parser.add_argument("--plot", action="store_true", help="生成 LR 曲线图")
    args = parser.parse_args()

    print("=" * 70)
    print(" OpenPI pi05_r1pro_chassis 对齐 — 端到端验证")
    print("=" * 70)

    results = {}

    # 1. 配置参数
    results["config"] = verify_config_alignment()

    # 2. LR Schedule
    try:
        results["lr_schedule"] = verify_lr_schedule()
    except Exception as e:
        print(f"    ERROR: {e}")
        results["lr_schedule"] = False

    # 3. Norm Stats
    try:
        results["norm_stats"] = verify_norm_stats()
    except Exception as e:
        print(f"    ERROR: {e}")
        results["norm_stats"] = False

    # 4. 数据集
    try:
        results["dataset"] = verify_dataset()
    except Exception as e:
        print(f"    ERROR: {e}")
        results["dataset"] = False

    # 5. 预训练模型
    try:
        results["pretrained"] = verify_pretrained_model()
    except Exception as e:
        print(f"    INFO: 模型检查跳过 ({e})")
        results["pretrained"] = None

    # 可选: LR 图
    if args.plot:
        try:
            from verify_lr_schedule import verify as lr_verify
            lr_verify(plot=True)
        except Exception:
            pass

    # 总结
    print()
    print("=" * 70)
    print(" 验证总结")
    print("=" * 70)

    all_critical_pass = True
    for name, passed in results.items():
        if passed is None:
            status = "SKIP"
        elif passed:
            status = "PASS"
        else:
            status = "FAIL"
            if name != "pretrained":
                all_critical_pass = False
        print(f"    {name:20s}  [{status}]")

    print()
    if all_critical_pass:
        print("  ALL CRITICAL CHECKS PASSED")
        print("  训练配置已与 OpenPI pi05_r1pro_chassis 对齐")
    else:
        print("  SOME CHECKS FAILED")
        print("  请修复上述问题后再启动训练")
    print("=" * 70)

    sys.exit(0 if all_critical_pass else 1)


if __name__ == "__main__":
    main()
