#!/usr/bin/env python3
"""
verify_training.py — 训练曲线对比验证

从 wandb 或 CSV 日志加载 OpenPI 和 LeRobot 的训练曲线，
对比 loss、grad_norm、lr 等关键指标。

用法:
    # 从 wandb 加载
    python bt/pi05/alig/trainr1/verify_training.py \
        --openpi-wandb "entity/project/run_id" \
        --lerobot-wandb "entity/project/run_id"

    # 从 CSV 文件加载
    python bt/pi05/alig/trainr1/verify_training.py \
        --openpi-csv path/to/openpi_metrics.csv \
        --lerobot-csv path/to/lerobot_metrics.csv

    # 仅分析 LeRobot 训练日志
    python bt/pi05/alig/trainr1/verify_training.py \
        --lerobot-dir bt/pi05/alig/trainr1/outputs/r1pro_chassis_aligned

输出:
    - 关键步数处的指标对比表
    - 训练曲线对比图 (PNG)
    - 对齐质量评估 (L1/L2 等价性判断)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_lerobot_training_log(output_dir: Path) -> dict:
    """从 LeRobot 训练输出目录加载指标。"""
    # LeRobot 将 wandb 日志保存在 wandb/ 子目录
    # 也可以从 train_config.json 读取配置确认参数
    metrics = {"step": [], "loss": [], "lr": [], "grad_norm": []}

    # 尝试从 wandb 本地文件加载
    wandb_dir = output_dir / "wandb"
    if wandb_dir.exists():
        # 查找最新的 run 目录
        run_dirs = sorted(wandb_dir.glob("run-*"))
        if run_dirs:
            run_dir = run_dirs[-1]
            history_file = run_dir / "files" / "wandb-history.jsonl"
            if history_file.exists():
                with open(history_file) as f:
                    for line in f:
                        entry = json.loads(line)
                        if "loss" in entry and "_step" in entry:
                            metrics["step"].append(entry["_step"])
                            metrics["loss"].append(entry.get("loss", float("nan")))
                            metrics["lr"].append(entry.get("lr", float("nan")))
                            metrics["grad_norm"].append(entry.get("grad_norm", float("nan")))
                print(f"  从 wandb 本地历史加载 {len(metrics['step'])} 条记录")
                return metrics

    print(f"  WARNING: 无法从 {output_dir} 加载训练日志")
    return metrics


def load_wandb_run(run_path: str) -> dict:
    """从 wandb API 加载训练指标。"""
    try:
        import wandb

        api = wandb.Api()
        run = api.run(run_path)

        metrics = {"step": [], "loss": [], "lr": [], "grad_norm": []}
        for row in run.scan_history(keys=["loss", "lr", "grad_norm"]):
            metrics["step"].append(row.get("_step", 0))
            metrics["loss"].append(row.get("loss", float("nan")))
            metrics["lr"].append(row.get("lr", float("nan")))
            metrics["grad_norm"].append(row.get("grad_norm", float("nan")))

        print(f"  从 wandb 加载 {len(metrics['step'])} 条记录: {run_path}")
        return metrics
    except Exception as e:
        print(f"  ERROR: 无法从 wandb 加载 ({e})")
        return {"step": [], "loss": [], "lr": [], "grad_norm": []}


def load_csv(path: Path) -> dict:
    """从 CSV 加载指标。"""
    import csv

    metrics = {"step": [], "loss": [], "lr": [], "grad_norm": []}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics["step"].append(int(row.get("step", row.get("_step", 0))))
            metrics["loss"].append(float(row.get("loss", "nan")))
            metrics["lr"].append(float(row.get("lr", "nan")))
            metrics["grad_norm"].append(float(row.get("grad_norm", "nan")))

    print(f"  从 CSV 加载 {len(metrics['step'])} 条记录: {path}")
    return metrics


def compare_metrics(openpi: dict, lerobot: dict, label: str = "loss"):
    """对比两组训练指标。"""
    if not openpi["step"] or not lerobot["step"]:
        print(f"  SKIP: 数据不足，无法对比 {label}")
        return

    openpi_steps = np.array(openpi["step"])
    openpi_vals = np.array(openpi[label])
    lerobot_steps = np.array(lerobot["step"])
    lerobot_vals = np.array(lerobot[label])

    # 在共同步数处对比
    key_steps = [100, 500, 1000, 5000, 10000, 15000, 20000, 25000, 30000,
                 35000, 50000, 75000, 100000]

    print(f"\n  [{label} 对比]")
    print(f"  {'Step':>6s}  {'OpenPI':>12s}  {'LeRobot':>12s}  {'相对差异':>10s}")
    print("  " + "-" * 48)

    diffs = []
    for step in key_steps:
        # 找最近的步数
        openpi_idx = np.argmin(np.abs(openpi_steps - step))
        lerobot_idx = np.argmin(np.abs(lerobot_steps - step))

        if abs(openpi_steps[openpi_idx] - step) > 50 or abs(lerobot_steps[lerobot_idx] - step) > 50:
            continue

        o_val = openpi_vals[openpi_idx]
        l_val = lerobot_vals[lerobot_idx]
        rel_diff = abs(o_val - l_val) / max(abs(o_val), 1e-30) * 100
        diffs.append(rel_diff)

        print(f"  {step:6d}  {o_val:12.6e}  {l_val:12.6e}  {rel_diff:8.2f}%")

    if diffs:
        print(f"\n  平均相对差异: {np.mean(diffs):.2f}%")
        print(f"  最大相对差异: {np.max(diffs):.2f}%")


def plot_comparison(openpi: dict, lerobot: dict, output_path: Path):
    """绘制训练曲线对比图。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  WARNING: matplotlib 未安装，跳过绘图")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    metrics = [
        ("loss", "Training Loss", axes[0]),
        ("lr", "Learning Rate", axes[1]),
        ("grad_norm", "Gradient Norm", axes[2]),
    ]

    for label, title, ax in metrics:
        if openpi["step"] and openpi[label]:
            ax.plot(openpi["step"], openpi[label], "b-", label="OpenPI", alpha=0.7, linewidth=1)
        if lerobot["step"] and lerobot[label]:
            ax.plot(lerobot["step"], lerobot[label], "r-", label="LeRobot", alpha=0.7, linewidth=1)
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Step")
    plt.suptitle("OpenPI vs LeRobot Training Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\n  对比图已保存: {output_path}")
    plt.close()


def assess_alignment(openpi: dict, lerobot: dict):
    """评估对齐质量等级。"""
    print("\n" + "=" * 70)
    print(" 对齐质量评估")
    print("=" * 70)

    if not openpi["step"] or not lerobot["step"]:
        print("  无法评估 — 数据不足")
        return

    # 在最后 1000 步对比 loss
    openpi_steps = np.array(openpi["step"])
    openpi_loss = np.array(openpi["loss"])
    lerobot_steps = np.array(lerobot["step"])
    lerobot_loss = np.array(lerobot["loss"])

    # 取最后 10% 步数的平均 loss
    final_fraction = 0.1
    openpi_final_mask = openpi_steps >= openpi_steps[-1] * (1 - final_fraction)
    lerobot_final_mask = lerobot_steps >= lerobot_steps[-1] * (1 - final_fraction)

    if np.any(openpi_final_mask) and np.any(lerobot_final_mask):
        openpi_final_loss = np.mean(openpi_loss[openpi_final_mask])
        lerobot_final_loss = np.mean(lerobot_loss[lerobot_final_mask])
        rel_diff = abs(openpi_final_loss - lerobot_final_loss) / max(openpi_final_loss, 1e-30) * 100

        print(f"\n  OpenPI 最终 loss (后 10%): {openpi_final_loss:.6f}")
        print(f"  LeRobot 最终 loss (后 10%): {lerobot_final_loss:.6f}")
        print(f"  相对差异: {rel_diff:.2f}%")

        if rel_diff < 5:
            print(f"\n  L1 等价: PASS (差异 < 5%)")
        elif rel_diff < 20:
            print(f"\n  L1 等价: PARTIAL (差异 < 20%, 可能需要调整)")
        else:
            print(f"\n  L1 等价: FAIL (差异 >= 20%)")

    # 检查 loss 趋势
    if len(lerobot_loss) > 10:
        early_loss = np.mean(lerobot_loss[:5])
        late_loss = np.mean(lerobot_loss[-5:])
        if late_loss < early_loss:
            print(f"  Loss 趋势: 下降 ({early_loss:.6f} → {late_loss:.6f}) OK")
        else:
            print(f"  Loss 趋势: 未下降 ({early_loss:.6f} → {late_loss:.6f}) WARNING")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="训练曲线对比验证")
    parser.add_argument("--openpi-wandb", help="OpenPI wandb run path (entity/project/run_id)")
    parser.add_argument("--lerobot-wandb", help="LeRobot wandb run path")
    parser.add_argument("--openpi-csv", type=Path, help="OpenPI metrics CSV file")
    parser.add_argument("--lerobot-csv", type=Path, help="LeRobot metrics CSV file")
    parser.add_argument("--lerobot-dir", type=Path, help="LeRobot training output directory")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "training_comparison.png")
    args = parser.parse_args()

    # 加载 OpenPI 数据
    openpi = {"step": [], "loss": [], "lr": [], "grad_norm": []}
    if args.openpi_wandb:
        openpi = load_wandb_run(args.openpi_wandb)
    elif args.openpi_csv:
        openpi = load_csv(args.openpi_csv)

    # 加载 LeRobot 数据
    lerobot = {"step": [], "loss": [], "lr": [], "grad_norm": []}
    if args.lerobot_wandb:
        lerobot = load_wandb_run(args.lerobot_wandb)
    elif args.lerobot_csv:
        lerobot = load_csv(args.lerobot_csv)
    elif args.lerobot_dir:
        lerobot = load_lerobot_training_log(args.lerobot_dir)

    # 对比
    if openpi["step"] and lerobot["step"]:
        compare_metrics(openpi, lerobot, "loss")
        compare_metrics(openpi, lerobot, "lr")
        compare_metrics(openpi, lerobot, "grad_norm")
        plot_comparison(openpi, lerobot, args.output)
        assess_alignment(openpi, lerobot)
    elif lerobot["step"]:
        print("\n  仅有 LeRobot 数据，展示训练统计:")
        steps = np.array(lerobot["step"])
        loss = np.array(lerobot["loss"])
        print(f"    总步数: {len(steps)}")
        print(f"    初始 loss: {loss[0]:.6f}")
        print(f"    最终 loss: {loss[-1]:.6f}")
        print(f"    最小 loss: {np.nanmin(loss):.6f}")
    else:
        print("\n  无训练数据可用。请提供 --openpi-wandb/csv 或 --lerobot-wandb/csv/dir 参数")


if __name__ == "__main__":
    main()
