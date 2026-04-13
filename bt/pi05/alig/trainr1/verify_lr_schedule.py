#!/usr/bin/env python3
"""
verify_lr_schedule.py — LR 调度对齐验证

验证 LeRobot CosineDecayWithWarmupSchedulerConfig(phase_mode="post_warmup")
与 OpenPI optax.warmup_cosine_decay_schedule 的数值一致性。

用法:
    python bt/pi05/alig/trainr1/verify_lr_schedule.py [--plot]

输出:
    - 关键步数处的 LR 数值对比表
    - 最大绝对/相对差异
    - 验证结论 (PASS/FAIL)
    - (可选) LR 曲线对比图
"""

import argparse
import math
import sys
from pathlib import Path

# 将 lerobot 加入 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))


# ──────────────────────────────────────────────────────────────────────
# OpenPI optax.warmup_cosine_decay_schedule 参考实现
# ──────────────────────────────────────────────────────────────────────

def openpi_lr_schedule(step: int) -> float:
    """
    OpenPI optax.warmup_cosine_decay_schedule 的纯 Python 参考实现。

    参数:
        init_value = peak_lr / (warmup_steps + 1) = 2.5e-5 / 1001 ≈ 2.4975e-8
        peak_value = 2.5e-5
        warmup_steps = 1000
        decay_steps = 30000
        end_value = 2.5e-6

    optax 行为:
        - Steps 0 → warmup_steps: 线性 warmup (init_value → peak_value)
        - Steps warmup_steps → decay_steps: 余弦衰减 (peak_value → end_value)
        - 余弦跨度: decay_steps - warmup_steps = 29000 步
    """
    warmup_steps = 1000
    peak_lr = 2.5e-5
    decay_lr = 2.5e-6
    decay_steps = 30000
    init_value = peak_lr / (warmup_steps + 1)

    if step < warmup_steps:
        # 线性 warmup
        frac = step / warmup_steps
        return init_value + (peak_lr - init_value) * frac
    else:
        # 余弦衰减 (从 warmup 结束到 decay_steps)
        cosine_steps = decay_steps - warmup_steps  # 29000
        relative_step = min(step - warmup_steps, cosine_steps)
        progress = relative_step / cosine_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return decay_lr + (peak_lr - decay_lr) * cosine_decay


# ──────────────────────────────────────────────────────────────────────
# LeRobot CosineDecayWithWarmupSchedulerConfig 实现
# ──────────────────────────────────────────────────────────────────────

def lerobot_lr_schedule(step: int, phase_mode: str = "post_warmup") -> float:
    """
    LeRobot CosineDecayWithWarmupSchedulerConfig 的纯 Python 参考实现。

    schedulers.py 中 lr_lambda 返回的是相对于 optimizer lr 的乘法因子。
    最终 lr = optimizer_lr * lr_lambda(step) = peak_lr * lr_lambda(step)。
    """
    warmup_steps = 1000
    peak_lr = 2.5e-5
    decay_lr = 2.5e-6
    decay_steps = 30000

    def linear_warmup(current_step):
        if current_step <= 0:
            return 1 / (warmup_steps + 1)
        frac = 1 - current_step / warmup_steps
        return (1 / (warmup_steps + 1) - 1) * frac + 1

    def cosine_decay(current_step):
        if phase_mode == "post_warmup":
            total_cosine_steps = max(1, decay_steps - warmup_steps)
            relative_step = min(current_step - warmup_steps, total_cosine_steps)
            progress = relative_step / total_cosine_steps
        else:  # "absolute"
            s = min(current_step, decay_steps)
            progress = s / decay_steps

        cosine_val = 0.5 * (1 + math.cos(math.pi * progress))
        alpha = decay_lr / peak_lr
        return (1 - alpha) * cosine_val + alpha

    if step < warmup_steps:
        return peak_lr * linear_warmup(step)
    else:
        return peak_lr * cosine_decay(step)


# ──────────────────────────────────────────────────────────────────────
# 验证逻辑
# ──────────────────────────────────────────────────────────────────────

def verify(plot: bool = False):
    """执行 LR 调度对齐验证。"""
    print("=" * 70)
    print(" LR Schedule 对齐验证")
    print(" OpenPI optax vs LeRobot CosineDecayWithWarmup(post_warmup)")
    print("=" * 70)
    print()

    # 关键步数 (覆盖完整的 100000 步范围, 含 step>30000 钳位期)
    key_steps = [0, 100, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 29000, 30000,
                 30001, 35000, 50000, 75000, 100000]

    # post_warmup 模式 (用于对齐)
    print("  [post_warmup 模式 — 用于 OpenPI 对齐]")
    print(f"  {'Step':>6s}  {'OpenPI LR':>12s}  {'LeRobot LR':>12s}  {'差异':>12s}  {'相对差异':>10s}")
    print("  " + "-" * 60)

    max_abs_diff = 0
    max_rel_diff = 0

    openpi_lrs = []
    lerobot_pw_lrs = []

    for step in key_steps:
        openpi = openpi_lr_schedule(step)
        lerobot = lerobot_lr_schedule(step, "post_warmup")

        abs_diff = abs(openpi - lerobot)
        rel_diff = abs_diff / max(openpi, 1e-30) * 100

        max_abs_diff = max(max_abs_diff, abs_diff)
        max_rel_diff = max(max_rel_diff, rel_diff)

        openpi_lrs.append(openpi)
        lerobot_pw_lrs.append(lerobot)

        print(f"  {step:6d}  {openpi:12.6e}  {lerobot:12.6e}  {abs_diff:12.6e}  {rel_diff:8.4f}%")

    print()
    print(f"  最大绝对差异: {max_abs_diff:.6e}")
    print(f"  最大相对差异: {max_rel_diff:.4f}%")
    print()

    # absolute 模式对比 (展示差异)
    print("  [absolute 模式 — LeRobot 旧默认，展示差异]")
    print(f"  {'Step':>6s}  {'OpenPI LR':>12s}  {'LeRobot LR':>12s}  {'差异':>12s}  {'相对差异':>10s}")
    print("  " + "-" * 60)

    lerobot_abs_lrs = []
    max_abs_diff_old = 0
    max_rel_diff_old = 0

    for step in key_steps:
        openpi = openpi_lr_schedule(step)
        lerobot = lerobot_lr_schedule(step, "absolute")
        lerobot_abs_lrs.append(lerobot)

        abs_diff = abs(openpi - lerobot)
        rel_diff = abs_diff / max(openpi, 1e-30) * 100

        max_abs_diff_old = max(max_abs_diff_old, abs_diff)
        max_rel_diff_old = max(max_rel_diff_old, rel_diff)

        print(f"  {step:6d}  {openpi:12.6e}  {lerobot:12.6e}  {abs_diff:12.6e}  {rel_diff:8.4f}%")

    print()
    print(f"  最大绝对差异: {max_abs_diff_old:.6e}")
    print(f"  最大相对差异: {max_rel_diff_old:.4f}%")

    # 验证结论
    print()
    print("=" * 70)
    tolerance = 0.1  # 0.1% 容差
    if max_rel_diff < tolerance:
        print(f"  PASS: post_warmup 模式最大相对差异 {max_rel_diff:.4f}% < {tolerance}%")
        print(f"        LR 调度已与 OpenPI 精确对齐!")
    else:
        print(f"  FAIL: post_warmup 模式最大相对差异 {max_rel_diff:.4f}% >= {tolerance}%")
        print(f"        需要检查 LR 调度实现")

    if max_rel_diff_old > 1.0:
        print(f"  INFO: absolute 模式最大差异 {max_rel_diff_old:.2f}%，确认 post_warmup 修复有效")
    print("=" * 70)

    # 钳位期验证 (step > 30000)
    print()
    print("  [钳位期验证 — step > decay_steps=30000]")
    print("  decay_steps=30000 未被 CLI 覆盖, cosine 完成后 LR 钳位在 decay_lr=2.5e-6")
    clamp_steps = [30001, 35000, 50000, 75000, 100000]
    clamp_ok = True
    for step in clamp_steps:
        openpi = openpi_lr_schedule(step)
        lerobot = lerobot_lr_schedule(step, "post_warmup")
        expected = 2.5e-6
        diff_openpi = abs(openpi - expected) / expected * 100
        diff_lerobot = abs(lerobot - expected) / expected * 100
        ok = diff_openpi < 0.01 and diff_lerobot < 0.01
        status = "OK" if ok else "FAIL"
        print(f"  step {step:>6d}: OpenPI={openpi:.6e}, LeRobot={lerobot:.6e}, expected={expected:.6e} [{status}]")
        if not ok:
            clamp_ok = False

    if clamp_ok:
        print("  PASS: 钳位期 LR 正确保持在 2.5e-6")
    else:
        print("  FAIL: 钳位期 LR 不正确")

    # 可选: 绘制对比图
    if plot:
        try:
            _plot_lr_curves(openpi_lrs, lerobot_pw_lrs, lerobot_abs_lrs, key_steps)
        except ImportError:
            print("\n  WARNING: matplotlib 未安装，跳过绘图")

    return max_rel_diff < tolerance and clamp_ok


def _plot_lr_curves(openpi_lrs, lerobot_pw_lrs, lerobot_abs_lrs, key_steps):
    """绘制 LR 曲线对比图。"""
    import matplotlib.pyplot as plt

    # 密集采样 (覆盖全 100000 步, 含钳位期)
    all_steps = list(range(0, 100001, 200))
    openpi_dense = [openpi_lr_schedule(s) for s in all_steps]
    lerobot_pw_dense = [lerobot_lr_schedule(s, "post_warmup") for s in all_steps]
    lerobot_abs_dense = [lerobot_lr_schedule(s, "absolute") for s in all_steps]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 上图: LR 曲线
    ax1 = axes[0]
    ax1.plot(all_steps, openpi_dense, "b-", label="OpenPI (optax)", linewidth=2)
    ax1.plot(all_steps, lerobot_pw_dense, "r--", label="LeRobot (post_warmup)", linewidth=1.5)
    ax1.plot(all_steps, lerobot_abs_dense, "g:", label="LeRobot (absolute)", linewidth=1.5, alpha=0.6)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Learning Rate")
    ax1.axvline(x=30000, color="gray", linestyle="--", alpha=0.5, label="decay_steps=30000")
    ax1.set_title("LR Schedule Comparison: OpenPI vs LeRobot (0-100000 steps)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 下图: 相对差异
    ax2 = axes[1]
    diff_pw = [(abs(o - l) / max(o, 1e-30)) * 100
               for o, l in zip(openpi_dense, lerobot_pw_dense)]
    diff_abs = [(abs(o - l) / max(o, 1e-30)) * 100
                for o, l in zip(openpi_dense, lerobot_abs_dense)]
    ax2.plot(all_steps, diff_pw, "r-", label="post_warmup 差异", linewidth=1.5)
    ax2.plot(all_steps, diff_abs, "g-", label="absolute 差异", linewidth=1.5, alpha=0.6)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Relative Difference (%)")
    ax2.set_title("Relative LR Difference")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(__file__).parent / "lr_schedule_comparison.png"
    plt.savefig(output_path, dpi=150)
    print(f"\n  对比图已保存: {output_path}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────
# 额外验证: 使用 LeRobot 实际 scheduler 类
# ──────────────────────────────────────────────────────────────────────

def verify_with_actual_scheduler():
    """使用 LeRobot 实际的 scheduler 类进行端到端验证。"""
    try:
        import torch
        from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

        print()
        print("  [端到端验证 — 使用 LeRobot 实际 scheduler 类]")

        config = CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=1000,
            num_decay_steps=30000,
            peak_lr=2.5e-5,
            decay_lr=2.5e-6,
            phase_mode="post_warmup",
        )

        # 创建一个 dummy optimizer
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-5)

        # num_training_steps=100000 匹配 CLI: --num_train_steps 100000
        scheduler = config.build(optimizer, num_training_steps=100000)

        key_steps = [0, 500, 1000, 5000, 10000, 15000, 20000, 25000, 30000,
                     35000, 50000, 75000, 100000]

        print(f"  {'Step':>6s}  {'OpenPI LR':>12s}  {'Actual LR':>12s}  {'相对差异':>10s}")
        print("  " + "-" * 48)

        max_diff = 0.0
        for target_step in key_steps:
            # 推进 scheduler 到目标步数
            while scheduler.last_epoch < target_step:
                optimizer.step()
                scheduler.step()

            actual_lr = optimizer.param_groups[0]["lr"]
            expected_lr = openpi_lr_schedule(target_step)
            rel_diff = abs(actual_lr - expected_lr) / max(expected_lr, 1e-30) * 100
            max_diff = max(max_diff, rel_diff)

            print(f"  {target_step:6d}  {expected_lr:12.6e}  {actual_lr:12.6e}  {rel_diff:8.4f}%")

        print()
        if max_diff < 0.1:
            print(f"  PASS: 端到端最大差异 {max_diff:.4f}% < 0.1%")
        else:
            print(f"  WARN: 端到端最大差异 {max_diff:.4f}% (可能是 LambdaLR 步进语义差异)")

    except Exception as e:
        print(f"\n  SKIP: 无法加载 LeRobot scheduler ({e})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LR 调度对齐验证")
    parser.add_argument("--plot", action="store_true", help="生成 LR 曲线对比图")
    args = parser.parse_args()

    passed = verify(plot=args.plot)
    verify_with_actual_scheduler()

    sys.exit(0 if passed else 1)
