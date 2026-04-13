#!/usr/bin/env python3
"""
checkpoint 清理工具 — 实现 OpenPI Orbax keep_period 逻辑

OpenPI 使用 Orbax CheckpointManager(max_to_keep=1, keep_period=N):
  - 保留所有 step % keep_period == 0 的 milestone checkpoint
  - 保留最新的 1 个 checkpoint
  - 删除其余中间 checkpoint

LeRobot 没有内置 keep_period 机制，所有 save_freq 间隔的 checkpoint 都会保留。
本脚本在训练后（或训练中）清理多余的 checkpoint。

用法:
    # 预览要删除的 checkpoint（不实际删除）
    python cleanup_checkpoints.py --checkpoint-dir outputs/checkpoints --keep-period 2500 --dry-run

    # 执行清理
    python cleanup_checkpoints.py --checkpoint-dir outputs/checkpoints --keep-period 2500

    # 作为模块调用
    from cleanup_checkpoints import cleanup_checkpoints
    kept, deleted = cleanup_checkpoints(Path("outputs/checkpoints"), keep_period=2500)
"""

import argparse
import re
import shutil
import sys
from pathlib import Path


def cleanup_checkpoints(
    checkpoint_base_dir: Path,
    keep_period: int = 2500,
    dry_run: bool = False,
) -> tuple[list[int], list[int]]:
    """
    清理 checkpoint 目录，保留 milestone 和最新 checkpoint。

    Args:
        checkpoint_base_dir: checkpoints/ 目录路径（包含 000500/, 001000/ 等子目录）
        keep_period: 保留 step % keep_period == 0 的 checkpoint
        dry_run: 如果为 True，仅打印要删除的内容，不实际删除

    Returns:
        (kept_steps, deleted_steps): 保留和删除的 step 列表
    """
    if not checkpoint_base_dir.is_dir():
        print(f"WARNING: checkpoint 目录不存在: {checkpoint_base_dir}")
        return [], []

    # 扫描所有 checkpoint 子目录（格式: 数字命名，如 000500, 001000）
    step_dirs: dict[int, Path] = {}
    for d in checkpoint_base_dir.iterdir():
        if d.is_dir() and re.match(r"^\d+$", d.name):
            step = int(d.name)
            step_dirs[step] = d

    if not step_dirs:
        print("INFO: 未找到任何 checkpoint 目录")
        return [], []

    sorted_steps = sorted(step_dirs.keys())
    latest_step = sorted_steps[-1]

    kept_steps = []
    deleted_steps = []

    for step in sorted_steps:
        is_milestone = step % keep_period == 0
        is_latest = step == latest_step

        if is_milestone or is_latest:
            kept_steps.append(step)
            reason = []
            if is_milestone:
                reason.append(f"milestone (step%{keep_period}==0)")
            if is_latest:
                reason.append("latest")
            if not dry_run:
                print(f"  KEEP   step {step:>8d}  [{', '.join(reason)}]")
        else:
            deleted_steps.append(step)
            if dry_run:
                print(f"  DELETE step {step:>8d}  (would delete)")
            else:
                print(f"  DELETE step {step:>8d}  (deleting...)")
                shutil.rmtree(step_dirs[step])

    return kept_steps, deleted_steps


def main():
    parser = argparse.ArgumentParser(
        description="清理 checkpoint，实现 OpenPI keep_period 逻辑"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="checkpoints/ 目录路径",
    )
    parser.add_argument(
        "--keep-period",
        type=int,
        default=2500,
        help="保留 step %% keep_period == 0 的 checkpoint (默认: 2500)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅预览，不实际删除",
    )
    args = parser.parse_args()

    print(f"Checkpoint 目录: {args.checkpoint_dir}")
    print(f"Keep Period:      {args.keep_period}")
    print(f"Dry Run:          {args.dry_run}")
    print()

    kept, deleted = cleanup_checkpoints(
        args.checkpoint_dir,
        keep_period=args.keep_period,
        dry_run=args.dry_run,
    )

    print()
    print(f"总计: {len(kept)} 个保留, {len(deleted)} 个{'将被' if args.dry_run else '已'}删除")

    if kept:
        print(f"保留的 step: {kept[0]}..{kept[-1]} (共 {len(kept)} 个)")
    if deleted and args.dry_run:
        print(f"\n使用不带 --dry-run 执行实际删除")

    return 0 if not deleted or args.dry_run else 0


if __name__ == "__main__":
    sys.exit(main())
