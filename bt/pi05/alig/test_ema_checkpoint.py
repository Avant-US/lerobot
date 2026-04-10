#!/usr/bin/env python3
"""
EMA Checkpoint 端到端验证脚本

验证内容:
  1. 文件结构: ema_state.safetensors 和 raw_trainable_params.safetensors 存在
  2. model.safetensors 中保存的是 EMA 参数（而非训练参数）
  3. ema_state.safetensors 中的数值与内存中的 EMA 参数一致
  4. raw_trainable_params.safetensors 中的数值与内存中的训练参数一致
  5. EMA 数值满足递推公式: ema_new = 0.99 * ema_old + 0.01 * param_new

用法:
  python -m bt.pi05.alig.test_ema_checkpoint [--steps N] [--ema-decay D]
  # 或通过 bt/pi05/alig/test_ema_checkpoint.sh 运行
"""

from __future__ import annotations

import argparse
import gc
import logging
import shutil
import sys
import tempfile
import types
from pathlib import Path

import torch
from safetensors.torch import load_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── 结果统计 ──────────────────────────────────────────────────────────────

_passed = 0
_failed = 0


def check(condition: bool, msg: str):
    """断言 + 计数, 不中断执行."""
    global _passed, _failed
    if condition:
        _passed += 1
        log.info("  ✓ PASS: %s", msg)
    else:
        _failed += 1
        log.error("  ✗ FAIL: %s", msg)


# ── 环境准备 ─────────────────────────────────────────────────────────────

def bootstrap_lerobot():
    """避免 lerobot.policies.__init__ 的全量导入."""
    if "lerobot.policies" in sys.modules:
        return
    import lerobot
    policies_dir = Path(lerobot.__file__).resolve().parent / "policies"
    pkg = types.ModuleType("lerobot.policies")
    pkg.__path__ = [str(policies_dir)]
    pkg.__package__ = "lerobot.policies"
    sys.modules["lerobot.policies"] = pkg


def make_policy_and_batch(device: str, ema_decay: float | None):
    """创建 PI05Policy + 预处理后的 dummy batch."""
    bootstrap_lerobot()
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.pi05 import make_pi05_pre_post_processors

    ACTION_DIM, STATE_DIM = 7, 14
    config = PI05Config(
        max_action_dim=32,
        max_state_dim=32,
        dtype="bfloat16",
        train_expert_only=True,
        gradient_checkpointing=True,
        ema_decay=ema_decay,
    )
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
    }

    policy = PI05Policy(config)
    dataset_stats = {
        "observation.state": {
            "mean": torch.zeros(STATE_DIM), "std": torch.ones(STATE_DIM),
            "min": torch.zeros(STATE_DIM), "max": torch.ones(STATE_DIM),
            "q01": torch.zeros(STATE_DIM), "q99": torch.ones(STATE_DIM),
        },
        "action": {
            "mean": torch.zeros(ACTION_DIM), "std": torch.ones(ACTION_DIM),
            "min": torch.zeros(ACTION_DIM), "max": torch.ones(ACTION_DIM),
            "q01": torch.zeros(ACTION_DIM), "q99": torch.ones(ACTION_DIM),
        },
        "observation.images.base_0_rgb": {
            "mean": torch.zeros(3, 224, 224), "std": torch.ones(3, 224, 224),
            "q01": torch.zeros(3, 224, 224), "q99": torch.ones(3, 224, 224),
        },
    }
    preprocessor, postprocessor = make_pi05_pre_post_processors(
        config=config, dataset_stats=dataset_stats
    )
    batch = {
        "observation.state": torch.randn(1, STATE_DIM, dtype=torch.float32, device=device),
        "action": torch.randn(1, config.chunk_size, ACTION_DIM, dtype=torch.float32, device=device),
        "observation.images.base_0_rgb": torch.rand(1, 3, 224, 224, dtype=torch.float32, device=device),
        "task": ["test task"],
    }
    batch = preprocessor(batch)
    return policy, config, batch, preprocessor, postprocessor


# ── 主测试 ────────────────────────────────────────────────────────────────

def run_test(steps: int = 5, ema_decay: float = 0.99):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s, Steps: %d, EMA decay: %s", device, steps, ema_decay)

    # ── 1. 创建 policy 并训练若干步 ──
    log.info("=" * 60)
    log.info("阶段 1: 创建 policy 并训练 %d 步 (ema_decay=%s)", steps, ema_decay)
    log.info("=" * 60)

    policy, config, batch, preprocessor, postprocessor = make_policy_and_batch(device, ema_decay)
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

    # 记录每步的 EMA 值用于公式验证
    ema_history: list[dict[str, torch.Tensor]] = []

    for step_i in range(1, steps + 1):
        policy.train()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            loss, _ = policy.forward(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 记录 optimizer step 后、EMA update 前的参数 (存 CPU 省显存)
        params_snapshot = {
            n: p.data.detach().cpu().clone() for n, p in policy.named_parameters() if p.requires_grad
        }

        policy.update()

        # 记录 EMA update 后的 EMA 值 (存 CPU)
        ema_snapshot = {k: v.detach().cpu().clone() for k, v in policy._ema_params.items()}
        ema_history.append({"params": params_snapshot, "ema": ema_snapshot})

        log.info("  step %d/%d  loss=%.4f", step_i, steps, loss.item())

    # ── 2. 验证 EMA 递推公式 ──
    log.info("=" * 60)
    log.info("阶段 2: 验证 EMA 递推公式")
    log.info("=" * 60)

    for i in range(1, len(ema_history)):
        ema_old = ema_history[i - 1]["ema"]
        params_new = ema_history[i]["params"]
        ema_actual = ema_history[i]["ema"]

        max_diff = 0.0
        for name in ema_actual:
            expected = ema_decay * ema_old[name] + (1 - ema_decay) * params_new[name]
            diff = (ema_actual[name] - expected).abs().max().item()
            max_diff = max(max_diff, diff)

        check(
            max_diff < 1e-5,
            f"Step {i+1} EMA 递推公式 (max_diff={max_diff:.2e})"
        )

    # ── 3. 记录保存前的内存状态 ──
    log.info("=" * 60)
    log.info("阶段 3: 保存 checkpoint 并验证文件")
    log.info("=" * 60)

    ema_before_save = {k: v.detach().cpu().clone() for k, v in policy._ema_params.items()}
    raw_before_save = {
        n: p.data.detach().cpu().clone() for n, p in policy.named_parameters() if p.requires_grad
    }

    # 验证 EMA 与 raw 已经分叉
    diverged_count = sum(
        1 for n in ema_before_save
        if not torch.allclose(ema_before_save[n], raw_before_save[n], atol=1e-7)
    )
    check(
        diverged_count > 0,
        f"EMA 与训练参数已分叉 ({diverged_count}/{len(ema_before_save)} 个参数不同)"
    )

    # ── 4. 调用 save_checkpoint ──
    tmpdir = Path(tempfile.mkdtemp(prefix="ema_ckpt_test_"))
    ckpt_dir = tmpdir / "checkpoint"
    log.info("  Checkpoint 目录: %s", ckpt_dir)

    # save_checkpoint 需要 TrainPipelineConfig, 我们直接调用底层 save_training_state
    # 并手动模拟 save_checkpoint 的 EMA 交换逻辑
    from lerobot.utils.train_utils import save_training_state

    pretrained_dir = ckpt_dir / "pretrained_model"
    pretrained_dir.mkdir(parents=True, exist_ok=True)

    # 交换到 EMA → 保存 model.safetensors → 恢复
    ema_backup = policy._swap_to_ema()
    policy.save_pretrained(pretrained_dir)
    policy._restore_from_backup(ema_backup)

    # 保存 training_state (含 EMA + raw)
    ema_state = policy._ema_params
    raw_trainable = {
        n: p.data.clone() for n, p in policy.named_parameters() if p.requires_grad
    }
    save_training_state(
        ckpt_dir, steps, optimizer, scheduler=None,
        ema_state=ema_state, raw_trainable_params=raw_trainable,
    )

    # ── 5. 验证文件结构 ──
    log.info("=" * 60)
    log.info("阶段 4: 验证 checkpoint 文件结构")
    log.info("=" * 60)

    model_safetensors = pretrained_dir / "model.safetensors"
    ema_safetensors = ckpt_dir / "training_state" / "ema_state.safetensors"
    raw_safetensors = ckpt_dir / "training_state" / "raw_trainable_params.safetensors"

    check(model_safetensors.exists(), f"model.safetensors 存在 ({model_safetensors})")
    check(ema_safetensors.exists(), f"ema_state.safetensors 存在 ({ema_safetensors})")
    check(raw_safetensors.exists(), f"raw_trainable_params.safetensors 存在 ({raw_safetensors})")

    if not (ema_safetensors.exists() and raw_safetensors.exists() and model_safetensors.exists()):
        log.error("关键文件缺失, 跳过后续验证")
        return

    # ── 6. 验证 model.safetensors 包含 EMA 参数 ──
    log.info("=" * 60)
    log.info("阶段 5: 验证 model.safetensors 内容 = EMA 参数")
    log.info("=" * 60)

    model_weights = load_file(str(model_safetensors), device="cpu")

    ema_match_count = 0
    ema_mismatch_count = 0
    raw_match_count = 0
    for name in ema_before_save:
        if name in model_weights:
            if torch.allclose(model_weights[name], ema_before_save[name], atol=1e-5):
                ema_match_count += 1
            else:
                ema_mismatch_count += 1
            if torch.allclose(model_weights[name], raw_before_save[name], atol=1e-5):
                raw_match_count += 1

    check(
        ema_match_count > 0 and ema_mismatch_count == 0,
        f"model.safetensors 中的可训练参数 = EMA 参数 ({ema_match_count} 匹配, {ema_mismatch_count} 不匹配)"
    )
    check(
        raw_match_count < ema_match_count,
        f"model.safetensors ≠ 原始训练参数 (仅 {raw_match_count}/{ema_match_count} 个与 raw 相同)"
    )

    # ── 7. 验证 ema_state.safetensors 数值 ──
    log.info("=" * 60)
    log.info("阶段 6: 验证 ema_state.safetensors 数值一致性")
    log.info("=" * 60)

    loaded_ema = load_file(str(ema_safetensors), device="cpu")

    ema_file_match = 0
    ema_file_total = 0
    max_ema_diff = 0.0
    for name in ema_before_save:
        if name in loaded_ema:
            ema_file_total += 1
            diff = (loaded_ema[name] - ema_before_save[name]).abs().max().item()
            max_ema_diff = max(max_ema_diff, diff)
            if diff < 1e-5:
                ema_file_match += 1

    check(
        ema_file_match == ema_file_total and ema_file_total > 0,
        f"ema_state.safetensors 与内存 EMA 一致 ({ema_file_match}/{ema_file_total}, max_diff={max_ema_diff:.2e})"
    )

    # ── 8. 验证 raw_trainable_params.safetensors 数值 ──
    log.info("=" * 60)
    log.info("阶段 7: 验证 raw_trainable_params.safetensors 数值一致性")
    log.info("=" * 60)

    loaded_raw = load_file(str(raw_safetensors), device="cpu")

    raw_file_match = 0
    raw_file_total = 0
    max_raw_diff = 0.0
    for name in raw_before_save:
        if name in loaded_raw:
            raw_file_total += 1
            diff = (loaded_raw[name] - raw_before_save[name]).abs().max().item()
            max_raw_diff = max(max_raw_diff, diff)
            if diff < 1e-5:
                raw_file_match += 1

    check(
        raw_file_match == raw_file_total and raw_file_total > 0,
        f"raw_trainable_params.safetensors 与内存训练参数一致 ({raw_file_match}/{raw_file_total}, max_diff={max_raw_diff:.2e})"
    )

    # ── 9. 验证 model 保存后 policy 参数已恢复为训练参数 ──
    log.info("=" * 60)
    log.info("阶段 8: 验证 save 后 policy 参数已恢复为训练参数")
    log.info("=" * 60)

    restore_ok = True
    for name, param in policy.named_parameters():
        if name in raw_before_save:
            if not torch.allclose(param.data.cpu(), raw_before_save[name], atol=1e-7):
                restore_ok = False
                break
    check(restore_ok, "save_checkpoint 后 policy 参数 = 原始训练参数 (非 EMA)")

    # ── 清理 ──
    shutil.rmtree(tmpdir, ignore_errors=True)
    del policy, optimizer, batch, model_weights, loaded_ema, loaded_raw
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── 汇总 ──
    log.info("=" * 60)
    total = _passed + _failed
    if _failed == 0:
        log.info("全部通过: %d/%d ✓", _passed, total)
    else:
        log.error("结果: %d passed, %d FAILED (共 %d)", _passed, _failed, total)
    log.info("=" * 60)

    return _failed == 0


def parse_args():
    p = argparse.ArgumentParser(description="EMA Checkpoint 验证")
    p.add_argument("--steps", type=int, default=5, help="训练步数 (默认 5)")
    p.add_argument("--ema-decay", type=float, default=0.99, help="EMA 衰减系数 (默认 0.99)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ok = run_test(steps=args.steps, ema_decay=args.ema_decay)
    sys.exit(0 if ok else 1)
