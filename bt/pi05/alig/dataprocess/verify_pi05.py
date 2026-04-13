#!/usr/bin/env python3
"""
验证转换后的 LeRobot v3.0 数据集是否兼容 Pi0.5 训练和评估。

验证级别:
  Level 1: 数据加载 — LeRobotDataset 加载 + 样本读取（无需 GPU）
  Level 2: 预处理器 — dataset_to_policy_features + NormalizerProcessorStep（无需 GPU）
  Level 3: 前向传播 — 加载 PI05Policy，运行 forward pass（需要 GPU，--run-forward-pass）

用法:
  python bt/pi05/alig/data/verify_pi05.py \
      --dataset-dir ./bt/pi05/alig/data/r1_pro_test_data_v30

  python bt/pi05/alig/data/verify_pi05.py \
      --dataset-dir ./bt/pi05/alig/data/r1_pro_chassis_v30 \
      --run-forward-pass \
      --pretrained-path lerobot/pi05_base \
      --num-steps 2
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import types
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def bootstrap_lerobot_policies_package():
    """避免执行 lerobot/policies/__init__.py 的全量导入。"""
    if "lerobot.policies" in sys.modules:
        return
    import lerobot

    policies_dir = Path(lerobot.__file__).resolve().parent / "policies"
    pkg = types.ModuleType("lerobot.policies")
    pkg.__path__ = [str(policies_dir)]
    pkg.__package__ = "lerobot.policies"
    sys.modules["lerobot.policies"] = pkg


def level1_data_loading(dataset_dir: Path) -> dict:
    """Level 1: 验证数据集加载。返回加载好的 dataset 和 metadata。"""
    logger.info("=" * 60)
    logger.info("Level 1: 数据加载验证")
    logger.info("=" * 60)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    repo_id = f"local/{dataset_dir.name}"

    # 1. 加载 metadata
    ds_meta = LeRobotDatasetMetadata(repo_id, root=str(dataset_dir))
    logger.info("  Metadata 加载成功")
    logger.info("    codebase_version: %s", ds_meta.info["codebase_version"])
    logger.info("    robot_type: %s", ds_meta.info.get("robot_type", "N/A"))
    logger.info("    fps: %s", ds_meta.info["fps"])
    logger.info("    total_episodes: %s", ds_meta.info["total_episodes"])
    logger.info("    total_frames: %s", ds_meta.info["total_frames"])
    logger.info("    features: %s", list(ds_meta.info["features"].keys()))

    assert ds_meta.info["codebase_version"] == "v3.0"

    # 2. 检查 stats 包含分位数
    stats = ds_meta.stats
    assert stats is not None, "stats 为 None"
    for feat_key in ["observation.state", "action"]:
        assert feat_key in stats, f"stats 缺少 {feat_key}"
        feat_stats = stats[feat_key]
        for q_key in ["q01", "q99"]:
            assert q_key in feat_stats, f"stats[{feat_key}] 缺少 {q_key}"
    logger.info("  Stats 验证通过 (含 q01/q99 分位数)")

    # 3. 加载完整 dataset
    dataset = LeRobotDataset(repo_id, root=str(dataset_dir))
    logger.info("  Dataset 加载成功: %d episodes, %d frames", dataset.num_episodes, dataset.num_frames)

    # 4. 随机读取样本
    num_samples = min(5, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)
    expected_keys = [
        "observation.images.head_rgb",
        "observation.images.left_wrist_rgb",
        "observation.images.right_wrist_rgb",
        "observation.state",
        "action",
    ]

    for idx in indices:
        sample = dataset[idx]
        for key in expected_keys:
            assert key in sample, f"样本[{idx}] 缺少 key: {key}"

        state = sample["observation.state"]
        action = sample["action"]
        assert state.shape[-1] == 23, f"state 维度错误: {state.shape}"
        assert action.shape[-1] == 23, f"action 维度错误: {action.shape}"

    logger.info("  样本读取验证通过 (%d samples, keys + shapes 正确)", num_samples)
    logger.info("Level 1 通过!")

    return {"dataset": dataset, "ds_meta": ds_meta}


def level2_preprocessor(dataset_dir: Path, context: dict) -> dict:
    """Level 2: 验证预处理器兼容性。"""
    logger.info("=" * 60)
    logger.info("Level 2: 预处理器兼容性验证")
    logger.info("=" * 60)

    bootstrap_lerobot_policies_package()

    from lerobot.configs.types import FeatureType, NormalizationMode
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    dataset = context["dataset"]
    ds_meta = context["ds_meta"]

    # 1. 提取 policy features
    features = dataset_to_policy_features(dataset.features)
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}

    logger.info("  Input features:")
    for k, ft in input_features.items():
        logger.info("    %s: type=%s, shape=%s", k, ft.type.name, ft.shape)
    logger.info("  Output features:")
    for k, ft in output_features.items():
        logger.info("    %s: type=%s, shape=%s", k, ft.type.name, ft.shape)

    # 验证分类正确
    assert "observation.state" in input_features, "observation.state 应为 input feature"
    assert "observation.images.head_rgb" in input_features, "observation.images.head_rgb 应为 input feature"
    assert "action" in output_features, "action 应为 output feature"
    assert input_features["observation.state"].type == FeatureType.STATE
    assert input_features["observation.images.head_rgb"].type == FeatureType.VISUAL
    assert output_features["action"].type == FeatureType.ACTION
    logger.info("  Feature 分类验证通过")

    # 2. 创建 PI05Config
    config = PI05Config(
        input_features=input_features,
        output_features=output_features,
    )
    config.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.QUANTILES,
        "ACTION": NormalizationMode.QUANTILES,
    }
    logger.info("  PI05Config 创建成功")

    # 3. 测试 NormalizerProcessorStep (通过 DataLoader 获取正确的 batch 格式)
    from lerobot.processor import NormalizerProcessorStep
    from lerobot.processor.converters import batch_to_transition, transition_to_batch

    normalizer = NormalizerProcessorStep(
        features={**config.input_features, **config.output_features},
        norm_map=config.normalization_mapping,
        stats=ds_meta.stats,
    )
    logger.info("  NormalizerProcessorStep 初始化成功")

    # 4. 用 DataLoader 获取 batch，通过 normalizer
    dl = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(dl))
    transition = batch_to_transition(batch)
    normalized = normalizer(transition)
    norm_batch = transition_to_batch(normalized)

    logger.info("  样本归一化通过")
    logger.info("    normalized state range: [%.3f, %.3f]",
                norm_batch["observation.state"].min().item(),
                norm_batch["observation.state"].max().item())
    logger.info("    normalized action range: [%.3f, %.3f]",
                norm_batch["action"].min().item(),
                norm_batch["action"].max().item())

    logger.info("Level 2 通过!")

    context["config"] = config
    context["input_features"] = input_features
    context["output_features"] = output_features
    return context


def level3_forward_pass(
    dataset_dir: Path,
    context: dict,
    pretrained_path: str,
    tokenizer_name: str,
    num_steps: int,
) -> None:
    """Level 3: 验证 Pi0.5 前向传播。"""
    logger.info("=" * 60)
    logger.info("Level 3: Pi0.5 前向传播验证")
    logger.info("=" * 60)

    bootstrap_lerobot_policies_package()

    from lerobot.configs.types import NormalizationMode
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
    from lerobot.processor import (
        AddBatchDimensionProcessorStep,
        DeviceProcessorStep,
        NormalizerProcessorStep,
        PolicyAction,
        PolicyProcessorPipeline,
        RenameObservationsProcessorStep,
        TokenizerProcessorStep,
        UnnormalizerProcessorStep,
    )
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
    from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME
    from transformers import AutoTokenizer

    dataset = context["dataset"]
    ds_meta = context["ds_meta"]
    config = context["config"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("  Device: %s", device)

    if device == "cpu":
        logger.warning("  无 GPU，forward pass 会很慢但仍可验证正确性")

    # 1. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False, trust_remote_code=True)
    logger.info("  Tokenizer 加载成功: %s", tokenizer_name)

    # 2. 创建带设备信息的 config
    config_with_device = PI05Config(
        dtype="float32" if device == "cpu" else "bfloat16",
        device=device,
        input_features=context["input_features"],
        output_features=context["output_features"],
    )
    config_with_device.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.QUANTILES,
        "ACTION": NormalizationMode.QUANTILES,
    }

    # 3. 加载 policy
    policy = PI05Policy.from_pretrained(pretrained_path, config=config_with_device)
    policy.to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    logger.info("  Policy 加载成功: %s params", f"{n_params:,}")

    # 4. 构建 preprocessor
    preprocessor = PolicyProcessorPipeline(
        steps=[
            RenameObservationsProcessorStep(rename_map={}),
            AddBatchDimensionProcessorStep(),
            NormalizerProcessorStep(
                features={**config_with_device.input_features, **config_with_device.output_features},
                norm_map=config_with_device.normalization_mapping,
                stats=ds_meta.stats,
            ),
            Pi05PrepareStateTokenizerProcessorStep(max_state_dim=config_with_device.max_state_dim),
            TokenizerProcessorStep(
                tokenizer=tokenizer,
                max_length=config_with_device.tokenizer_max_length,
                padding_side="right",
                padding="max_length",
            ),
            DeviceProcessorStep(device=device),
        ],
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
    )

    # 5. 运行 forward pass
    policy.train()
    for step in range(num_steps):
        sample = dataset[step % len(dataset)]
        batch = preprocessor(sample)

        autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        with torch.autocast(device_type=device, dtype=autocast_dtype):
            loss, aux = policy.forward(batch)

        loss_val = loss.item()
        logger.info("  Step %d: loss = %.4f (finite: %s)", step + 1, loss_val, torch.isfinite(loss).item())
        assert torch.isfinite(loss), f"Step {step + 1}: loss 不是有限值!"

    logger.info("Level 3 通过! (%d steps, loss 有限)", num_steps)


def main():
    parser = argparse.ArgumentParser(description="验证转换后数据集的 Pi0.5 兼容性")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="转换后的 v3.0 数据集目录")
    parser.add_argument("--run-forward-pass", action="store_true", help="运行 Level 3 前向传播验证")
    parser.add_argument("--pretrained-path", default="lerobot/pi05_base", help="Pi0.5 预训练模型路径")
    parser.add_argument("--tokenizer-name", default="google/paligemma-3b-pt-224", help="Tokenizer 名称")
    parser.add_argument("--num-steps", type=int, default=2, help="前向传播步数")
    args = parser.parse_args()

    # Level 1
    context = level1_data_loading(args.dataset_dir)

    # Level 2
    context = level2_preprocessor(args.dataset_dir, context)

    # Level 3 (可选)
    if args.run_forward_pass:
        level3_forward_pass(
            args.dataset_dir,
            context,
            pretrained_path=args.pretrained_path,
            tokenizer_name=args.tokenizer_name,
            num_steps=args.num_steps,
        )

    logger.info("=" * 60)
    logger.info("全部验证通过!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
