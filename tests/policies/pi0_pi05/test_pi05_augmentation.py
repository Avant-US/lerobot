#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for PI05 data augmentation (OpenPI alignment).

CPU tests use a mock policy to avoid needing GPU + HuggingFace weights.
GPU integration tests (marked with @require_cuda + @require_hf_token) test
the full forward/inference pipeline with augmentation enabled.
"""

import gc
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: E402


# ── Mock infrastructure ──────────────────────────────────────────────────


@dataclass
class MockAugConfig:
    """Simulates PI05Config augmentation fields for unit testing _augment_image."""

    augmentation_enabled: bool = True
    aug_crop_scale: float = 0.95
    aug_rotate_degrees: float = 5.0
    aug_color_brightness: float = 0.3
    aug_color_contrast: float = 0.4
    aug_color_saturation: float = 0.5
    aug_wrist_patterns: tuple = ("wrist",)


def make_mock_policy(config=None):
    """Construct a mock policy with only self.config needed by _augment_image."""
    if config is None:
        config = MockAugConfig()
    mock = MagicMock()
    mock.config = config
    return mock


def make_marker_image(batch_size=2, h=224, w=224):
    """Construct a marker image: white center with 5px black border.

    Geometric transforms (crop/rotate) will displace the border pixels,
    making them detectable.
    """
    img = torch.ones(batch_size, h, w, 3, dtype=torch.float32)
    border = 5
    img[:, :border, :, :] = 0.0
    img[:, -border:, :, :] = 0.0
    img[:, :, :border, :] = 0.0
    img[:, :, -border:, :] = 0.0
    return img


# ════════════════════════════════════════════════════════════════════════
# CPU unit tests (no GPU, no model weights needed)
# ════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("resolution", [(224, 224), (256, 256)])
def test_augment_output_shape_and_dtype(batch_size, resolution):
    """_augment_image preserves shape, dtype, and keeps values in [0, 1]."""
    h, w = resolution
    img = torch.rand(batch_size, h, w, 3, dtype=torch.float32)
    policy = make_mock_policy()

    result = PI05Policy._augment_image(policy, img, "observation.images.base_0_rgb")

    assert result.shape == img.shape, f"Shape mismatch: {result.shape} != {img.shape}"
    assert result.dtype == torch.float32, f"Dtype mismatch: {result.dtype}"
    assert result.min() >= 0.0, f"Value below 0: {result.min()}"
    assert result.max() <= 1.0, f"Value above 1: {result.max()}"


@pytest.mark.parametrize("fill_value", [0.0, 1.0, 0.5])
def test_augment_value_range_edge_cases(fill_value):
    """Extreme input values (all black/white/gray) stay in [0, 1] after augmentation."""
    img = torch.full((2, 224, 224, 3), fill_value, dtype=torch.float32)
    policy = make_mock_policy()

    torch.manual_seed(0)
    result = PI05Policy._augment_image(policy, img, "observation.images.base_0_rgb")

    assert result.min() >= 0.0, f"fill={fill_value}: min={result.min()}"
    assert result.max() <= 1.0, f"fill={fill_value}: max={result.max()}"


def test_augment_non_wrist_has_geometric():
    """Non-wrist camera gets geometric transforms (crop+resize+rotate), detectable via border change."""
    img = make_marker_image(batch_size=4, h=224, w=224)
    policy = make_mock_policy()

    torch.manual_seed(42)
    result = PI05Policy._augment_image(policy, img, "observation.images.base_0_rgb")

    # Border pixels: original is all 0 (black). Geometric transforms shift content.
    border_pixels = torch.cat([
        result[:, :2, :, :].reshape(-1),
        result[:, -2:, :, :].reshape(-1),
        result[:, :, :2, :].reshape(-1),
        result[:, :, -2:, :].reshape(-1),
    ])
    original_border = torch.cat([
        img[:, :2, :, :].reshape(-1),
        img[:, -2:, :, :].reshape(-1),
        img[:, :, :2, :].reshape(-1),
        img[:, :, -2:, :].reshape(-1),
    ])

    border_changed = not torch.allclose(border_pixels, original_border, atol=1e-6)
    assert border_changed, (
        "non-wrist camera: border pixels should change after geometric augmentation"
    )


def test_augment_wrist_no_geometric():
    """Wrist camera gets only color jitter, no geometric transforms."""
    img = make_marker_image(batch_size=4, h=224, w=224)
    policy = make_mock_policy()

    torch.manual_seed(42)
    result = PI05Policy._augment_image(policy, img, "observation.images.left_wrist_0_rgb")

    # Border position should be preserved (no geometric transform)
    # For all-black border pixels, brightness(0*factor)=0, so they stay near 0
    border_original = img[:, :5, :, :].reshape(-1)
    border_result = result[:, :5, :, :].reshape(-1)
    assert torch.allclose(border_result, border_original, atol=0.15), (
        "wrist camera: border pixels should stay in place (no geometric transform). "
        f"Max diff: {(border_result - border_original).abs().max()}"
    )

    # Verify color jitter is applied using a mid-gray image (0.5).
    # White (1.0) pixels are unsuitable because upward jitter gets clamped back to 1.0
    # by torchvision's internal clamping, making changes invisible.
    gray_img = torch.full((4, 224, 224, 3), 0.5, dtype=torch.float32)
    torch.manual_seed(42)
    gray_result = PI05Policy._augment_image(policy, gray_img, "observation.images.left_wrist_0_rgb")
    color_changed = not torch.allclose(gray_result, gray_img, atol=1e-6)
    assert color_changed, "wrist camera: center pixel colors should change after color jitter"


@pytest.mark.parametrize(
    "camera_key, expected_is_wrist",
    [
        ("observation.images.base_0_rgb", False),
        ("observation.images.left_wrist_0_rgb", True),
        ("observation.images.right_wrist_0_rgb", True),
        ("observation.images.head_rgb", False),
        ("observation.images.wrist_cam", True),
        ("observation.images.overhead_rgb", False),
    ],
)
def test_augment_wrist_pattern_matching(camera_key, expected_is_wrist):
    """Camera key classification: wrist cameras get no geometric transforms."""
    # Disable color jitter to isolate the geometric transform detection.
    # Color jitter (especially contrast) can shift black border pixels beyond atol,
    # creating false positives for geometric detection.
    config = MockAugConfig(
        aug_wrist_patterns=("wrist",),
        aug_color_brightness=0.0,
        aug_color_contrast=0.0,
        aug_color_saturation=0.0,
    )
    policy = make_mock_policy(config)

    img = make_marker_image(batch_size=2, h=224, w=224)
    torch.manual_seed(123)
    result = PI05Policy._augment_image(policy, img, camera_key)

    # Check if border shifted (signal of geometric transform)
    border_original = img[:, :2, :, :].reshape(-1)
    border_result = result[:, :2, :, :].reshape(-1)
    has_geometric = not torch.allclose(border_result, border_original, atol=0.1)

    if expected_is_wrist:
        assert not has_geometric, f"{camera_key}: wrist camera should NOT have geometric transforms"
    else:
        assert has_geometric, f"{camera_key}: non-wrist camera should have geometric transforms"


def test_augment_per_sample_independence():
    """Identical batch samples produce different augmented outputs (per-sample RNG)."""
    batch_size = 8
    single = torch.rand(1, 224, 224, 3, dtype=torch.float32)
    img = single.expand(batch_size, -1, -1, -1).clone()

    policy = make_mock_policy()
    torch.manual_seed(42)
    result = PI05Policy._augment_image(policy, img, "observation.images.base_0_rgb")

    num_different_pairs = 0
    for i in range(1, batch_size):
        if not torch.allclose(result[0], result[i], atol=1e-6):
            num_different_pairs += 1

    assert num_different_pairs >= batch_size // 2, (
        f"Only {num_different_pairs}/{batch_size - 1} samples differ from sample 0. "
        "Per-sample independence may not be working."
    )


def test_augment_deterministic_with_seed():
    """Fixed torch seed produces bitwise-identical augmentation results."""
    img = torch.rand(4, 224, 224, 3, dtype=torch.float32)
    policy = make_mock_policy()

    torch.manual_seed(42)
    result1 = PI05Policy._augment_image(policy, img.clone(), "observation.images.base_0_rgb")

    torch.manual_seed(42)
    result2 = PI05Policy._augment_image(policy, img.clone(), "observation.images.base_0_rgb")

    torch.testing.assert_close(result1, result2)


def test_augment_color_jitter_parameters():
    """Different color jitter config values produce different outputs."""
    img = torch.full((4, 224, 224, 3), 0.5, dtype=torch.float32)

    # Config A: brightness=0.3
    config_a = MockAugConfig(aug_color_brightness=0.3)
    policy_a = make_mock_policy(config_a)
    torch.manual_seed(42)
    result_a = PI05Policy._augment_image(policy_a, img.clone(), "observation.images.wrist_cam")

    # Config B: brightness=0.0
    config_b = MockAugConfig(aug_color_brightness=0.0)
    policy_b = make_mock_policy(config_b)
    torch.manual_seed(42)
    result_b = PI05Policy._augment_image(policy_b, img.clone(), "observation.images.wrist_cam")

    assert not torch.allclose(result_a, result_b, atol=1e-4), (
        "Different brightness settings should produce different results"
    )


# ════════════════════════════════════════════════════════════════════════
# GPU integration tests (require CUDA + HuggingFace token)
# ════════════════════════════════════════════════════════════════════════

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.pi05 import PI05Config, make_pi05_pre_post_processors  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402
from tests.utils import require_cuda, require_hf_token  # noqa: E402

ACTION_DIM = 7
STATE_DIM = 14
MAX_ACTION_DIM = 32
MAX_STATE_DIM = 32
BATCH_SIZE = 1


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _make_config(**overrides) -> PI05Config:
    kwargs = dict(
        max_action_dim=MAX_ACTION_DIM,
        max_state_dim=MAX_STATE_DIM,
        dtype="bfloat16",
        train_expert_only=True,
        gradient_checkpointing=True,
    )
    kwargs.update(overrides)
    config = PI05Config(**kwargs)
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        "observation.images.base_0_rgb": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 224, 224)
        ),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
    }
    return config


def _make_dataset_stats():
    return {
        "observation.state": {
            "mean": torch.zeros(STATE_DIM),
            "std": torch.ones(STATE_DIM),
            "min": torch.zeros(STATE_DIM),
            "max": torch.ones(STATE_DIM),
            "q01": torch.zeros(STATE_DIM),
            "q99": torch.ones(STATE_DIM),
        },
        "action": {
            "mean": torch.zeros(ACTION_DIM),
            "std": torch.ones(ACTION_DIM),
            "min": torch.zeros(ACTION_DIM),
            "max": torch.ones(ACTION_DIM),
            "q01": torch.zeros(ACTION_DIM),
            "q99": torch.ones(ACTION_DIM),
        },
        "observation.images.base_0_rgb": {
            "mean": torch.zeros(3, 224, 224),
            "std": torch.ones(3, 224, 224),
            "q01": torch.zeros(3, 224, 224),
            "q99": torch.ones(3, 224, 224),
        },
    }


def _make_batch(config, device):
    preprocessor, postprocessor = make_pi05_pre_post_processors(
        config=config, dataset_stats=_make_dataset_stats()
    )
    batch = {
        "observation.state": torch.randn(
            BATCH_SIZE, STATE_DIM, dtype=torch.float32, device=device
        ),
        "action": torch.randn(
            BATCH_SIZE, config.chunk_size, ACTION_DIM, dtype=torch.float32, device=device
        ),
        "observation.images.base_0_rgb": torch.rand(
            BATCH_SIZE, 3, 224, 224, dtype=torch.float32, device=device
        ),
        "task": ["Pick up the object"] * BATCH_SIZE,
    }
    batch = preprocessor(batch)
    return batch, preprocessor, postprocessor


@require_cuda
@require_hf_token
def test_augment_disabled_by_default():
    """Default config (augmentation_enabled=False): forward works, loss finite."""
    _cleanup()
    set_seed(42)

    config = _make_config()
    assert not config.augmentation_enabled

    policy = PI05Policy(config)
    batch, _, _ = _make_batch(config, config.device)

    policy.train()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss, loss_dict = policy.forward(batch)

    assert torch.isfinite(loss), f"Loss should be finite, got {loss}"
    assert "loss" in loss_dict

    del policy, batch
    _cleanup()


@require_cuda
@require_hf_token
def test_augment_enabled_training_forward():
    """augmentation_enabled=True: training forward works, loss finite, backward works."""
    _cleanup()
    set_seed(42)

    config = _make_config(augmentation_enabled=True)
    policy = PI05Policy(config)
    batch, _, _ = _make_batch(config, config.device)

    policy.train()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss, loss_dict = policy.forward(batch)

    assert torch.isfinite(loss), f"Loss should be finite with augmentation, got {loss}"

    loss.backward()
    grad_exists = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in policy.parameters()
        if p.requires_grad
    )
    assert grad_exists, "At least some gradients should exist and be finite"
    assert "loss" in loss_dict

    del policy, batch
    _cleanup()


@require_cuda
@require_hf_token
def test_augment_eval_mode_no_effect():
    """augmentation_enabled=True + model.eval(): inference is deterministic."""
    _cleanup()
    set_seed(42)

    config = _make_config(augmentation_enabled=True)
    policy = PI05Policy(config)
    batch, _, _ = _make_batch(config, config.device)

    policy.eval()

    set_seed(42)
    policy.reset()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        action1 = policy.predict_action_chunk(batch).clone()

    set_seed(42)
    policy.reset()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        action2 = policy.predict_action_chunk(batch).clone()

    torch.testing.assert_close(action1, action2, msg=(
        "With augmentation_enabled=True but model.eval(), "
        "inference should be deterministic."
    ))

    del policy, batch
    _cleanup()


@require_cuda
@require_hf_token
def test_augment_training_vs_eval_differ():
    """Train mode with augmentation vs eval mode produce different loss."""
    _cleanup()
    set_seed(42)

    config = _make_config(augmentation_enabled=True)
    policy = PI05Policy(config)
    batch, _, _ = _make_batch(config, config.device)

    policy.train()
    set_seed(100)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss_train, _ = policy.forward(batch)
    loss_train_val = loss_train.item()

    policy.eval()
    set_seed(100)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss_eval, _ = policy.forward(batch)
    loss_eval_val = loss_eval.item()

    assert abs(loss_train_val - loss_eval_val) > 1e-6, (
        f"train loss ({loss_train_val:.6f}) and eval loss ({loss_eval_val:.6f}) "
        "should differ when augmentation is enabled"
    )

    del policy, batch
    _cleanup()
