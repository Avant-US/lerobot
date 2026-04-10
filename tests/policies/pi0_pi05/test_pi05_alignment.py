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
Tests for OpenPI alignment features: Loss truncation and EMA.

These tests verify that the `loss_include_padding` and `ema_decay` config options
correctly reproduce OpenPI JAX behavior in LeRobot's PI0.5 implementation.

Uses train_expert_only=True + bfloat16 to fit within GPU memory.
"""

import gc

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.pi05 import (  # noqa: E402
    PI05Config,
    PI05Policy,
    make_pi05_pre_post_processors,
)
from lerobot.utils.random_utils import set_seed  # noqa: E402
from tests.utils import require_cuda, require_hf_token  # noqa: E402

# ── Shared fixtures ────────────────────────────────────────────────────

ACTION_DIM = 7
STATE_DIM = 14
MAX_ACTION_DIM = 32
MAX_STATE_DIM = 32
BATCH_SIZE = 1


def _cleanup():
    """Force GPU memory cleanup between tests."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _make_config(**overrides) -> PI05Config:
    """Create a PI05Config with sensible test defaults.
    Uses train_expert_only + bfloat16 + gradient_checkpointing to save GPU memory.
    """
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
    """Create dummy dataset stats matching the test config."""
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
    """Create a dummy training batch."""
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


# ════════════════════════════════════════════════════════════════════════
# Loss 截断对齐 tests
# ════════════════════════════════════════════════════════════════════════


@require_cuda
@require_hf_token
def test_loss_truncation_gradient_coverage():
    """Verify gradient coverage difference between truncated and full-dim loss.

    loss_include_padding=False (default): only first ACTION_DIM columns of
    action_out_proj get gradients; padding columns have zero gradient.

    loss_include_padding=True (OpenPI): all MAX_ACTION_DIM columns get gradients.
    """
    _cleanup()
    set_seed(42)

    # ── Truncated mode (default) ──
    config_trunc = _make_config(loss_include_padding=False)
    policy_trunc = PI05Policy(config_trunc)
    batch_trunc, _, _ = _make_batch(config_trunc, config_trunc.device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss_trunc, _ = policy_trunc.forward(batch_trunc)
    loss_trunc.backward()

    # action_out_proj maps hidden_dim -> max_action_dim
    grad_trunc = policy_trunc.model.action_out_proj.weight.grad  # [max_action_dim, hidden]
    assert grad_trunc is not None, "action_out_proj should have gradients"

    # Padding rows (ACTION_DIM:) should have zero gradient
    padding_grad_norm = grad_trunc[ACTION_DIM:].abs().sum().item()
    action_grad_norm = grad_trunc[:ACTION_DIM].abs().sum().item()
    assert padding_grad_norm == 0.0, (
        f"Truncated mode: padding dim gradients should be 0, got {padding_grad_norm}"
    )
    assert action_grad_norm > 0.0, "Truncated mode: action dim gradients should be non-zero"

    # Save loss value for comparison
    loss_trunc_val = loss_trunc.item()

    # Cleanup before creating second model
    del policy_trunc, batch_trunc, loss_trunc, grad_trunc
    _cleanup()

    # ── Full-dim mode (OpenPI) ──
    set_seed(42)
    config_full = _make_config(loss_include_padding=True)
    policy_full = PI05Policy(config_full)
    batch_full, _, _ = _make_batch(config_full, config_full.device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss_full, _ = policy_full.forward(batch_full)
    loss_full.backward()

    grad_full = policy_full.model.action_out_proj.weight.grad
    assert grad_full is not None

    # ALL rows should have non-zero gradients
    padding_grad_norm_full = grad_full[ACTION_DIM:].abs().sum().item()
    action_grad_norm_full = grad_full[:ACTION_DIM].abs().sum().item()
    assert padding_grad_norm_full > 0.0, (
        f"Full-dim mode: padding dim gradients should be non-zero, got {padding_grad_norm_full}"
    )
    assert action_grad_norm_full > 0.0, "Full-dim mode: action dim gradients should be non-zero"

    # Also verify loss values differ
    loss_full_val = loss_full.item()
    assert abs(loss_trunc_val - loss_full_val) > 1e-6, (
        f"Truncated loss ({loss_trunc_val:.6f}) and full loss ({loss_full_val:.6f}) should differ"
    )

    del policy_full, batch_full, loss_full, grad_full
    _cleanup()


# ════════════════════════════════════════════════════════════════════════
# EMA 对齐 tests
# ════════════════════════════════════════════════════════════════════════


@require_cuda
@require_hf_token
def test_ema_disabled_by_default():
    """When ema_decay=None (default), update() is a no-op and _ema_params stays None."""
    _cleanup()
    set_seed(42)
    config = _make_config(ema_decay=None)
    policy = PI05Policy(config)

    assert policy._ema_params is None, "EMA should be None when ema_decay is not set"

    # Simulate a training step
    batch, _, _ = _make_batch(config, config.device)
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss, _ = policy.forward(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    policy.update()

    assert policy._ema_params is None, "EMA should remain None after update() when disabled"

    del policy, batch, optimizer
    _cleanup()


@require_cuda
@require_hf_token
def test_ema_init_on_first_update():
    """EMA is lazily initialized on first update() call.

    After first update(), _ema_params should equal current params (since init copies
    current params then immediately applies ema = 0.99*current + 0.01*current = current).
    """
    _cleanup()
    set_seed(42)
    config = _make_config(ema_decay=0.99)
    policy = PI05Policy(config)

    assert policy._ema_params is None, "EMA should be None before first update()"

    # Do one optimizer step then update EMA
    batch, _, _ = _make_batch(config, config.device)
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss, _ = policy.forward(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Record params BEFORE EMA update (post-optimizer-step)
    params_before_ema = {
        n: p.data.clone() for n, p in policy.named_parameters() if p.requires_grad
    }

    # First update: _init_ema copies current params, then ema = 0.99*current + 0.01*current = current
    policy.update()

    assert policy._ema_params is not None, "EMA should be initialized after first update()"

    # EMA params should match current params after first init
    for name in policy._ema_params:
        assert torch.allclose(policy._ema_params[name], params_before_ema[name], atol=1e-6), (
            f"After first update, EMA[{name}] should equal current params"
        )

    del policy, batch, optimizer, params_before_ema
    _cleanup()


@require_cuda
@require_hf_token
def test_ema_update_formula():
    """Verify EMA update follows: ema_new = decay * ema_old + (1 - decay) * param_new."""
    _cleanup()
    set_seed(42)
    decay = 0.99
    config = _make_config(ema_decay=decay)
    policy = PI05Policy(config)
    batch, _, _ = _make_batch(config, config.device)
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

    # Step 1: init EMA
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss, _ = policy.forward(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    policy.update()

    # Record EMA after init
    ema_after_init = {k: v.clone() for k, v in policy._ema_params.items()}

    # Step 2: another optimizer step
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss, _ = policy.forward(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Record current params (after optimizer step, before EMA update)
    params_new = {
        n: p.data.clone() for n, p in policy.named_parameters() if p.requires_grad
    }

    # Update EMA
    policy.update()

    # Verify formula: ema_new = 0.99 * ema_old + 0.01 * param_new
    for name in policy._ema_params:
        expected = decay * ema_after_init[name] + (1 - decay) * params_new[name]
        assert torch.allclose(policy._ema_params[name], expected, atol=1e-5), (
            f"EMA update formula mismatch for {name}: "
            f"max diff = {(policy._ema_params[name] - expected).abs().max().item()}"
        )

    del policy, batch, optimizer, ema_after_init, params_new
    _cleanup()


@require_cuda
@require_hf_token
def test_ema_inference_uses_ema_params():
    """select_action should use EMA parameters, producing different output than raw params."""
    _cleanup()
    set_seed(42)
    config = _make_config(ema_decay=0.99)
    policy = PI05Policy(config)
    batch, _, postprocessor = _make_batch(config, config.device)
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)

    # Train several steps to diverge EMA from raw params
    for _ in range(5):
        policy.train()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, _ = policy.forward(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        policy.update()

    # Verify EMA and raw params have diverged
    has_diverged = False
    for name, param in policy.named_parameters():
        if name in policy._ema_params:
            if not torch.allclose(param.data, policy._ema_params[name], atol=1e-7):
                has_diverged = True
                break
    assert has_diverged, "EMA and raw params should diverge after training steps"

    # Get action with EMA (normal path)
    policy.eval()
    policy.reset()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        action_ema = policy.select_action(batch).clone()

    # Get action WITHOUT EMA by temporarily disabling it
    saved_ema = policy._ema_params
    policy._ema_params = None
    policy.reset()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        action_raw = policy.select_action(batch).clone()
    policy._ema_params = saved_ema

    # Actions should differ
    assert not torch.allclose(action_ema, action_raw, atol=1e-6), (
        "EMA and raw actions should differ after training"
    )

    del policy, batch, optimizer
    _cleanup()


@require_cuda
@require_hf_token
def test_ema_swap_restore_roundtrip():
    """_swap_to_ema / _restore_from_backup should be a perfect roundtrip."""
    _cleanup()
    set_seed(42)
    config = _make_config(ema_decay=0.99)
    policy = PI05Policy(config)
    batch, _, _ = _make_batch(config, config.device)
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)

    # Train a few steps to create divergent EMA
    for _ in range(3):
        policy.train()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, _ = policy.forward(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        policy.update()

    # Record original training params
    original_params = {
        n: p.data.clone() for n, p in policy.named_parameters() if p.requires_grad
    }

    # Swap to EMA
    backup = policy._swap_to_ema()
    assert backup is not None, "Swap should return backup dict"

    # Verify model now holds EMA params
    for name, param in policy.named_parameters():
        if name in policy._ema_params:
            assert torch.allclose(param.data, policy._ema_params[name], atol=1e-7), (
                f"After swap, model param {name} should equal EMA param"
            )

    # Restore
    policy._restore_from_backup(backup)

    # Verify model restored to original training params
    for name, param in policy.named_parameters():
        if name in original_params:
            assert torch.allclose(param.data, original_params[name], atol=1e-7), (
                f"After restore, model param {name} should equal original training param"
            )

    del policy, batch, optimizer, original_params
    _cleanup()


@require_cuda
@require_hf_token
def test_ema_nested_swap_protection():
    """_swap_to_ema should return None when already active (prevents nested swaps)."""
    _cleanup()
    set_seed(42)
    config = _make_config(ema_decay=0.99)
    policy = PI05Policy(config)
    batch, _, _ = _make_batch(config, config.device)
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

    # Init EMA
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss, _ = policy.forward(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    policy.update()

    # First swap succeeds
    backup = policy._swap_to_ema()
    assert backup is not None
    assert policy._ema_active is True

    # Nested swap returns None (no-op)
    nested_backup = policy._swap_to_ema()
    assert nested_backup is None, "Nested swap should return None"

    # Restore
    policy._restore_from_backup(backup)
    assert policy._ema_active is False

    del policy, batch, optimizer
    _cleanup()
