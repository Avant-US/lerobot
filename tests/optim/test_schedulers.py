# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import math

import torch
from packaging.version import Version
from torch.optim.lr_scheduler import LambdaLR

from lerobot.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
    DiffuserSchedulerConfig,
    VQBeTSchedulerConfig,
    load_scheduler_state,
    save_scheduler_state,
)
from lerobot.utils.constants import SCHEDULER_STATE


def test_diffuser_scheduler(optimizer):
    config = DiffuserSchedulerConfig(name="cosine", num_warmup_steps=5)
    scheduler = config.build(optimizer, num_training_steps=100)
    assert isinstance(scheduler, LambdaLR)

    optimizer.step()  # so that we don't get torch warning
    scheduler.step()
    expected_state_dict = {
        "_get_lr_called_within_step": False,
        "_last_lr": [0.0002],
        "_step_count": 2,
        "base_lrs": [0.001],
        "last_epoch": 1,
        "lr_lambdas": [None],
    }

    if Version(torch.__version__) >= Version("2.8"):
        expected_state_dict["_is_initial"] = False

    assert scheduler.state_dict() == expected_state_dict


def test_vqbet_scheduler(optimizer):
    config = VQBeTSchedulerConfig(num_warmup_steps=10, num_vqvae_training_steps=20, num_cycles=0.5)
    scheduler = config.build(optimizer, num_training_steps=100)
    assert isinstance(scheduler, LambdaLR)

    optimizer.step()
    scheduler.step()
    expected_state_dict = {
        "_get_lr_called_within_step": False,
        "_last_lr": [0.001],
        "_step_count": 2,
        "base_lrs": [0.001],
        "last_epoch": 1,
        "lr_lambdas": [None],
    }

    if Version(torch.__version__) >= Version("2.8"):
        expected_state_dict["_is_initial"] = False

    assert scheduler.state_dict() == expected_state_dict


def test_cosine_decay_with_warmup_scheduler(optimizer):
    config = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=10, num_decay_steps=90, peak_lr=0.01, decay_lr=0.001
    )
    scheduler = config.build(optimizer, num_training_steps=100)
    assert isinstance(scheduler, LambdaLR)

    optimizer.step()
    scheduler.step()
    expected_state_dict = {
        "_get_lr_called_within_step": False,
        "_last_lr": [0.0001818181818181819],
        "_step_count": 2,
        "base_lrs": [0.001],
        "last_epoch": 1,
        "lr_lambdas": [None],
    }

    if Version(torch.__version__) >= Version("2.8"):
        expected_state_dict["_is_initial"] = False

    assert scheduler.state_dict() == expected_state_dict


def test_save_scheduler_state(scheduler, tmp_path):
    save_scheduler_state(scheduler, tmp_path)
    assert (tmp_path / SCHEDULER_STATE).is_file()


def test_save_load_scheduler_state(scheduler, tmp_path):
    save_scheduler_state(scheduler, tmp_path)
    loaded_scheduler = load_scheduler_state(scheduler, tmp_path)

    assert scheduler.state_dict() == loaded_scheduler.state_dict()


# ════════════════════════════════════════════════════════════════════════
# phase_mode tests
# ════════════════════════════════════════════════════════════════════════


def test_cosine_decay_default_is_absolute(optimizer):
    """phase_mode defaults to 'absolute', behavior identical to pre-change code."""
    config = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=10, num_decay_steps=90, peak_lr=0.01, decay_lr=0.001
    )
    assert config.phase_mode == "absolute"

    scheduler = config.build(optimizer, num_training_steps=100)
    optimizer.step()
    scheduler.step()

    # This value comes from the existing test_cosine_decay_with_warmup_scheduler
    assert scheduler.state_dict()["_last_lr"] == [0.0001818181818181819]


def test_cosine_decay_post_warmup_matches_optax(model_params):
    """Verify phase_mode='post_warmup' matches optax.warmup_cosine_decay_schedule
    at key steps using pi05_r1pro_chassis real parameters."""
    peak_lr = 2.5e-5
    decay_lr = 2.5e-6
    warmup_steps = 1000
    decay_steps = 30000

    config = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=warmup_steps,
        num_decay_steps=decay_steps,
        peak_lr=peak_lr,
        decay_lr=decay_lr,
        phase_mode="post_warmup",
    )

    init_lr = peak_lr / (warmup_steps + 1)

    def optax_lr(step):
        """Hand-computed optax warmup_cosine_decay_schedule reference."""
        if step < warmup_steps:
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        progress = (step - warmup_steps) / (decay_steps - warmup_steps)
        return decay_lr + 0.5 * (peak_lr - decay_lr) * (1 + math.cos(math.pi * progress))

    checkpoints = [0, 500, 999, 1000, 1001, 15000, 29999, 30000]

    for target_step in checkpoints:
        optimizer = torch.optim.AdamW(model_params, lr=peak_lr)
        scheduler = config.build(optimizer, num_training_steps=decay_steps)

        for _ in range(target_step):
            optimizer.step()
            scheduler.step()

        actual_lr = scheduler.get_last_lr()[0]
        expected_lr = optax_lr(target_step)

        if expected_lr > 0:
            rel_error = abs(actual_lr - expected_lr) / expected_lr
            assert rel_error < 1e-10, (
                f"Step {target_step}: actual={actual_lr:.15e}, expected={expected_lr:.15e}, "
                f"rel_error={rel_error:.2e}"
            )


def test_cosine_decay_post_warmup_peak_exact(model_params):
    """At step=warmup_steps, LR must be exactly peak_lr in post_warmup mode."""
    config = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=1000,
        num_decay_steps=30000,
        peak_lr=2.5e-5,
        decay_lr=2.5e-6,
        phase_mode="post_warmup",
    )
    optimizer = torch.optim.AdamW(model_params, lr=2.5e-5)
    scheduler = config.build(optimizer, num_training_steps=30000)

    for _ in range(1000):
        optimizer.step()
        scheduler.step()

    actual = scheduler.get_last_lr()[0]
    assert actual == 2.5e-5, f"Step 1000: expected peak_lr=2.5e-5, got {actual}"


def test_cosine_decay_absolute_vs_post_warmup_differ(model_params):
    """Two modes agree during warmup and at endpoint, but differ in post-warmup cosine segment."""
    kwargs = dict(num_warmup_steps=100, num_decay_steps=1000, peak_lr=1e-3, decay_lr=1e-4)

    def get_lr_at_step(phase_mode, step):
        config = CosineDecayWithWarmupSchedulerConfig(phase_mode=phase_mode, **kwargs)
        opt = torch.optim.AdamW(model_params, lr=kwargs["peak_lr"])
        sched = config.build(opt, num_training_steps=1000)
        for _ in range(step):
            opt.step()
            sched.step()
        return sched.get_last_lr()[0]

    # Warmup segment: both modes should agree
    for step in [0, 50, 99]:
        abs_lr = get_lr_at_step("absolute", step)
        pw_lr = get_lr_at_step("post_warmup", step)
        assert abs(abs_lr - pw_lr) < 1e-15, f"Warmup step {step}: modes should agree"

    # Post-warmup: should differ
    for step in [100, 500, 800]:
        abs_lr = get_lr_at_step("absolute", step)
        pw_lr = get_lr_at_step("post_warmup", step)
        assert abs_lr != pw_lr, f"Post-warmup step {step}: modes should differ"

    # Endpoint: should agree at decay_lr
    abs_lr = get_lr_at_step("absolute", 1000)
    pw_lr = get_lr_at_step("post_warmup", 1000)
    assert abs(abs_lr - pw_lr) < 1e-12, "Final step: modes should agree at decay_lr"


def test_cosine_decay_post_warmup_auto_scaling(model_params):
    """Auto-scaling with post_warmup: scaled warmup endpoint should be peak_lr."""
    config = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=1000,
        num_decay_steps=30000,
        peak_lr=2.5e-5,
        decay_lr=2.5e-6,
        phase_mode="post_warmup",
    )
    optimizer = torch.optim.AdamW(model_params, lr=2.5e-5)
    # num_training_steps=3000 triggers auto-scaling: actual_warmup=100, actual_decay=3000
    scheduler = config.build(optimizer, num_training_steps=3000)

    # Step to actual_warmup=100
    for _ in range(100):
        optimizer.step()
        scheduler.step()

    actual = scheduler.get_last_lr()[0]
    assert abs(actual - 2.5e-5) < 1e-15, (
        f"After auto-scaling, step=100 should be peak_lr, got {actual}"
    )

    # Step to endpoint 3000
    for _ in range(2900):
        optimizer.step()
        scheduler.step()

    actual = scheduler.get_last_lr()[0]
    assert abs(actual - 2.5e-6) < 1e-12, (
        f"At final step 3000, should be decay_lr, got {actual}"
    )


def test_scheduler_state_save_load_with_phase_mode(model_params, tmp_path):
    """save/load_scheduler_state roundtrip works with phase_mode."""
    config = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=100,
        num_decay_steps=1000,
        peak_lr=1e-3,
        decay_lr=1e-4,
        phase_mode="post_warmup",
    )
    optimizer = torch.optim.AdamW(model_params, lr=1e-3)
    scheduler = config.build(optimizer, num_training_steps=1000)

    for _ in range(100):
        optimizer.step()
        scheduler.step()

    # Save
    save_scheduler_state(scheduler, tmp_path)
    assert (tmp_path / SCHEDULER_STATE).is_file()

    # Construct new scheduler and load
    optimizer2 = torch.optim.AdamW(model_params, lr=1e-3)
    scheduler2 = config.build(optimizer2, num_training_steps=1000)
    loaded = load_scheduler_state(scheduler2, tmp_path)

    assert loaded.state_dict()["last_epoch"] == scheduler.state_dict()["last_epoch"]

    # Continue stepping both; they should produce same LR
    optimizer2.step()
    loaded.step()
    lr_after_load = loaded.get_last_lr()[0]

    optimizer.step()
    scheduler.step()
    lr_continued = scheduler.get_last_lr()[0]

    assert abs(lr_after_load - lr_continued) < 1e-15, (
        f"Resume mismatch: loaded={lr_after_load}, continued={lr_continued}"
    )
