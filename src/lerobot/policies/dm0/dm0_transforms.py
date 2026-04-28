# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Pad / delta-action / quantile norm helpers for DM0Policy (dexbotic-compatible stats)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def pad_to_dim(x: Tensor, ndim: int) -> Tensor:
    """Zero-pad the last dimension of `x` to length `ndim`."""
    d = x.shape[-1]
    if d >= ndim:
        return x[..., :ndim]
    return torch.nn.functional.pad(x, (0, ndim - d))


def _wrap_periodic(x: Tensor, dims: list[int], period: float) -> Tensor:
    """Wrap selected dimensions of `x` into [-period/2, +period/2]."""
    if not dims or period <= 0:
        return x
    half = period / 2.0
    sub = x[..., dims]
    sub = torch.where(sub > half, sub - period, sub)
    sub = torch.where(sub < -half, sub + period, sub)
    x[..., dims] = sub
    return x


def compute_delta(
    action: Tensor,
    state: Tensor,
    non_delta_mask: tuple[int, ...] | list[int],
    periodic_mask: tuple[int, ...] | list[int] | None = None,
    periodic_range: tuple[float, float] = (-3.14159, 3.14159),
) -> Tensor:
    """Match dexbotic `DeltaAction`: delta = action - state; non-delta dims keep absolute action.

    If ``periodic_mask`` is non-empty, those dimensions are wrapped into the periodic interval
    defined by ``periodic_range`` so that e.g. yaw differences cross the ±π boundary correctly.
    """
    state_b = state.unsqueeze(1)
    delta = action - state_b
    if periodic_mask:
        period = float(periodic_range[1] - periodic_range[0])
        delta = _wrap_periodic(delta, list(periodic_mask), period)
    if non_delta_mask:
        idx = list(non_delta_mask)
        delta[..., idx] = action[..., idx]
    return delta


def compute_absolute(
    delta: Tensor,
    state: Tensor,
    non_delta_mask: tuple[int, ...] | list[int],
    periodic_mask: tuple[int, ...] | list[int] | None = None,
    periodic_range: tuple[float, float] = (-3.14159, 3.14159),
) -> Tensor:
    """Inverse of :func:`compute_delta`, with optional ±period/2 wrap on ``periodic_mask`` dims."""
    state_b = state.unsqueeze(1)
    absolute = state_b + delta
    if periodic_mask:
        period = float(periodic_range[1] - periodic_range[0])
        absolute = _wrap_periodic(absolute, list(periodic_mask), period)
    if non_delta_mask:
        idx = list(non_delta_mask)
        absolute[..., idx] = delta[..., idx]
    return absolute


def quantile_normalize(x: Tensor, vmin: Tensor, vmax: Tensor, eps: float = 1e-6) -> Tensor:
    """Map to [-1, 1] using min/max (dexbotic stores q01/q99 in `min`/`max` fields)."""
    return (x - vmin) / (vmax - vmin + eps) * 2.0 - 1.0


def quantile_denormalize(x: Tensor, vmin: Tensor, vmax: Tensor, eps: float = 1e-6) -> Tensor:
    """Invert `quantile_normalize`."""
    return (x + 1.0) * 0.5 * (vmax - vmin + eps) + vmin


def load_norm_stats(
    path: str | Path,
    device: torch.device,
    max_state_dim: int,
    max_action_dim: int,
) -> dict[str, Any]:
    """Load dexbotic `norm_stats.json` and build padded min/max tensors on `device`."""
    path = Path(path)
    with path.open() as f:
        raw = json.load(f)
    stats = raw.get("norm_stats", raw)

    def _padded_min_max(key: str) -> tuple[Tensor, Tensor]:
        block = stats[key]
        max_dim = max_state_dim if key == "state" else max_action_dim
        mn_l = list(block["min"])
        mx_l = list(block["max"])
        while len(mn_l) < max_dim:
            mn_l.append(-1.0)
            mx_l.append(1.0)
        mn = torch.tensor(mn_l[:max_dim], dtype=torch.float32, device=device)
        mx = torch.tensor(mx_l[:max_dim], dtype=torch.float32, device=device)
        return mn, mx

    action_min, action_max = _padded_min_max("action")
    state_min, state_max = _padded_min_max("state")
    return {
        "action_min": action_min,
        "action_max": action_max,
        "state_min": state_min,
        "state_max": state_max,
    }
