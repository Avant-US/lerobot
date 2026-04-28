#!/usr/bin/env python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Compute a dexbotic-style ``norm_stats.json`` from a LeRobotDataset.

The DM0 policy expects ``norm_stats_path`` to point at a JSON file with the schema::

    {
      "norm_stats": {
        "default": {"min": -1, "max": 1},
        "action":  {"min": [...], "max": [...], "mean": [...], "std": [...]},
        "state":   {"min": [...], "max": [...], "mean": [...], "std": [...]},
      }
    }

where ``min`` / ``max`` are the q01 / q99 quantiles per dimension. The action statistics
are computed on **delta-action** chunks of length ``--chunk-size`` (with selected
``--non-delta-mask`` dims kept absolute), padded to ``--max-action-dim``. State stats are
computed on the per-frame state padded to ``--max-state-dim``.

This mirrors ``DM0Policy.forward`` so the buffers loaded back in the policy match what
training will see at runtime.

Reads action / state directly from ``hf_dataset`` (no video decoding needed).

Example::

    python lerobot/examples/dm0/compute_norm_stats.py \\
      --repo-id your-org/r1_pro_chassis_v3_lerobot \\
      --output ./norm_stats/r1_pro_chassis_v3.json \\
      --chunk-size 50 --non-delta-mask 14 15 \\
      --max-action-dim 32 --max-state-dim 32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _pad_last_dim(x: np.ndarray, ndim: int) -> np.ndarray:
    d = x.shape[-1]
    if d >= ndim:
        return x[..., :ndim]
    pad_widths = [(0, 0)] * (x.ndim - 1) + [(0, ndim - d)]
    return np.pad(x, pad_widths)


def _wrap_periodic(delta: np.ndarray, dims: list[int], period: float) -> np.ndarray:
    if not dims or period <= 0:
        return delta
    half = period / 2.0
    sub = delta[..., dims].copy()
    sub = np.where(sub > half, sub - period, sub)
    sub = np.where(sub < -half, sub + period, sub)
    delta[..., dims] = sub
    return delta


def _build_delta_chunks(
    actions: np.ndarray,
    states: np.ndarray,
    chunk_size: int,
    non_delta_mask: list[int],
    periodic_mask: list[int],
    periodic_range: tuple[float, float],
    use_delta: bool,
) -> np.ndarray:
    """Return (T, chunk_size, action_dim) chunks for a single contiguous episode segment.

    Future frames past episode end are padded with the last action (matches
    ``AddTrajectory(padding_mode='last')``).
    """
    t = actions.shape[0]
    chunks = np.empty((t, chunk_size, actions.shape[-1]), dtype=actions.dtype)
    for i in range(t):
        end = min(i + chunk_size, t)
        chunks[i, : end - i] = actions[i:end]
        if end - i < chunk_size:
            chunks[i, end - i :] = actions[t - 1]
    if not use_delta:
        return chunks
    delta = chunks - states[:, None, :]
    if periodic_mask:
        period = float(periodic_range[1] - periodic_range[0])
        delta = _wrap_periodic(delta, list(periodic_mask), period)
    if non_delta_mask:
        for d in non_delta_mask:
            delta[..., d] = chunks[..., d]
    return delta


def _quantile_stats(x: np.ndarray) -> dict[str, list[float]]:
    x64 = x.astype(np.float64, copy=False)
    return {
        "min": np.quantile(x64, 0.01, axis=0).tolist(),
        "max": np.quantile(x64, 0.99, axis=0).tolist(),
        "mean": x64.mean(axis=0).tolist(),
        "std": x64.std(axis=0).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True, help="LeRobotDataset repo id.")
    parser.add_argument("--root", default=None, help="Local dataset root override.")
    parser.add_argument("--episodes", type=int, nargs="*", default=None, help="Subset of episode indices.")
    parser.add_argument("--output", required=True, help="Path of the norm_stats.json to write.")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--max-state-dim", type=int, default=32)
    parser.add_argument("--max-action-dim", type=int, default=32)
    parser.add_argument(
        "--non-delta-mask",
        type=int,
        nargs="*",
        default=[],
        help="Action dimensions kept absolute (e.g. gripper indices).",
    )
    parser.add_argument("--periodic-mask", type=int, nargs="*", default=[])
    parser.add_argument("--periodic-range", type=float, nargs=2, default=(-3.14159, 3.14159))
    parser.add_argument(
        "--no-delta",
        dest="use_delta",
        action="store_false",
        help="Disable delta-action transform (matches DM0Config.use_delta_action=False).",
    )
    parser.set_defaults(use_delta=True)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=320_000,
        help="Cap total per-feature samples for q01/q99 (mirrors dexbotic's 2500*128 cap).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"[dm0-norm-stats] loading {args.repo_id} (root={args.root})")
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.root,
        episodes=args.episodes,
        download_videos=False,
    )
    hf = dataset.hf_dataset.with_format(
        "numpy", columns=["action", "observation.state", "episode_index", "frame_index"]
    )

    ep_idx = np.asarray(hf["episode_index"])
    fr_idx = np.asarray(hf["frame_index"])
    actions = np.asarray(hf["action"])
    states = np.asarray(hf["observation.state"])

    order = np.lexsort((fr_idx, ep_idx))
    ep_idx = ep_idx[order]
    actions = actions[order]
    states = states[order]

    actions = _pad_last_dim(actions.astype(np.float32, copy=False), args.max_action_dim)
    states = _pad_last_dim(states.astype(np.float32, copy=False), args.max_state_dim)

    seg_breaks = np.where(np.diff(ep_idx) != 0)[0] + 1
    seg_starts = np.concatenate([[0], seg_breaks])
    seg_ends = np.concatenate([seg_breaks, [len(ep_idx)]])

    action_chunks: list[np.ndarray] = []
    total_action_rows = 0
    for s, e in tqdm(list(zip(seg_starts, seg_ends, strict=True)), desc="episodes"):
        chunks = _build_delta_chunks(
            actions[s:e],
            states[s:e],
            chunk_size=args.chunk_size,
            non_delta_mask=list(args.non_delta_mask),
            periodic_mask=list(args.periodic_mask),
            periodic_range=tuple(args.periodic_range),
            use_delta=args.use_delta,
        )
        flat = chunks.reshape(-1, chunks.shape[-1])
        action_chunks.append(flat)
        total_action_rows += flat.shape[0]

    action_concat = np.concatenate(action_chunks, axis=0)
    state_concat = states  # one row per frame

    if action_concat.shape[0] > args.max_samples:
        sel = rng.choice(action_concat.shape[0], size=args.max_samples, replace=False)
        action_concat = action_concat[sel]
    if state_concat.shape[0] > args.max_samples:
        sel = rng.choice(state_concat.shape[0], size=args.max_samples, replace=False)
        state_concat = state_concat[sel]

    norm_stats = {
        "default": {"min": -1, "max": 1},
        "action": _quantile_stats(action_concat),
        "state": _quantile_stats(state_concat),
    }

    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump({"norm_stats": norm_stats}, f, indent=2)

    print(f"[dm0-norm-stats] wrote {out}")
    print(
        f"  action samples: {action_concat.shape[0]} (dim={action_concat.shape[1]}, "
        f"chunk_size={args.chunk_size}, use_delta={args.use_delta})"
    )
    print(f"  state  samples: {state_concat.shape[0]} (dim={state_concat.shape[1]})")


if __name__ == "__main__":
    main()
