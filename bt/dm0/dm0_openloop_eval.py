#!/usr/bin/env python3
"""DM0 open-loop action MSE on a LeRobotDataset (LeRobot policy path).

Loads ``DM0Policy.from_pretrained`` and runs ``predict_action_chunk`` so preprocessing
matches training / ``select_action`` (quantile norm, diffusion, delta → absolute).

Metrics mirror ``dexbotic/playground/benchmarks/real/r1_openloop_eval.py``:
per-episode / overall MSE, per-dimension, per-group, avg inference ms, optional
multi-checkpoint comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

_LEROBOT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _LEROBOT_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.dm0.modeling_dm0 import DM0Policy
from lerobot.utils.constants import ACTION, OBS_STATE


def _action_to_numpy(item: dict, action_dim: int) -> np.ndarray:
    a = item[ACTION]
    if isinstance(a, torch.Tensor):
        a = a.detach().float().cpu().numpy()
    return np.asarray(a, dtype=np.float32).reshape(-1)[:action_dim]


def _build_gt_chunk(actions_array: np.ndarray, start_idx: int, horizon: int) -> np.ndarray:
    end_idx = min(start_idx + horizon, len(actions_array))
    chunk = actions_array[start_idx:end_idx]
    if len(chunk) < horizon:
        pad = np.repeat(chunk[-1:], horizon - len(chunk), axis=0)
        chunk = np.concatenate([chunk, pad], axis=0)
    return chunk


def _group_mse(per_dim_mse: np.ndarray) -> dict[str, float]:
    d = int(per_dim_mse.shape[0])
    groups: dict[str, float] = {}
    if d >= 7:
        groups["left_arm[0:7]"] = float(np.mean(per_dim_mse[0:7]))
    if d >= 14:
        groups["right_arm[7:14]"] = float(np.mean(per_dim_mse[7:14]))
    if d >= 16:
        groups["grippers[14:16]"] = float(np.mean(per_dim_mse[14:16]))
    if d >= 20:
        groups["torso[16:20]"] = float(np.mean(per_dim_mse[16:20]))
    if d > 20:
        groups[f"extra[20:{d}]"] = float(np.mean(per_dim_mse[20:d]))
    if not groups:
        return {"all": float(np.mean(per_dim_mse))}
    return groups


def _dim_names(action_dim: int) -> list[str]:
    if action_dim == 20:
        return [
            "left_arm_0",
            "left_arm_1",
            "left_arm_2",
            "left_arm_3",
            "left_arm_4",
            "left_arm_5",
            "left_arm_6",
            "right_arm_0",
            "right_arm_1",
            "right_arm_2",
            "right_arm_3",
            "right_arm_4",
            "right_arm_5",
            "right_arm_6",
            "left_gripper",
            "right_gripper",
            "torso_0",
            "torso_1",
            "torso_2",
            "torso_3",
        ]
    base_names = [
        "left_arm_0",
        "left_arm_1",
        "left_arm_2",
        "left_arm_3",
        "left_arm_4",
        "left_arm_5",
        "left_arm_6",
        "right_arm_0",
        "right_arm_1",
        "right_arm_2",
        "right_arm_3",
        "right_arm_4",
        "right_arm_5",
        "right_arm_6",
        "left_gripper",
        "right_gripper",
        "torso_0",
        "torso_1",
        "torso_2",
        "torso_3",
        "action_21",
        "action_22",
        "action_23",
    ]
    if action_dim <= len(base_names):
        return base_names[:action_dim]
    extra = [f"action_{i:02d}" for i in range(len(base_names), action_dim)]
    return base_names + extra


def _episode_relative_indices(dataset: LeRobotDataset) -> list[list[int]]:
    """Return one list of hf_dataset row indices per episode_index, ordered by time."""
    ep_to_pairs: dict[int, list[tuple[int, int]]] = defaultdict(list)
    n = len(dataset)
    for rel in range(n):
        item = dataset[rel]
        ep = int(item["episode_index"].item())
        if "frame_index" in item:
            fi = int(item["frame_index"].item())
        else:
            fi = rel
        ep_to_pairs[ep].append((fi, rel))
    out: list[list[int]] = []
    for ep in sorted(ep_to_pairs):
        pairs = sorted(ep_to_pairs[ep], key=lambda x: (x[0], x[1]))
        out.append([r for _, r in pairs])
    return out


def _build_policy_batch(
    policy: DM0Policy,
    items: list[dict],
    device: torch.device,
) -> dict[str, torch.Tensor | list[str]]:
    batch: dict[str, torch.Tensor | list[str]] = {"task": [str(it["task"]) for it in items]}
    states = torch.stack([it[OBS_STATE].detach().float() for it in items]).to(device)
    if states.dim() == 3:
        states = states[:, -1]
    batch[OBS_STATE] = states
    for key in policy._image_keys():
        imgs = torch.stack([it[key].detach().float() for it in items]).to(device)
        if imgs.dim() == 5:
            imgs = imgs[:, -1]
        batch[key] = imgs
    return batch


def _predict_batch(
    policy: DM0Policy,
    device: torch.device,
    dataset: LeRobotDataset,
    rel_indices: list[int],
) -> tuple[np.ndarray, float]:
    items = [dataset[i] for i in rel_indices]
    batch = _build_policy_batch(policy, items, device)
    t0 = time.monotonic()
    with torch.inference_mode():
        actions = policy.predict_action_chunk(batch)
    dt_ms = (time.monotonic() - t0) * 1000.0 / max(len(rel_indices), 1)
    arr = actions.detach().float().cpu().numpy()
    return arr, dt_ms


def evaluate_checkpoint(
    args: argparse.Namespace,
    checkpoint: str,
    dataset: LeRobotDataset,
    episodes: list[list[int]],
    device: torch.device,
) -> dict:
    policy = DM0Policy.from_pretrained(
        checkpoint,
        strict=False,
    )
    policy.eval()
    policy.to(device)

    action_dim = int(policy._original_action_dim)
    probe_idx = episodes[0][0]
    probe_item = dataset[probe_idx]
    for k in policy._image_keys():
        if k not in probe_item:
            raise KeyError(
                f"Policy expects visual key {k!r} missing in dataset frame. "
                f"Frame keys (sample): {sorted(probe_item)}"
            )

    per_episode_mse: list[float] = []
    all_mse: list[float] = []
    per_dim_mse_all: list[np.ndarray] = []
    infer_times_ms: list[float] = []
    total_samples = 0
    eval_horizon_used: int | None = None

    for ep_idx, rel_list in enumerate(episodes):
        actions_array = np.stack(
            [_action_to_numpy(dataset[r], action_dim) for r in rel_list],
            axis=0,
        )
        upper = len(rel_list) - args.action_horizon
        eval_positions = list(range(0, max(upper, 0), args.sample_interval))
        if not eval_positions and rel_list:
            eval_positions = [0]

        ep_mse_list: list[float] = []
        for i in range(0, len(eval_positions), args.batch_size):
            pos_batch = eval_positions[i : i + args.batch_size]
            rel_batch = [rel_list[p] for p in pos_batch]
            pred_arr, infer_ms = _predict_batch(policy, device, dataset, rel_batch)
            infer_times_ms.append(infer_ms)
            if pred_arr.ndim != 3:
                raise RuntimeError(f"Unexpected prediction shape: {pred_arr.shape}")
            pred_arr = pred_arr[:, :, :action_dim]

            eval_horizon = min(args.action_horizon, pred_arr.shape[1])
            if eval_horizon_used is None:
                eval_horizon_used = eval_horizon
            gt_chunks = np.stack(
                [_build_gt_chunk(actions_array, p, eval_horizon) for p in pos_batch],
                axis=0,
            )
            pred_chunks = pred_arr[:, :eval_horizon, :]
            sq = (pred_chunks - gt_chunks) ** 2
            mse_each = np.mean(sq, axis=(1, 2))
            dim_mse_each = np.mean(sq, axis=1)
            all_mse.extend(mse_each.tolist())
            ep_mse_list.extend(mse_each.tolist())
            per_dim_mse_all.extend([d for d in dim_mse_each])
            total_samples += len(pos_batch)

        ep_mean = float(np.mean(ep_mse_list)) if ep_mse_list else float("nan")
        per_episode_mse.append(ep_mean)
        print(f"  Episode {ep_idx}: MSE = {ep_mean:.6f} ({len(eval_positions)} samples)")

    overall_mse = float(np.mean(all_mse)) if all_mse else float("nan")
    per_dim_mse = (
        np.mean(np.stack(per_dim_mse_all, axis=0), axis=0)
        if per_dim_mse_all
        else np.full((action_dim,), np.nan, dtype=np.float32)
    )
    group_mse = _group_mse(per_dim_mse)
    avg_infer_ms = float(np.mean(infer_times_ms)) if infer_times_ms else float("nan")

    ns = policy.config.norm_stats_path

    print(f"\n--- Results for {checkpoint} ---")
    print(f"Overall MSE:        {overall_mse:.6f}")
    print(f"Avg infer time:     {avg_infer_ms:.1f} ms")
    print(f"Total samples:      {total_samples}")
    print("\nPer-dimension MSE:")
    for i, (name, mse_val) in enumerate(zip(_dim_names(action_dim), per_dim_mse.tolist())):
        print(f"  [{i:2d}] {name:16s}: {mse_val:.6f}")
    print("\nPer-group MSE:")
    for k, v in group_mse.items():
        print(f"  {k:16s}: {v:.6f}")

    return {
        "mode": "abs_action_openloop_lerobot_dataset_dm0_predict_action_chunk",
        "dataset_repo": args.dataset_repo,
        "dataset_root": str(Path(args.dataset_root).resolve()) if args.dataset_root else None,
        "checkpoint": checkpoint,
        "norm_stats": ns if ns else None,
        "num_episodes": len(episodes),
        "num_samples": total_samples,
        "batch_size": args.batch_size,
        "sample_interval": args.sample_interval,
        "action_dim": action_dim,
        "action_horizon_requested": args.action_horizon,
        "action_horizon_used": int(eval_horizon_used or 0),
        "mse_abs_action": overall_mse,
        "rmse_abs_action": float(np.sqrt(overall_mse)) if overall_mse == overall_mse else float("nan"),
        "avg_infer_ms": avg_infer_ms,
        "per_episode_mse": per_episode_mse,
        "per_dim_mse": per_dim_mse.tolist(),
        "per_group_mse": group_mse,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DM0 open-loop MSE on LeRobotDataset using DM0Policy.predict_action_chunk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        nargs="+",
        required=True,
        help="LeRobot DM0 checkpoint dir(s) (e.g. .../checkpoints/last/pretrained_model).",
    )
    p.add_argument("--dataset-repo", type=str, required=True, help="LeRobot dataset repo_id.")
    p.add_argument("--dataset-root", type=str, default=None, help="Local root for the dataset cache.")
    p.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of episode_index values to load.",
    )
    p.add_argument("--revision", type=str, default=None, help="Dataset / hub revision.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap total frames loaded across episodes (head of episode list), like r1_openloop_eval.",
    )
    p.add_argument(
        "--action-horizon",
        type=int,
        default=50,
        help="MSE horizon in steps (clipped to model chunk length).",
    )
    p.add_argument(
        "--sample-interval",
        type=int,
        default=10,
        help="Stride between open-loop start frames inside each episode.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference.",
    )
    p.add_argument(
        "--no-download-videos",
        action="store_true",
        help="Pass download_videos=False to LeRobotDataset (only if frames already local).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    dataset = LeRobotDataset(
        repo_id=args.dataset_repo,
        root=args.dataset_root,
        episodes=args.episodes,
        revision=args.revision,
        download_videos=not args.no_download_videos,
    )
    episodes = _episode_relative_indices(dataset)
    if not episodes:
        raise ValueError("No episodes found in dataset.")

    if args.max_samples is not None:
        remaining = args.max_samples
        trimmed: list[list[int]] = []
        for ep in episodes:
            if remaining <= 0:
                break
            take = ep[: min(len(ep), remaining)]
            if take:
                trimmed.append(take)
                remaining -= len(take)
        episodes = trimmed
    if not episodes:
        raise ValueError("No episodes left after --max-samples.")

    # Smoke: first frame has required keys (fail fast before long eval)
    probe = dataset[episodes[0][0]]
    if OBS_STATE not in probe:
        raise KeyError(f"Dataset item missing {OBS_STATE!r}. Keys: {sorted(probe)}")

    all_results: list[dict] = []
    for ckpt in args.checkpoint:
        print(f"\n{'=' * 60}")
        print(f"Evaluating checkpoint: {ckpt}")
        print(f"{'=' * 60}")
        result = evaluate_checkpoint(args, ckpt, dataset, episodes, device)
        all_results.append(result)
        print("\nJSON summary:")
        print(json.dumps(result, indent=2))

    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("Checkpoint comparison")
        print(f"{'=' * 60}")
        print(f"{'Checkpoint':<55s} {'MSE':>10s} {'Infer(ms)':>10s}")
        print("-" * 80)
        for r in all_results:
            ckpt_name = Path(r["checkpoint"]).name
            print(f"{ckpt_name:<55s} {r['mse_abs_action']:>10.6f} {r['avg_infer_ms']:>10.1f}")
        best = min(all_results, key=lambda x: x["mse_abs_action"])
        print(f"\nBest checkpoint: {best['checkpoint']} (MSE = {best['mse_abs_action']:.6f})")


if __name__ == "__main__":
    main()
