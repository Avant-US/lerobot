#!/usr/bin/env python3
"""GR00T-style open-loop trajectory plots for DM0 on ``LeRobotDataset``.

Replays the same *chunk re-inference* pattern as ``calc_mse_for_single_trajectory``:
every ``action_horizon`` steps, run ``predict_action_chunk`` once and append ``H`` rows
of predicted vs ground-truth actions (GT from the episode trajectory, padded like
``dm0_openloop_eval``).

``plot_dm0_openloop_trajectory`` uses one **shared y-axis range** across all action
dimensions (from min/max of gt + pred, and state when shapes match), so curves are
easier to compare. Use ``--no-unify-y-scale`` for per-subplot auto scaling.

Pass ``--ylim-min`` and ``--ylim-max`` together to force the same y-range on every
subplot (overrides auto shared limits).

Example::

    python lerobot/bt/dm0/dm0_openloop_policy.py \\
        --checkpoint /path/to/pretrained_model \\
        --dataset-repo local/my_dataset \\
        --dataset-root /data/my_dataset \\
        --episode-index 0 \\
        --steps 300 \\
        --action-horizon 16 \\
        --save-plot /tmp/dm0_openloop.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_DM0_DIR = Path(__file__).resolve().parent
_LEROBOT_ROOT = _DM0_DIR.parents[2]
_SRC = _LEROBOT_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_DM0_DIR) not in sys.path:
    sys.path.insert(0, str(_DM0_DIR))

import dm0_openloop_eval as ole  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.dm0.modeling_dm0 import DM0Policy  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402


def _state_to_numpy_1d(item: dict) -> np.ndarray:
    s = item[OBS_STATE]
    if isinstance(s, torch.Tensor):
        s = s.detach().float().cpu().numpy()
    return np.asarray(s, dtype=np.float32).reshape(-1)


def collect_gr00t_style_openloop(
    policy: object,
    dm0: DM0Policy,
    dataset: LeRobotDataset,
    rel_list: list[int],
    *,
    steps: int,
    action_horizon: int,
    device: torch.device,
    plot_state: bool = False,
) -> dict[str, Any]:
    """Build state / GT / pred arrays in the same layout as GR00T's eval helper.

    At ``t % action_horizon == 0`` (within the episode), runs ``predict_action_chunk``
    on ``dataset[rel_list[t]]`` and appends ``H`` rows where ``H`` is
    ``min(action_horizon, model_chunk_len)``. GT rows come from
    ``_build_gt_chunk`` on the episode's ``ACTION`` trajectory.

    Arrays are truncated to ``steps`` rows (like GR00T's ``[:steps]`` slice).
    """
    action_dim = int(dm0._original_action_dim)
    actions_array = np.stack(
        [ole._action_to_numpy(dataset[r], action_dim) for r in rel_list],
        axis=0,
    )
    ep_len = len(rel_list)
    steps_use = min(int(steps), ep_len)

    state_joints_across_time: list[np.ndarray] = []
    gt_action_across_time: list[np.ndarray] = []
    pred_action_across_time: list[np.ndarray] = []

    for step_count in range(steps_use):
        if step_count % action_horizon != 0:
            continue

        item = dataset[rel_list[step_count]]
        state_vec = _state_to_numpy_1d(item) if plot_state else None
        batch = ole._build_policy_batch(dm0, [item], device)
        with torch.inference_mode():
            pred = dm0.predict_action_chunk(batch)
        pred_np = pred.detach().float().cpu().numpy()[0, :, :action_dim]
        h_model = int(pred_np.shape[0])
        h = min(int(action_horizon), h_model)
        pred_np = pred_np[:h, :]
        gt_chunk = ole._build_gt_chunk(actions_array, step_count, h)

        for j in range(h):
            if plot_state and state_vec is not None:
                state_joints_across_time.append(state_vec)
            pred_action_across_time.append(pred_np[j].astype(np.float32, copy=False))
            gt_action_across_time.append(gt_chunk[j].astype(np.float32, copy=False))

    if not gt_action_across_time:
        raise ValueError(
            "No predictions collected (empty episode, steps_use==0, or bad horizon)."
        )

    gt_action_across_time_a = np.stack(gt_action_across_time, axis=0)
    pred_action_across_time_a = np.stack(pred_action_across_time, axis=0)
    assert gt_action_across_time_a.shape == pred_action_across_time_a.shape

    n = min(int(steps), gt_action_across_time_a.shape[0])
    gt_action_across_time_a = gt_action_across_time_a[:n]
    pred_action_across_time_a = pred_action_across_time_a[:n]

    if plot_state and state_joints_across_time:
        state_joints_across_time_a = np.stack(state_joints_across_time, axis=0)[:n]
    else:
        state_joints_across_time_a = np.empty((0, 0), dtype=np.float32)

    mse = float(np.mean((gt_action_across_time_a - pred_action_across_time_a) ** 2))
    if np.isnan(pred_action_across_time_a).any():
        raise ValueError("Predicted action contains NaN.")

    dim_names = ole._dim_names(int(gt_action_across_time_a.shape[1]))
    return {
        "state_joints_across_time": state_joints_across_time_a,
        "gt_action_across_time": gt_action_across_time_a,
        "pred_action_across_time": pred_action_across_time_a,
        "dim_names": dim_names,
        "traj_id": 0,
        "mse": mse,
        "action_dim": int(gt_action_across_time_a.shape[1]),
        "action_horizon": int(action_horizon),
        "steps": int(steps),
        "steps_plotted": int(n),
    }


def _unified_ylim(
    gt: np.ndarray,
    pred: np.ndarray,
    state: np.ndarray,
    *,
    margin_frac: float,
) -> tuple[float, float] | None:
    """Single y-range over all timesteps and dimensions (and state if same shape as gt)."""
    parts: list[np.ndarray] = [gt.reshape(-1), pred.reshape(-1)]
    if state.size and state.shape == gt.shape:
        parts.append(state.reshape(-1))
    stacked = np.concatenate(parts, axis=0)
    lo = float(np.nanmin(stacked))
    hi = float(np.nanmax(stacked))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if hi <= lo:
        mid = lo
        return (mid - 1.0, mid + 1.0)
    span = hi - lo
    pad = span * float(margin_frac) if span > 0 else 0.1
    return (lo - pad, hi + pad)


def plot_dm0_openloop_trajectory(
    info: dict[str, Any],
    *,
    save_plot_path: str | Path | None = None,
    show: bool = False,
    title_suffix: str = "",
    unify_y_scale: bool = True,
    y_margin_frac: float = 0.05,
    ylim_fixed: tuple[float, float] | None = None,
) -> None:
    """Per-dimension matplotlib figure in the style of GR00T's ``plot_trajectory``.

    If ``ylim_fixed`` is ``(y0, y1)``, every subplot uses that y-range (manual override).
    Otherwise, when ``unify_y_scale`` is true, y-range is computed from data.
    """
    if save_plot_path is not None:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    action_dim = int(info["action_dim"])
    state_joints_across_time: np.ndarray = info["state_joints_across_time"]
    gt_action_across_time: np.ndarray = info["gt_action_across_time"]
    pred_action_across_time: np.ndarray = info["pred_action_across_time"]
    dim_names: list[str] = info.get("dim_names") or [f"dim_{i}" for i in range(action_dim)]
    traj_id = int(info.get("traj_id", 0))
    mse = float(info["mse"])
    action_horizon = int(info["action_horizon"])
    steps = int(info["steps_plotted"] if "steps_plotted" in info else info["steps"])

    fig_h = max(4.0 * action_dim + 2.0, 6.0)
    fig, axes = plt.subplots(nrows=action_dim, ncols=1, figsize=(10, fig_h))
    axes_flat = np.atleast_1d(axes).ravel()

    plt.subplots_adjust(top=0.92, left=0.1, right=0.96, hspace=0.4)

    title = (
        f"DM0 open-loop — episode / traj id: {traj_id}\n"
        f"Unnormalized MSE: {mse:.6f}{title_suffix}"
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", color="#2E86AB", y=0.95)

    for i, ax in enumerate(axes_flat):
        if (
            state_joints_across_time.size
            and state_joints_across_time.shape == gt_action_across_time.shape
        ):
            ax.plot(state_joints_across_time[:, i], label="state (obs)", alpha=0.7)
        ax.plot(gt_action_across_time[:, i], label="gt action", linewidth=2)
        ax.plot(pred_action_across_time[:, i], label="pred action", linewidth=2)

        for j in range(0, steps, action_horizon):
            if j == 0:
                ax.plot(
                    j,
                    float(gt_action_across_time[j, i]),
                    "ro",
                    label="inference point",
                    markersize=6,
                )
            else:
                ax.plot(j, float(gt_action_across_time[j, i]), "ro", markersize=4)

        name = dim_names[i] if i < len(dim_names) else f"dim_{i}"
        ax.set_title(f"Action — {name}", fontsize=12, fontweight="bold", pad=10)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time step (padded chunk replay)", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)

    if ylim_fixed is not None:
        y0, y1 = float(ylim_fixed[0]), float(ylim_fixed[1])
        if not (np.isfinite(y0) and np.isfinite(y1)):
            raise ValueError(f"ylim_fixed must be finite; got {ylim_fixed!r}")
        if y0 >= y1:
            raise ValueError(f"ylim_fixed requires min < max; got {ylim_fixed!r}")
        for ax in axes_flat:
            ax.set_ylim(y0, y1)
            ax.set_ylabel("Value (manual y-range)", fontsize=10)
    elif unify_y_scale:
        ylim = _unified_ylim(
            gt_action_across_time,
            pred_action_across_time,
            state_joints_across_time,
            margin_frac=y_margin_frac,
        )
        if ylim is not None:
            y0, y1 = ylim
            for ax in axes_flat:
                ax.set_ylim(y0, y1)
                ax.set_ylabel("Value (shared scale)", fontsize=10)

    if save_plot_path is not None:
        out = Path(save_plot_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot to {out.resolve()}")
    elif show:
        plt.show()
    else:
        plt.close(fig)


def run_gr00t_style_plot(
    *,
    checkpoint: str,
    dataset_repo: str,
    dataset_root: str | None,
    episode_index: int,
    steps: int,
    action_horizon: int,
    save_plot_path: str | Path | None,
    show: bool,
    plot_state: bool,
    device: torch.device,
    peft_base_checkpoint: str | None,
    merge_lora: bool,
    revision: str | None,
    episodes_filter: list[int] | None,
    no_download_videos: bool,
    unify_y_scale: bool = True,
    y_margin_frac: float = 0.05,
    ylim_fixed: tuple[float, float] | None = None,
) -> dict[str, Any]:
    ole._sanitize_hf_cache_env()
    dataset_root_path: Path | None = None
    if dataset_root:
        dataset_root_path = Path(dataset_root).expanduser().resolve()
        if not dataset_root_path.is_dir():
            raise FileNotFoundError(f"--dataset-root is not a directory: {dataset_root_path}")
        os.environ.setdefault(
            "HF_DATASETS_CACHE",
            str(dataset_root_path / "_hf_datasets_parquet_cache"),
        )
    ole._ensure_hf_datasets_cache()
    if dataset_root_path is not None:
        fallback = dataset_root_path / "_hf_datasets_parquet_cache"
        dsc = os.environ.get("HF_DATASETS_CACHE")
        dsc_path = Path(dsc).expanduser() if dsc else None
        if not dsc or dsc_path is None or (dsc_path.exists() and not dsc_path.is_dir()):
            os.environ["HF_DATASETS_CACHE"] = str(fallback)
            fallback.mkdir(parents=True, exist_ok=True)

    dataset = LeRobotDataset(
        repo_id=dataset_repo,
        root=dataset_root,
        episodes=episodes_filter,
        revision=revision,
        download_videos=not no_download_videos,
    )
    episodes = ole._episode_relative_indices(dataset)
    if not episodes:
        raise ValueError("No episodes found in dataset.")
    if episode_index < 0 or episode_index >= len(episodes):
        raise IndexError(
            f"episode_index={episode_index} out of range (num episodes={len(episodes)})."
        )
    rel_list = episodes[episode_index]

    probe = dataset[rel_list[0]]
    if OBS_STATE not in probe:
        raise KeyError(f"Dataset item missing {OBS_STATE!r}. Keys: {sorted(probe)}")
    if ACTION not in probe:
        raise KeyError(f"Dataset item missing {ACTION!r}. Keys: {sorted(probe)}")

    policy = ole._load_policy(
        checkpoint,
        peft_base_override=peft_base_checkpoint,
        merge_lora=merge_lora,
    )
    dm0 = ole._unwrap_dm0(policy)
    policy.eval()
    policy.to(device)

    info = collect_gr00t_style_openloop(
        policy,
        dm0,
        dataset,
        rel_list,
        steps=steps,
        action_horizon=action_horizon,
        device=device,
        plot_state=plot_state,
    )
    info["traj_id"] = episode_index

    title_suffix = f"\ncheckpoint: {Path(checkpoint).name}"
    plot_dm0_openloop_trajectory(
        info,
        save_plot_path=save_plot_path,
        show=show,
        title_suffix=title_suffix,
        unify_y_scale=unify_y_scale,
        y_margin_frac=y_margin_frac,
        ylim_fixed=ylim_fixed,
    )

    summary = {
        "mse": info["mse"],
        "steps_plotted": info["steps_plotted"],
        "action_dim": info["action_dim"],
        "action_horizon": info["action_horizon"],
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "dataset_repo": dataset_repo,
        "episode_index": episode_index,
    }
    print(json.dumps(summary, indent=2))
    return info


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GR00T-style DM0 open-loop trajectory plot (chunk re-inference).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset-repo", type=str, required=True)
    p.add_argument("--dataset-root", type=str, default=None)
    p.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of episode_index values to load into LeRobotDataset.",
    )
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--episode-index", type=int, default=0, help="Which episode to plot.")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--action-horizon", type=int, default=16)
    p.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="If set, write PNG to this path (non-interactive backend).",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Show an interactive window (requires a display).",
    )
    p.add_argument("--plot-state", action="store_true")
    p.add_argument(
        "--no-unify-y-scale",
        action="store_true",
        help="Let each subplot use its own y auto-scale (default: one shared y-range for all rows).",
    )
    p.add_argument(
        "--y-margin-frac",
        type=float,
        default=0.05,
        help="Padding fraction applied below min and above max when using shared y-scale.",
    )
    p.add_argument(
        "--ylim-min",
        type=float,
        default=None,
        dest="ylim_min",
        help="Fixed y lower bound for every subplot (must pass --ylim-max too). Overrides auto shared scale.",
    )
    p.add_argument(
        "--ylim-max",
        type=float,
        default=None,
        dest="ylim_max",
        help="Fixed y upper bound for every subplot (must pass --ylim-min too). Overrides auto shared scale.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--peft-base-checkpoint", type=str, default=None)
    p.add_argument("--merge-lora", action="store_true")
    p.add_argument("--no-download-videos", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_cli()
    if not args.save_plot and not args.show:
        raise SystemExit("Pass --save-plot PATH and/or --show.")
    lo, hi = args.ylim_min, args.ylim_max
    if (lo is None) ^ (hi is None):
        raise SystemExit("Pass both --ylim-min and --ylim-max, or neither.")
    ylim_fixed: tuple[float, float] | None = None
    if lo is not None and hi is not None:
        ylim_fixed = (float(lo), float(hi))
        if ylim_fixed[0] >= ylim_fixed[1]:
            raise SystemExit("--ylim-min must be strictly less than --ylim-max.")
    run_gr00t_style_plot(
        checkpoint=args.checkpoint,
        dataset_repo=args.dataset_repo,
        dataset_root=args.dataset_root,
        episode_index=args.episode_index,
        steps=args.steps,
        action_horizon=args.action_horizon,
        save_plot_path=args.save_plot,
        show=bool(args.show),
        plot_state=bool(args.plot_state),
        device=torch.device(args.device),
        peft_base_checkpoint=args.peft_base_checkpoint,
        merge_lora=bool(args.merge_lora),
        revision=args.revision,
        episodes_filter=args.episodes,
        no_download_videos=bool(args.no_download_videos),
        unify_y_scale=not bool(args.no_unify_y_scale),
        y_margin_frac=float(args.y_margin_frac),
        ylim_fixed=ylim_fixed,
    )


if __name__ == "__main__":
    main()
