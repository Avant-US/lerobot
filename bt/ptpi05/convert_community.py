"""
Download and convert episodes from HuggingFaceVLA/community_dataset_v3 to LeRobot v3.0.

The community_dataset_v3 repo contains ~791 sub-datasets in v2.1 format.
Each sub-dataset lives under {user}/{dataset_name}/ with:
  - data/chunk-000/episode_XXXXXX.parquet  (state, action, timestamps)
  - meta/info.json, episodes.jsonl, tasks.jsonl
  - videos/chunk-000/{camera_key}/episode_XXXXXX.mp4

This script:
  1. Discovers sub-datasets and groups them by compatible feature schema
  2. Randomly samples N episodes across sub-datasets
  3. Downloads parquet + video files, decodes video frames
  4. Writes a unified v3.0 dataset with images embedded in parquet

Usage:
    python -m bt.ptpi05.convert_community --num-episodes 37 --output-dir data/community_pt_v30
    python -m bt.ptpi05.convert_community --num-episodes 5 --output-dir data/community_pt_v30  # quick test
"""

import argparse
import json
import logging
import random
import shutil
from collections import defaultdict
from pathlib import Path

import av
import datasets
import numpy as np
import pandas as pd
from PIL import Image
from huggingface_hub import HfApi, hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ID = "HuggingFaceVLA/community_dataset_v3"
IMAGE_SIZE = 224
DEFAULT_FPS = 30


def list_subdatasets(api: HfApi) -> list[str]:
    """Discover all sub-dataset prefixes in the monorepo."""
    files = api.list_repo_files(REPO_ID, repo_type="dataset")
    prefixes = set()
    for f in files:
        parts = f.split("/")
        if len(parts) >= 3 and parts[-1] == "info.json" and "meta" in parts:
            prefix = "/".join(parts[: parts.index("meta")])
            prefixes.add(prefix)
    return sorted(prefixes)


def download_info(api: HfApi, prefix: str) -> dict | None:
    """Download and parse info.json for a sub-dataset."""
    try:
        path = hf_hub_download(
            REPO_ID, f"{prefix}/meta/info.json", repo_type="dataset"
        )
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to download info for {prefix}: {e}")
        return None


def classify_features(info: dict) -> tuple[int, int, list[str]] | None:
    """Extract (state_dim, action_dim, camera_keys) from info.json features."""
    features = info.get("features", {})
    state_dim = action_dim = 0
    cameras = []
    for key, ft in features.items():
        if key == "observation.state":
            shape = ft.get("shape", [])
            state_dim = shape[0] if shape else 0
        elif key == "action":
            shape = ft.get("shape", [])
            action_dim = shape[0] if shape else 0
        elif key.startswith("observation.images.") and ft.get("dtype") in ("video", "image"):
            cameras.append(key)
    if state_dim > 0 and action_dim > 0 and cameras:
        return (state_dim, action_dim, cameras)
    return None


def download_episode_parquet(prefix: str, ep_idx: int) -> Path:
    chunk_idx = ep_idx // 1000
    filename = f"{prefix}/data/chunk-{chunk_idx:03d}/episode_{ep_idx:06d}.parquet"
    return Path(hf_hub_download(REPO_ID, filename, repo_type="dataset"))


def download_episode_video(prefix: str, camera_key: str, ep_idx: int) -> Path:
    chunk_idx = ep_idx // 1000
    filename = f"{prefix}/videos/chunk-{chunk_idx:03d}/{camera_key}/episode_{ep_idx:06d}.mp4"
    return Path(hf_hub_download(REPO_ID, filename, repo_type="dataset"))


def download_tasks(prefix: str) -> dict[int, str]:
    """Download tasks.jsonl and return {task_index: task_text}."""
    try:
        path = hf_hub_download(REPO_ID, f"{prefix}/meta/tasks.jsonl", repo_type="dataset")
        import jsonlines
        with jsonlines.open(path) as reader:
            return {t["task_index"]: t["task"] for t in reader}
    except Exception:
        return {0: "manipulation task"}


def decode_video_frames(video_path: Path, expected_frames: int | None = None) -> list[Image.Image]:
    """Decode all frames from a video file, resize to IMAGE_SIZE."""
    frames = []
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    for frame in container.decode(stream):
        img = frame.to_image().resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        frames.append(img)
    container.close()
    if expected_frames is not None and len(frames) != expected_frames:
        logger.warning(
            f"Video {video_path.name}: expected {expected_frames} frames, got {len(frames)}"
        )
        if len(frames) > expected_frames:
            frames = frames[:expected_frames]
        elif len(frames) < expected_frames:
            while len(frames) < expected_frames:
                frames.append(frames[-1] if frames else Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE)))
    return frames


def convert_community(
    output_dir: str = "data/community_pt_v30",
    num_episodes: int = 37,
    seed: int = 42,
    max_probe: int = 120,
):
    random.seed(seed)
    np.random.seed(seed)
    root = Path(output_dir)
    if root.exists():
        shutil.rmtree(root)

    api = HfApi()

    # 1. Discover sub-datasets
    logger.info("Discovering sub-datasets...")
    all_prefixes = list_subdatasets(api)
    logger.info(f"Found {len(all_prefixes)} sub-datasets")

    # 2. Probe a sample to find compatible schemas
    probe_prefixes = random.sample(all_prefixes, min(max_probe, len(all_prefixes)))
    schema_groups: dict[tuple[int, int], list[tuple[str, dict, list[str]]]] = defaultdict(list)

    logger.info(f"Probing {len(probe_prefixes)} sub-datasets for compatible schemas...")
    for i, prefix in enumerate(probe_prefixes):
        info = download_info(api, prefix)
        if info is None:
            continue
        result = classify_features(info)
        if result is None:
            continue
        state_dim, action_dim, cameras = result
        total_eps = info.get("total_episodes", 0)
        if total_eps <= 0:
            continue
        schema_groups[(state_dim, action_dim)].append((prefix, info, cameras))
        if (i + 1) % 20 == 0:
            logger.info(f"  Probed {i+1}/{len(probe_prefixes)}, found {sum(len(v) for v in schema_groups.values())} compatible")

    if not schema_groups:
        raise RuntimeError("No compatible sub-datasets found")

    # Pick the largest group
    best_schema = max(schema_groups.keys(), key=lambda k: len(schema_groups[k]))
    group = schema_groups[best_schema]
    state_dim, action_dim = best_schema
    logger.info(
        f"Best schema: state_dim={state_dim}, action_dim={action_dim}, "
        f"{len(group)} sub-datasets"
    )

    # 3. Plan episode downloads: 1 episode per sub-dataset, spread across sub-datasets
    random.shuffle(group)
    download_plan: list[tuple[str, int, dict, str]] = []  # (prefix, ep_idx, info, camera_key)

    for prefix, info, cameras in group:
        if len(download_plan) >= num_episodes:
            break
        total_eps = info.get("total_episodes", 1)
        ep_idx = random.randint(0, total_eps - 1)
        camera_key = cameras[0]
        download_plan.append((prefix, ep_idx, info, camera_key))

    if len(download_plan) < num_episodes:
        for prefix, info, cameras in group:
            if len(download_plan) >= num_episodes:
                break
            total_eps = info.get("total_episodes", 1)
            if total_eps < 2:
                continue
            for _ in range(min(total_eps - 1, num_episodes - len(download_plan))):
                ep_idx = random.randint(0, total_eps - 1)
                existing = {(p, e) for p, e, _, _ in download_plan}
                if (prefix, ep_idx) not in existing:
                    download_plan.append((prefix, ep_idx, info, cameras[0]))

    actual_count = len(download_plan)
    logger.info(f"Planned {actual_count} episode downloads")

    # 4. Download and process episodes
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)

    all_data: dict[str, list] = {
        "observation.state": [],
        "observation.images.top": [],
        "action": [],
        "timestamp": [],
        "frame_index": [],
        "episode_index": [],
        "index": [],
        "task_index": [],
    }

    episodes_meta = []
    task_set: dict[str, int] = {}
    global_frame_idx = 0
    all_states = []
    all_actions = []

    for ep_i, (prefix, ep_idx, info, camera_key) in enumerate(download_plan):
        logger.info(f"[{ep_i+1}/{actual_count}] Downloading {prefix} episode {ep_idx}...")

        try:
            pq_path = download_episode_parquet(prefix, ep_idx)
            df = pd.read_parquet(pq_path)
            ep_len = len(df)

            video_path = download_episode_video(prefix, camera_key, ep_idx)
            frames = decode_video_frames(video_path, expected_frames=ep_len)
        except Exception as e:
            logger.warning(f"  Failed: {e}, skipping")
            continue

        tasks = download_tasks(prefix)
        task_idx_orig = int(df["task_index"].iloc[0]) if "task_index" in df.columns else 0
        task_text = tasks.get(task_idx_orig, "manipulation task")
        if task_text not in task_set:
            task_set[task_text] = len(task_set)
        task_idx_new = task_set[task_text]

        fps = info.get("fps", DEFAULT_FPS)

        for frame_i in range(ep_len):
            state = np.array(df["observation.state"].iloc[frame_i], dtype=np.float32)
            action = np.array(df["action"].iloc[frame_i], dtype=np.float32)

            if len(state) < state_dim:
                state = np.pad(state, (0, state_dim - len(state)))
            else:
                state = state[:state_dim]
            if len(action) < action_dim:
                action = np.pad(action, (0, action_dim - len(action)))
            else:
                action = action[:action_dim]

            all_states.append(state)
            all_actions.append(action)

            all_data["observation.state"].append(state.tolist())
            all_data["observation.images.top"].append(frames[frame_i])
            all_data["action"].append(action.tolist())
            all_data["timestamp"].append(float(frame_i) / fps)
            all_data["frame_index"].append(frame_i)
            all_data["episode_index"].append(ep_i)
            all_data["index"].append(global_frame_idx)
            all_data["task_index"].append(task_idx_new)
            global_frame_idx += 1

        from_idx = global_frame_idx - ep_len
        episodes_meta.append({
            "episode_index": ep_i,
            "tasks": json.dumps([task_text]),
            "length": ep_len,
            "dataset_from_index": from_idx,
            "dataset_to_index": global_frame_idx,
            "data/chunk_index": 0,
            "data/file_index": 0,
        })

        logger.info(f"  OK: {ep_len} frames, source={prefix}")

    total_episodes = len(episodes_meta)
    total_frames = global_frame_idx
    logger.info(f"Total: {total_episodes} episodes, {total_frames} frames")

    if total_episodes == 0:
        raise RuntimeError("No episodes successfully downloaded")

    # 5. Write data parquet
    logger.info("Writing data parquet...")
    hf_features = datasets.Features({
        "observation.state": datasets.Sequence(
            length=state_dim, feature=datasets.Value("float32")
        ),
        "observation.images.top": datasets.Image(),
        "action": datasets.Sequence(
            length=action_dim, feature=datasets.Value("float32")
        ),
        "timestamp": datasets.Value("float32"),
        "frame_index": datasets.Value("int64"),
        "episode_index": datasets.Value("int64"),
        "index": datasets.Value("int64"),
        "task_index": datasets.Value("int64"),
    })

    hf_ds = datasets.Dataset.from_dict(all_data, features=hf_features)
    hf_ds.to_parquet(data_dir / "file-000.parquet")

    # 6. Write info.json
    v30_info = {
        "codebase_version": "v3.0",
        "robot_type": "community_mixed",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(task_set),
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 0,
        "fps": DEFAULT_FPS,
        "splits": {"train": f"0:{total_frames}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": None,
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [state_dim],
                "names": None,
            },
            "observation.images.top": {
                "dtype": "image",
                "shape": [3, IMAGE_SIZE, IMAGE_SIZE],
                "names": ["channels", "height", "width"],
            },
            "action": {
                "dtype": "float32",
                "shape": [action_dim],
                "names": None,
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(v30_info, f, indent=4)

    # 7. Write tasks.parquet
    task_items = sorted(task_set.items(), key=lambda x: x[1])
    tasks_df = pd.DataFrame(
        {"task_index": [idx for _, idx in task_items]},
        index=pd.Index([text for text, _ in task_items], name="task"),
    )
    tasks_df.to_parquet(meta_dir / "tasks.parquet")

    # 8. Write episodes parquet
    episodes_dir = meta_dir / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    ep_ds = datasets.Dataset.from_dict(
        {k: [r[k] for r in episodes_meta] for k in episodes_meta[0]}
    )
    ep_ds.to_parquet(episodes_dir / "file-000.parquet")

    # 9. Write stats.json
    all_states_arr = np.array(all_states)
    all_actions_arr = np.array(all_actions)

    def compute_stats(arr):
        return {
            "mean": arr.mean(axis=0).tolist(),
            "std": np.clip(arr.std(axis=0), 1e-6, None).tolist(),
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "q01": np.percentile(arr, 1, axis=0).tolist(),
            "q99": np.percentile(arr, 99, axis=0).tolist(),
            "count": [len(arr)],
        }

    stats = {
        "observation.state": compute_stats(all_states_arr),
        "action": compute_stats(all_actions_arr),
        "observation.images.top": {
            "mean": [[[0.485]], [[0.456]], [[0.406]]],
            "std": [[[0.229]], [[0.224]], [[0.225]]],
            "min": [[[0.0]], [[0.0]], [[0.0]]],
            "max": [[[1.0]], [[1.0]], [[1.0]]],
            "q01": [[[0.0]], [[0.0]], [[0.0]]],
            "q99": [[[1.0]], [[1.0]], [[1.0]]],
            "count": [total_frames],
        },
    }
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    logger.info(f"Conversion complete: {root}")
    logger.info(f"  Episodes: {total_episodes}, Frames: {total_frames}")
    logger.info(f"  State dim: {state_dim}, Action dim: {action_dim}")
    logger.info(f"  Tasks: {len(task_set)}")
    return str(root)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Convert HuggingFaceVLA/community_dataset_v3 episodes to LeRobot v3.0"
    )
    p.add_argument("--output-dir", default="data/community_pt_v30")
    p.add_argument("--num-episodes", type=int, default=37)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-probe", type=int, default=120,
                   help="Max sub-datasets to probe for schema compatibility")
    args = p.parse_args()
    convert_community(args.output_dir, args.num_episodes, args.seed, args.max_probe)
