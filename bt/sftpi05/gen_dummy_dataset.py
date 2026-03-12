"""
Generate a minimal dummy LeRobot-v3.0 dataset for PI05 fine-tuning testing.

Creates a small dataset loadable by LeRobotDataset with:
  - Random state vectors
  - Random RGB images (embedded in parquet via HF datasets.Image)
  - Random action vectors
  - Task descriptions
  - Proper v3.0 meta files (info.json, tasks.parquet, episodes parquet, stats.json)

Usage:
    python -m bt.sftpi05.gen_dummy_dataset [--output-dir /path/to/dir] [--num-episodes 3] [--frames-per-ep 60]
"""

import argparse
import json
import shutil
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
from PIL import Image

STATE_DIM = 14
ACTION_DIM = 7
IMAGE_SIZE = 224
FPS = 10
TASK_TEXT = "pick up the object and place it in the bin"


def generate_dummy_dataset(
    output_dir: str = "data/dummy_pi05_sft",
    num_episodes: int = 3,
    frames_per_ep: int = 60,
):
    root = Path(output_dir)
    if root.exists():
        shutil.rmtree(root)

    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    total_frames = num_episodes * frames_per_ep

    # ======= Collect all frame data =======
    all_states = []
    all_actions = []

    data_dict = {
        "observation.state": [],
        "observation.images.top": [],
        "action": [],
        "timestamp": [],
        "frame_index": [],
        "episode_index": [],
        "index": [],
        "task_index": [],
    }

    global_idx = 0
    for ep in range(num_episodes):
        for frame in range(frames_per_ep):
            state = np.random.randn(STATE_DIM).astype(np.float32)
            action = np.random.randn(ACTION_DIM).astype(np.float32)
            all_states.append(state)
            all_actions.append(action)

            img_array = np.random.randint(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            pil_img = Image.fromarray(img_array)

            data_dict["observation.state"].append(state.tolist())
            data_dict["observation.images.top"].append(pil_img)
            data_dict["action"].append(action.tolist())
            data_dict["timestamp"].append(float(frame) / FPS)
            data_dict["frame_index"].append(frame)
            data_dict["episode_index"].append(ep)
            data_dict["index"].append(global_idx)
            data_dict["task_index"].append(0)
            global_idx += 1

    # ======= Write data parquet (with embedded images) =======
    hf_features = datasets.Features({
        "observation.state": datasets.Sequence(length=STATE_DIM, feature=datasets.Value("float32")),
        "observation.images.top": datasets.Image(),
        "action": datasets.Sequence(length=ACTION_DIM, feature=datasets.Value("float32")),
        "timestamp": datasets.Value("float32"),
        "frame_index": datasets.Value("int64"),
        "episode_index": datasets.Value("int64"),
        "index": datasets.Value("int64"),
        "task_index": datasets.Value("int64"),
    })

    hf_ds = datasets.Dataset.from_dict(data_dict, features=hf_features)
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    hf_ds.to_parquet(data_dir / "file-000.parquet")

    # ======= info.json =======
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [STATE_DIM],
            "names": None,
        },
        "observation.images.top": {
            "dtype": "image",
            "shape": [3, IMAGE_SIZE, IMAGE_SIZE],
            "names": ["channels", "height", "width"],
        },
        "action": {
            "dtype": "float32",
            "shape": [ACTION_DIM],
            "names": None,
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }

    info = {
        "codebase_version": "v3.0",
        "robot_type": "dummy_robot",
        "total_episodes": num_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": FPS,
        "splits": {"train": f"0:{total_frames}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": None,
        "features": features,
    }

    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    # ======= tasks.parquet (v3.0: task text as index, task_index as column) =======
    tasks_df = pd.DataFrame(
        {"task_index": [0]},
        index=pd.Index([TASK_TEXT], name="task"),
    )
    tasks_df.to_parquet(meta_dir / "tasks.parquet")

    # ======= episodes parquet (v3.0: meta/episodes/chunk-000/file-000.parquet) =======
    episodes_dir = meta_dir / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    ep_records = []
    g_idx = 0
    for ep in range(num_episodes):
        from_idx = g_idx
        to_idx = from_idx + frames_per_ep
        ep_records.append({
            "episode_index": ep,
            "tasks": json.dumps([TASK_TEXT]),
            "length": frames_per_ep,
            "dataset_from_index": from_idx,
            "dataset_to_index": to_idx,
            "data/chunk_index": 0,
            "data/file_index": 0,
        })
        g_idx = to_idx

    ep_ds = datasets.Dataset.from_dict({
        k: [r[k] for r in ep_records] for k in ep_records[0]
    })
    ep_ds.to_parquet(episodes_dir / "file-000.parquet")

    # ======= stats.json =======
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

    print(f"Dummy v3.0 dataset generated at: {root}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Frames per episode: {frames_per_ep}")
    print(f"  Total frames: {total_frames}")
    print(f"  State dim: {STATE_DIM}, Action dim: {ACTION_DIM}")
    return str(root)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate dummy PI05 SFT dataset (v3.0)")
    ap.add_argument("--output-dir", type=str, default="data/dummy_pi05_sft")
    ap.add_argument("--num-episodes", type=int, default=3)
    ap.add_argument("--frames-per-ep", type=int, default=60)
    args = ap.parse_args()
    generate_dummy_dataset(args.output_dir, args.num_episodes, args.frames_per_ep)
