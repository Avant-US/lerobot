"""
Dataset adapter for physical-intelligence/libero (v2.0 format).

Loads the HuggingFace parquet-based dataset and presents it as a PyTorch
Dataset that produces batches compatible with PI05Policy.forward().

Feature mapping (libero v2.0 -> PI05 expected):
    image          -> observation.images.top
    wrist_image    -> observation.images.wrist
    state          -> observation.state
    actions        -> action
"""

import io
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

LIBERO_REPO = "physical-intelligence/libero"
LIBERO_REVISION = "main"

# v2.0 feature name -> PI05 canonical name
FEATURE_RENAME = {
    "image": "observation.images.top",
    "wrist_image": "observation.images.wrist",
    "state": "observation.state",
    "actions": "action",
}


class LiberoDataset(Dataset):
    """
    Loads physical-intelligence/libero (v2.0 parquet) and outputs dicts
    ready for the PI05 preprocessor.

    Each __getitem__ returns:
        observation.state   : (state_dim,) float32
        observation.images.top  : (3, H, W) float32 in [0,1]
        observation.images.wrist : (3, H, W) float32 in [0,1]
        action              : (chunk_size, action_dim) float32
        task                : str
    """

    def __init__(
        self,
        repo_id: str = LIBERO_REPO,
        revision: str = LIBERO_REVISION,
        episodes: list[int] | None = None,
        chunk_size: int = 50,
        root: str | None = None,
        max_episodes: int | None = None,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.revision = revision
        self.chunk_size = chunk_size

        # Download metadata
        local_root = self._download_meta(root)
        self.local_root = Path(local_root) if local_root else None

        # Load info
        info_path = hf_hub_download(repo_id, "meta/info.json", repo_type="dataset", revision=revision)
        with open(info_path) as f:
            self.info = json.load(f)

        self.fps = self.info["fps"]

        # Load tasks
        tasks_path = hf_hub_download(repo_id, "meta/tasks.jsonl", repo_type="dataset", revision=revision)
        self.tasks = {}
        with open(tasks_path) as f:
            for line in f:
                t = json.loads(line)
                self.tasks[t["task_index"]] = t["task"]

        # Load episodes metadata
        ep_path = hf_hub_download(repo_id, "meta/episodes.jsonl", repo_type="dataset", revision=revision)
        self.episodes_meta = []
        with open(ep_path) as f:
            for line in f:
                self.episodes_meta.append(json.loads(line))

        # Filter episodes
        if episodes is not None:
            self.episodes_meta = [e for e in self.episodes_meta if e["episode_index"] in episodes]
        if max_episodes is not None:
            self.episodes_meta = self.episodes_meta[:max_episodes]

        # Build frame index: list of (episode_index, frame_index_in_episode, chunk_index)
        self.frame_index = []
        for ep in self.episodes_meta:
            ep_idx = ep["episode_index"]
            ep_len = ep["length"]
            chunk_idx = ep_idx // 1000
            for frame_idx in range(ep_len - chunk_size):
                self.frame_index.append((ep_idx, frame_idx, chunk_idx))

        logger.info(
            f"LiberoDataset: {len(self.episodes_meta)} episodes, "
            f"{len(self.frame_index)} trainable frames, chunk_size={chunk_size}"
        )

        # Stats
        stats_path = hf_hub_download(repo_id, "meta/stats.json", repo_type="dataset", revision=revision)
        with open(stats_path) as f:
            raw_stats = json.load(f)
        self.stats = self._build_stats(raw_stats)

        # Cache loaded episodes
        self._ep_cache: dict[int, list[dict]] = {}

    def _download_meta(self, root):
        """Just ensures meta files are available; data is loaded on demand."""
        return root

    def _build_stats(self, raw_stats: dict) -> dict:
        """Convert raw stats to torch tensors with renamed keys, adding q01/q99 from min/max."""
        out = {}
        for old_key, new_key in FEATURE_RENAME.items():
            if old_key not in raw_stats:
                continue
            s = raw_stats[old_key]
            entry = {}
            for stat_name in ["mean", "std", "min", "max"]:
                if stat_name in s:
                    entry[stat_name] = torch.tensor(s[stat_name], dtype=torch.float32)
            # PI05 MEAN_STD mode needs mean/std; also provide q01/q99 from min/max
            if "min" in entry:
                entry["q01"] = entry["min"].clone()
            if "max" in entry:
                entry["q99"] = entry["max"].clone()
            out[new_key] = entry
        return out

    def _load_episode(self, ep_idx: int, chunk_idx: int) -> list[dict]:
        """Load a full episode's data from parquet."""
        if ep_idx in self._ep_cache:
            return self._ep_cache[ep_idx]

        filename = f"data/chunk-{chunk_idx:03d}/episode_{ep_idx:06d}.parquet"
        pq_path = hf_hub_download(
            self.repo_id, filename, repo_type="dataset", revision=self.revision
        )
        table = pq.read_table(pq_path)

        rows = []
        for i in range(table.num_rows):
            row = {}
            for col_name in table.schema.names:
                val = table.column(col_name)[i].as_py()
                row[col_name] = val
            rows.append(row)

        # LRU-style cache (keep last 10 episodes)
        if len(self._ep_cache) > 10:
            oldest = next(iter(self._ep_cache))
            del self._ep_cache[oldest]
        self._ep_cache[ep_idx] = rows
        return rows

    def _decode_image(self, img_struct: dict) -> torch.Tensor:
        """Decode a HF Image struct {bytes, path} to (3, H, W) float32 in [0,1]."""
        img_bytes = img_struct["bytes"]
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
        return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)

    def __len__(self):
        return len(self.frame_index)

    def __getitem__(self, idx):
        ep_idx, frame_idx, chunk_idx = self.frame_index[idx]
        rows = self._load_episode(ep_idx, chunk_idx)

        current = rows[frame_idx]

        # State
        state = torch.tensor(current["state"], dtype=torch.float32)

        # Images
        img_top = self._decode_image(current["image"])
        img_wrist = self._decode_image(current["wrist_image"])

        # Action chunk: frames [frame_idx, frame_idx + chunk_size)
        actions = []
        for t in range(self.chunk_size):
            fi = min(frame_idx + t, len(rows) - 1)
            actions.append(rows[fi]["actions"])
        action = torch.tensor(actions, dtype=torch.float32)  # (chunk_size, action_dim)

        # Task
        task_idx = current["task_index"]
        task = self.tasks.get(task_idx, "manipulation task")

        return {
            "observation.state": state,
            "observation.images.top": img_top,
            "observation.images.wrist": img_wrist,
            "action": action,
            "task": task,
        }

    @property
    def num_frames(self):
        return len(self.frame_index)

    @property
    def num_episodes(self):
        return len(self.episodes_meta)

    @property
    def state_dim(self):
        return self.info["features"]["state"]["shape"][0]

    @property
    def action_dim(self):
        return self.info["features"]["actions"]["shape"][0]
