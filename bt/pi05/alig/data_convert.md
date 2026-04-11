# LeRobot 数据格式分析与 R1 Pro 数据集转换方案

> **日期**: 2026-04-10
> **目标**: 详细解释 LeRobot v2.1 / v3.0 数据格式，分析本地 R1 Pro 数据集与标准格式的差异，并给出转换方案
> **涉及数据集**:
> - `/mnt/r/share/lkx/pi/data/r1_pro_data_convert_chassis`（64 episodes, 61,923 frames）
> - `/mnt/r/share/lkx/pi/data/r1_pro_test_data`（4 episodes, 3,366 frames）
> **代码库**: LeRobot (`/home/Luogang/SRC/Robot/lerobot`), OpenPI (`/mnt/r/share/lkx/pi/openpi`)

---

## 1. LeRobot v2.1 数据格式详解

**版本定义**: v2.1 是 LeRobot 的上一代数据格式，在 LeRobot v3.0 发布前是默认格式。

### 1.1 目录结构

```
dataset_root/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet      ← 每个 episode 一个文件
│       ├── episode_000001.parquet
│       ├── ...
│       └── episode_000063.parquet
├── meta/
│   ├── info.json                       ← 数据集全局元数据
│   ├── tasks.jsonl                     ← 任务描述（JSONL 格式）
│   ├── episodes.jsonl                  ← Episode 元数据（JSONL 格式）
│   └── episodes_stats.jsonl            ← 每个 episode 的统计信息（JSONL 格式）
└── videos/                             ← 可选，视频文件
    └── chunk-000/
        └── {camera_name}/
            ├── episode_000000.mp4
            └── ...
```

### 1.2 元数据文件

#### `meta/info.json`

数据集级别的全局配置，包含版本、特征定义、分片信息等：

```json
{
    "codebase_version": "v2.1",
    "robot_type": "r1_pro",
    "total_episodes": 64,
    "total_frames": 61923,
    "total_tasks": 1,
    "total_videos": 0,                  ← v2.1 特有字段
    "total_chunks": 1,                  ← v2.1 特有字段
    "chunks_size": 1000,
    "fps": 14,
    "splits": {"train": "0:64"},
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    "features": {
        "feature_name": {
            "dtype": "float32" | "int64" | "image" | "video",
            "shape": [dim1, dim2, ...],
            "names": ["name1", ...] | null
        }
    }
}
```

**关键字段说明**：
- `data_path` 模板使用 `episode_chunk` 和 `episode_index`，指向 per-episode 文件
- `total_chunks` 和 `total_videos` 是 v2.1 特有字段（v3.0 中被移除）
- `features` 字典定义所有数据列的类型和形状

#### `meta/tasks.jsonl`

每行一条 JSON，定义任务描述与索引的映射：

```jsonl
{"task_index": 0, "task": "Open the door with a downward-press handle, go through it, and enter the room."}
{"task_index": 1, "task": "open the door handle"}
```

#### `meta/episodes.jsonl`

每行一条 JSON，定义每个 episode 的基本信息：

```jsonl
{"episode_index": 0, "tasks": ["Open the door with a downward-press handle, go through it, and enter the room."], "length": 1135}
{"episode_index": 1, "tasks": ["Open the door with a downward-press handle, go through it, and enter the room."], "length": 942}
```

字段：
- `episode_index`: episode 序号
- `tasks`: 该 episode 对应的任务描述列表
- `length`: 帧数

#### `meta/episodes_stats.jsonl`

每行一条 JSON，存储每个 episode 的统计信息。用于归一化和数据质量检查：

```jsonl
{
  "episode_index": 0,
  "stats": {
    "state": {
      "min": [23 个 float...],
      "max": [23 个 float...],
      "mean": [23 个 float...],
      "std": [23 个 float...],
      "count": [1135]
    },
    "actions": { ... },
    "head_rgb": {
      "min": [[[0.0]], [[0.0]], [[0.0]]],
      "max": [[[1.0]], [[1.0]], [[1.0]]],
      "mean": [[[0.497]], [[0.281]], [[0.206]]],
      "std": [[[0.164]], [[0.151]], [[0.204]]],
      "count": [195]                          ← 采样帧数（非全部帧）
    },
    "timestamp": {"min": [0.0], "max": [81.0], ...},
    "frame_index": {"min": [0], "max": [1134], ...},
    "episode_index": {"min": [0], "max": [0], ...},
    "index": {"min": [0], "max": [1134], ...},
    "task_index": {"min": [0], "max": [0], ...}
  }
}
```

**统计维度说明**：
- 向量特征（state, actions）：per-dimension 统计
- 图像特征（*_rgb）：per-channel 统计，形状为 `[C, 1, 1]`，值归一化到 [0, 1]
- 标量特征（timestamp, frame_index 等）：形状为 `[1]`

### 1.3 数据文件 (Parquet)

每个 episode 对应一个独立的 parquet 文件：`data/chunk-000/episode_XXXXXX.parquet`

parquet 列与 `info.json` 中 `features` 的 key 一一对应。图像以嵌入式 PNG 二进制存储（`dtype: "image"`），使用 HuggingFace `datasets.Image` 类型。

### 1.4 视频文件（可选）

每个 episode、每个相机一个 MP4 文件：`videos/chunk-000/{camera_name}/episode_XXXXXX.mp4`

当 `dtype` 为 `"video"` 时，parquet 中不存储图像数据，而是在加载时从视频文件解码。

---

## 2. LeRobot v3.0 数据格式详解

**版本定义**: v3.0 是 LeRobot 当前版本（`CODEBASE_VERSION = "v3.0"`），定义于 `lerobot/datasets/lerobot_dataset.py:83`。

### 2.1 目录结构

```
dataset_root/
├── data/
│   ├── chunk-000/
│   │   ├── file-000.parquet            ← 合并多个 episode 的数据
│   │   ├── file-001.parquet
│   │   └── ...
│   └── chunk-001/
│       └── ...
├── meta/
│   ├── info.json                       ← 数据集全局元数据（更新版）
│   ├── stats.json                      ← 全局统计信息（新增）
│   ├── tasks.parquet                   ← 任务描述（Parquet 格式，不再是 JSONL）
│   ├── subtasks.parquet                ← 可选子任务
│   └── episodes/
│       └── chunk-000/
│           ├── file-000.parquet        ← Episode 元数据 + 统计（合并）
│           └── ...
└── videos/
    └── {camera_name}/                  ← 相机名提升到顶层
        └── chunk-000/
            ├── file-000.mp4            ← 合并多个 episode 的视频
            └── ...
```

### 2.2 与 v2.1 的关键变化

#### 变化 1: 数据文件合并

v2.1 中每个 episode 一个 parquet 文件，v3.0 中多个 episode 合并到一个文件（按大小分片，默认 `DEFAULT_DATA_FILE_SIZE_IN_MB = 100`）。

- **v2.1**: `data/chunk-000/episode_000000.parquet`（每 episode 一个文件）
- **v3.0**: `data/chunk-000/file-000.parquet`（多 episode 合并，文件名用 `file-xxx` 而非 `episode_xxx`）

路径模板变化：
```
v2.1: data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet
v3.0: data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet
```

#### 变化 2: 元数据格式从 JSONL 改为 Parquet

| 文件 | v2.1 | v3.0 |
|------|------|------|
| 任务 | `meta/tasks.jsonl` | `meta/tasks.parquet` |
| Episode | `meta/episodes.jsonl` | `meta/episodes/chunk-000/file-000.parquet` |
| Episode 统计 | `meta/episodes_stats.jsonl` | 合并到 `meta/episodes/chunk-000/file-000.parquet` |
| 全局统计 | 无独立文件 | `meta/stats.json`（新增） |

#### 变化 3: Episode 元数据内容扩充

v3.0 的 episode parquet 包含更多字段，用于精确定位数据和视频位置：

| 列名 | 类型 | 说明 |
|------|------|------|
| `episode_index` | int | Episode 序号 |
| `tasks` | list[str] | 任务描述 |
| `length` | int | 帧数 |
| `dataset_from_index` | int | 在数据集中的起始全局索引 |
| `dataset_to_index` | int | 在数据集中的结束全局索引 |
| `data/chunk_index` | int | 数据所在 chunk 编号 |
| `data/file_index` | int | 数据所在文件编号 |
| `videos/{cam}/chunk_index` | int | 视频所在 chunk 编号 |
| `videos/{cam}/file_index` | int | 视频所在文件编号 |
| `videos/{cam}/from_timestamp` | float | 视频起始时间戳 |
| `videos/{cam}/to_timestamp` | float | 视频结束时间戳 |
| `stats/{feature}/min` | ndarray | 该 episode 的最小值统计 |
| `stats/{feature}/max` | ndarray | 该 episode 的最大值统计 |
| `stats/{feature}/mean` | ndarray | 均值 |
| `stats/{feature}/std` | ndarray | 标准差 |
| `stats/{feature}/count` | ndarray | 样本数 |
| `stats/{feature}/q01, q10, q50, q90, q99` | ndarray | 分位数统计 |

#### 变化 4: 视频文件路径调整

```
v2.1: videos/chunk-000/{camera_name}/episode_000000.mp4   ← chunk 在外层
v3.0: videos/{camera_name}/chunk-000/file-000.mp4          ← camera 在外层，多 episode 合并
```

#### 变化 5: info.json 字段变化

**v2.1 特有** → v3.0 中移除：
- `total_chunks`
- `total_videos`

**v3.0 新增**：
- `data_files_size_in_mb`（默认 100）
- `video_files_size_in_mb`（默认 200）
- 每个 feature 增加 `fps` 字段

#### 变化 6: 全局统计 stats.json

v3.0 新增 `meta/stats.json`，由所有 episode 的统计聚合而成。包含：

```json
{
  "observation.state": {
    "min": [...], "max": [...], "mean": [...], "std": [...],
    "count": [...],
    "q01": [...], "q10": [...], "q50": [...], "q90": [...], "q99": [...]
  },
  "action": { ... }
}
```

注意 v3.0 的 stats 增加了**分位数**（q01/q10/q50/q90/q99），这对于 pi0.5 的 quantile normalization 至关重要。

### 2.3 LeRobot v3.0 特征命名约定

LeRobot 标准数据集使用以下命名规范：

| 类别 | 命名约定 | 示例 |
|------|---------|------|
| 图像观测 | `observation.images.{camera_name}` | `observation.images.head_rgb` |
| 视频观测 | `observation.images.{camera_name}` (dtype=video) | `observation.images.cam_high` |
| 状态观测 | `observation.state` | `observation.state` |
| 环境状态 | `observation.environment_state` | `observation.environment_state` |
| 动作 | `action` | `action` (单数) |
| 元数据 | `timestamp`, `frame_index`, `episode_index`, `index`, `task_index` | — |

**这是一个重要约定**，LeRobot 的 PI05Policy 依赖这些前缀来识别特征类型。

---

## 3. v2.1 vs v3.0 完整差异对照表

| 特性 | v2.1 | v3.0 |
|------|------|------|
| **版本号** | `"v2.1"` | `"v3.0"` |
| **数据文件** | 每 episode 一个 parquet | 按大小合并多 episode (默认 100MB/文件) |
| **数据路径模板** | `data/chunk-{chunk}/episode_{idx}.parquet` | `data/chunk-{chunk}/file-{file}.parquet` |
| **元数据格式** | JSONL (tasks.jsonl, episodes.jsonl, episodes_stats.jsonl) | Parquet (tasks.parquet, episodes/*.parquet) |
| **全局统计** | 无独立文件 | `meta/stats.json`（含分位数） |
| **Episode 统计** | 单独 `episodes_stats.jsonl` | 合并到 `episodes/*.parquet` 中 |
| **视频路径** | `videos/chunk-{c}/{cam}/episode_{i}.mp4` | `videos/{cam}/chunk-{c}/file-{f}.mp4` |
| **视频存储** | 每 episode 一个 mp4 | 多 episode 拼接（默认 200MB/文件） |
| **info.json 独有字段** | `total_chunks`, `total_videos` | `data_files_size_in_mb`, `video_files_size_in_mb` |
| **feature 级 fps** | 无 | 每个 feature 都有 `fps` 字段 |
| **分位数统计** | 无 | q01, q10, q50, q90, q99 |
| **向后兼容** | — | 自动检测 v2.1 并提示转换 |

---

## 4. 本地数据集格式分析

### 4.1 `r1_pro_data_convert_chassis` (训练集)

**来源**: 由 `/mnt/r/share/lkx/pi/scripts/convert_r1pro_chassis_data.py` 生成，专为 OpenPI 框架设计。

**基本信息**:
- 版本: v2.1
- Robot: r1_pro
- Episodes: 64
- 总帧数: 61,923
- FPS: 14 Hz
- 任务: 1 个 ("Open the door with a downward-press handle, go through it, and enter the room.")

**目录结构**:
```
r1_pro_data_convert_chassis/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet (约 700MB)
│       ├── episode_000001.parquet
│       └── ... (共 64 个文件)
└── meta/
    ├── info.json
    ├── tasks.jsonl
    ├── episodes.jsonl
    └── episodes_stats.jsonl
```

**Parquet Schema (10 列)**:

| 列名 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `head_rgb` | Image (PNG binary) | 360×640×3 | 头部相机 |
| `left_wrist_rgb` | Image (PNG binary) | 480×640×3 | 左腕相机 |
| `right_wrist_rgb` | Image (PNG binary) | 480×640×3 | 右腕相机 |
| `state` | float32[23] | (23,) | 机器人状态 |
| `actions` | float32[23] | (23,) | 动作 |
| `timestamp` | float32 | (1,) | 时间戳 |
| `frame_index` | int64 | (1,) | 帧索引 |
| `episode_index` | int64 | (1,) | Episode 索引 |
| `index` | int64 | (1,) | 全局索引 |
| `task_index` | int64 | (1,) | 任务索引 |

**State/Action 维度 (23-dim)**:

| 索引 | 维度 | 含义 |
|------|------|------|
| [0:7] | 7 | left_arm (关节角度) |
| [7:14] | 7 | right_arm (关节角度) |
| [14] | 1 | left_gripper |
| [15] | 1 | right_gripper |
| [16:20] | 4 | torso |
| [20:23] | 3 | chassis_velocities (x, y, rotation) |

### 4.2 `r1_pro_test_data` (测试集)

**基本信息**:
- 版本: v2.1
- Robot: r1_pro
- Episodes: 4
- 总帧数: 3,366
- FPS: 14 Hz
- 任务: 2 个

**Episode 详情**:

| Episode | 帧数 | 任务 |
|---------|------|------|
| 0 | 958 | Open the door with a downward-press handle... |
| 1 | 443 | open the door handle |
| 2 | 1,006 | Open the door with a downward-press handle... |
| 3 | 959 | Open the door with a downward-press handle... |

**Schema**: 与训练集完全相同（10 列，相同类型和形状）。

### 4.3 图像存储方式

两个数据集的图像均以 **嵌入式 PNG** 存储在 parquet 中（不使用外部视频文件）：

```
parquet 列结构:
  head_rgb: struct {
    bytes: binary,   ← PNG 编码的图像字节
    path: string     ← 原始文件路径（信息性，非必需）
  }
```

这意味着 `total_videos: 0`，不需要视频解码器。

---

## 5. 本地数据集与 LeRobot 标准格式的差异分析

### 5.1 差异总览

| 差异项 | 本地数据集（当前） | LeRobot v2.1 标准 | LeRobot v3.0 标准 | 严重程度 |
|--------|------------------|-------------------|-------------------|---------|
| **版本号** | v2.1 | v2.1 ✅ | v3.0 ❌ | **致命** |
| **图像列名** | `head_rgb` | 无约定 | `observation.images.head_rgb` ❌ | **致命** |
| **状态列名** | `state` | 无约定 | `observation.state` ❌ | **致命** |
| **动作列名** | `actions` (复数) | 无约定 | `action` (单数) ❌ | **致命** |
| **元数据格式** | JSONL | JSONL ✅ | Parquet ❌ | 中 |
| **数据分片** | per-episode | per-episode ✅ | 合并文件 ❌ | 中 |
| **全局统计** | 无 | 无 ✅ | stats.json ❌ | 中 |
| **分位数** | 无 | 无 ✅ | q01/q10/q50/q90/q99 ❌ | 低 |
| **feature fps** | 无 | 无 ✅ | 每 feature 有 fps ❌ | 低 |

### 5.2 差异 1: 列名不符合 LeRobot 约定（致命）

这是**最关键的差异**。本地数据集使用 OpenPI 的平铺列名，而 LeRobot pi0.5 期望层级命名：

```
当前数据集列名          →  LeRobot PI05Policy 期望的列名
────────────────────────────────────────────────────────
head_rgb              →  observation.images.head_rgb
left_wrist_rgb        →  observation.images.left_wrist_rgb
right_wrist_rgb       →  observation.images.right_wrist_rgb
state                 →  observation.state
actions               →  action                              ← 注意：复数→单数
timestamp             →  timestamp                           ← 无需改动
frame_index           →  frame_index                         ← 无需改动
episode_index         →  episode_index                       ← 无需改动
index                 →  index                               ← 无需改动
task_index            →  task_index                          ← 无需改动
```

**为什么必须修改**：

LeRobot 的 `PI05Policy` 通过 `input_features` 配置中的 key 来从 batch 中读取对应数据。例如：

```python
# PI05Config 中的 feature 定义
input_features = {
    "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(23,)),
    "observation.images.head_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    ...
}
```

这些 key 会直接作为 `batch["observation.images.head_rgb"]` 来索引训练数据。如果数据集列名不匹配，会导致 `KeyError`。

**与 OpenPI 的对比**：

OpenPI 框架不要求这种命名约定。OpenPI 的 `R1ProChassisInputs` transform 直接使用 `head_rgb`、`state`、`actions` 等平铺名称，并在内部重映射为 `image.base_0_rgb` 等模型期望的格式。这就是为什么这些数据集可以直接用于 OpenPI 但不能直接用于 LeRobot。

### 5.3 差异 2: 版本不匹配（致命）

当前 LeRobot 代码库版本为 v3.0（`CODEBASE_VERSION = "v3.0"`）。当尝试加载 v2.1 数据集时，LeRobot 会抛出 `BackwardCompatibilityError`：

```python
# lerobot/datasets/backward_compatibility.py
class BackwardCompatibilityError(Exception):
    """当数据集版本低于代码库版本时抛出"""
```

这意味着即使列名正确，v2.1 数据集也无法被当前 LeRobot 代码直接加载。

### 5.4 差异 3: 元数据和统计格式差异（可自动转换）

v2.1 的 JSONL 元数据需要转换为 v3.0 的 Parquet 元数据。这些差异可以通过 LeRobot 官方提供的 `convert_dataset_v21_to_v30.py` 脚本自动处理。

### 5.5 差异 4: 缺少分位数统计（影响训练质量）

pi0.5 模型使用 **quantile normalization**（基于 q01/q99）而非标准 z-score normalization。v2.1 格式的 `episodes_stats.jsonl` 只包含 min/max/mean/std/count，不含分位数。v3.0 转换过程会重新计算统计信息，包括分位数。

---

## 6. 兼容性分析：能否用于 LeRobot pi0.5 训练？

### 6.1 对于 OpenPI pi0.5 训练 ✅ 可以直接使用

**结论**: 两个数据集**可以直接用于** OpenPI 框架的 pi0.5 训练。

**原因**:
1. OpenPI 的 `data_loader.py` 使用 `LeRobotDataset` 加载数据，但使用的是**旧版 LeRobot 库**（兼容 v2.1）
2. 列名 (`head_rgb`, `state`, `actions`) 与 `R1ProChassisInputs` transform 完全匹配
3. OpenPI 的 `compute_norm_stats.py` 会独立计算 norm stats，不依赖数据集内的统计信息

**数据流**:
```
LeRobot v2.1 数据集
  ├─ head_rgb, left_wrist_rgb, right_wrist_rgb (原始列名)
  ├─ state (23-dim), actions (23-dim)
  └─ metadata
       ↓ [LeRobotDataset.__getitem__() — 保持原始列名]
       ↓ [无 RepackTransform（SimpleDataConfig 未定义）]
R1ProChassisInputs transform
  ├─ head_rgb → image.base_0_rgb
  ├─ left_wrist_rgb → image.left_wrist_0_rgb
  ├─ right_wrist_rgb → image.right_wrist_0_rgb
  └─ state, actions 保持不变
       ↓ [Normalize — 使用预计算的 norm_stats]
       ↓ [ModelTransforms — 图像 resize 224×224, prompt tokenize]
Model Input
```

### 6.2 对于 LeRobot pi0.5 训练 ❌ 不能直接使用

**结论**: 两个数据集**不能直接用于**当前 LeRobot 框架的 pi0.5 训练。

**原因**:

| 阻碍 | 具体问题 | 后果 |
|------|---------|------|
| **版本不兼容** | v2.1 vs v3.0 | `BackwardCompatibilityError`，无法加载 |
| **列名不匹配** | `head_rgb` vs `observation.images.head_rgb` | `KeyError`，无法读取数据 |
| **动作列名** | `actions` vs `action` | `KeyError`，无法读取动作 |
| **缺少分位数** | 无 q01/q99 | 无法进行 quantile normalization |

**必须进行格式转换后才能使用。**

---

## 7. 使用指南：转换与验证

> **脚本位置**: `bt/pi05/alig/data/`
> **Python 环境**: `/mnt/r/Venv/lerobot-venv/bin/python`
> **工作目录**: 所有命令均在 `/home/Luogang/SRC/Robot/lerobot` 下执行

### 7.1 脚本一览

| 脚本 | 用途 |
|------|------|
| `bt/pi05/alig/data/convert_r1pro_to_lerobot.py` | 主转换脚本：列名重命名 + v2.1→v3.0 + 分位数计算 + 基本验证 |
| `bt/pi05/alig/data/verify_pi05.py` | Pi0.5 兼容性验证：数据加载 / 预处理器 / 前向传播（可选） |
| `bt/pi05/alig/data/run_convert.sh` | 一键运行：转换两个数据集 + 验证 |

### 7.2 快速开始（一键运行）

```bash
cd /home/Luogang/SRC/Robot/lerobot

# 使用 lerobot-venv 环境
export PATH="/mnt/r/Venv/lerobot-venv/bin:$PATH"

# 一键转换两个数据集 + Pi0.5 Level 1-2 验证
bash bt/pi05/alig/data/run_convert.sh

# 如需额外运行 Pi0.5 forward pass 验证（需要 GPU）
bash bt/pi05/alig/data/run_convert.sh --forward-pass
```

一键脚本会依次执行：
1. 转换 `r1_pro_test_data`（4 episodes，全量）
2. 转换 `r1_pro_data_convert_chassis`（随机采样 10 episodes）
3. 对两个转换结果运行 Pi0.5 Level 1+2 验证

### 7.3 分步操作

#### 7.3.1 转换测试集（全量，4 episodes，约 2 分钟）

```bash
cd /home/Luogang/SRC/Robot/lerobot

python bt/pi05/alig/data/convert_r1pro_to_lerobot.py \
    --input /mnt/r/share/lkx/pi/data/r1_pro_test_data \
    --output bt/pi05/alig/data/r1_pro_test_data_v30
```

#### 7.3.2 转换训练集（采样子集，约 3 分钟）

训练集有 64 episodes、45GB。先采样 10 个 episodes 验证流程：

```bash
python bt/pi05/alig/data/convert_r1pro_to_lerobot.py \
    --input /mnt/r/share/lkx/pi/data/r1_pro_data_convert_chassis \
    --output bt/pi05/alig/data/r1_pro_chassis_v30 \
    --sample-episodes 10
```

#### 7.3.3 转换训练集（全量，64 episodes，约 30 分钟）

确认流程无误后，转换完整数据集：

```bash
python bt/pi05/alig/data/convert_r1pro_to_lerobot.py \
    --input /mnt/r/share/lkx/pi/data/r1_pro_data_convert_chassis \
    --output /path/to/output/r1_pro_chassis_v30
```

> **磁盘空间**：全量转换需要约 90GB 临时空间（原始数据不会被修改）。转换完成后脚本会自动清理中间目录。

### 7.4 `convert_r1pro_to_lerobot.py` 参数说明

```
必选参数:
  --input              输入数据集目录（OpenPI v2.1 格式）
  --output             输出数据集目录（LeRobot v3.0 格式）

可选参数:
  --sample-episodes N  随机采样 N 个 episodes（不指定则转换全部）
  --seed SEED          随机种子，默认 42（采样时使用，保证可复现）
  --skip-v30           仅执行列名重命名，跳过 v2.1→v3.0 升级（调试用）
  --skip-verify        跳过转换后的基本验证
```

### 7.5 转换流程详解

脚本内部执行 4 个阶段：

```
Phase 0+1 (合并执行)          Phase 2                    Phase 2.5              Phase 3
列名重命名 + Episode 采样 ──→ v2.1→v3.0 官方转换器 ──→ 计算分位数统计 ──→ 基本验证
                                                                               
  - Parquet 列重命名            - 合并数据文件             - 从 parquet 计算       - info.json 版本
  - Episode 重编号(采样时)       - JSONL→Parquet 元数据      q01/q10/q50/q90/q99   - features 命名
  - Task 重映射(采样时)          - 生成 stats.json         - 更新 stats.json      - 分位数存在
  - info.json 更新              - 升级 codebase_version   - 更新 episode 元数据   - LeRobotDataset 加载
  - episodes_stats 更新          - 清理中间目录                                   - 样本 shape 验证
```

**关键设计**：Phase 0 和 Phase 1 合并执行——读取原始 parquet 时同时完成列重命名和 episode 重编号，避免对每个 ~700MB 的文件写两次。

**为什么需要 Phase 2.5**：LeRobot 官方 v2.1→v3.0 转换器（`convert_dataset_v21_to_v30.py`）的 `aggregate_stats` 只聚合源数据中已有的统计键。v2.1 的 `episodes_stats.jsonl` 不含分位数（q01/q10/q50/q90/q99），所以转换后的 `stats.json` 也不含分位数。但 Pi0.5 的 quantile normalization 依赖这些分位数。Phase 2.5 从 v3.0 数据 parquet 直接计算分位数并补充到 `stats.json`。

### 7.6 验证转换结果

#### 7.6.1 Pi0.5 兼容性验证（Level 1+2，无需 GPU）

```bash
python bt/pi05/alig/data/verify_pi05.py \
    --dataset-dir bt/pi05/alig/data/r1_pro_test_data_v30
```

验证内容：

| 级别 | 检查项 | 需要 GPU |
|------|--------|---------|
| **Level 1: 数据加载** | LeRobotDatasetMetadata 加载、codebase_version=v3.0、stats 含 q01/q99、LeRobotDataset 加载、随机 5 个样本的 keys 和 shapes 验证 | 否 |
| **Level 2: 预处理器** | `dataset_to_policy_features()` 正确分类（3 VISUAL + 1 STATE + 1 ACTION）、PI05Config 创建、NormalizerProcessorStep 初始化、QUANTILES 归一化验证（state/action 归一化到 [-1, 1]） | 否 |

期望输出：

```
Level 1: 数据加载验证
  Metadata 加载成功
    codebase_version: v3.0
    total_episodes: 4
    total_frames: 3366
    features: ['observation.images.head_rgb', ..., 'observation.state', 'action', ...]
  Stats 验证通过 (含 q01/q99 分位数)
  Dataset 加载成功: 4 episodes, 3366 frames
  样本读取验证通过 (5 samples, keys + shapes 正确)
Level 1 通过!

Level 2: 预处理器兼容性验证
  Input features:
    observation.images.head_rgb: type=VISUAL, shape=(3, 360, 640)
    observation.images.left_wrist_rgb: type=VISUAL, shape=(3, 480, 640)
    observation.images.right_wrist_rgb: type=VISUAL, shape=(3, 480, 640)
    observation.state: type=STATE, shape=(23,)
  Output features:
    action: type=ACTION, shape=(23,)
  Feature 分类验证通过
  NormalizerProcessorStep 初始化成功
  样本归一化通过
    normalized state range: [-1.000, 1.000]
    normalized action range: [-1.000, 1.000]
Level 2 通过!
全部验证通过!
```

#### 7.6.2 Pi0.5 前向传播验证（Level 3，需要 GPU）

```bash
python bt/pi05/alig/data/verify_pi05.py \
    --dataset-dir bt/pi05/alig/data/r1_pro_test_data_v30 \
    --run-forward-pass \
    --pretrained-path lerobot/pi05_base \
    --num-steps 2
```

Level 3 额外检查：加载 PI05Policy 预训练模型，构建完整 preprocessor pipeline（含 tokenizer），运行 1-2 步 forward pass，验证 loss 为有限值。

#### 7.6.3 `verify_pi05.py` 参数说明

```
必选参数:
  --dataset-dir        转换后的 v3.0 数据集目录

可选参数:
  --run-forward-pass   启用 Level 3 前向传播验证（需要 GPU + 预训练模型）
  --pretrained-path    Pi0.5 预训练模型路径，默认 lerobot/pi05_base
  --tokenizer-name     Tokenizer，默认 google/paligemma-3b-pt-224
  --num-steps N        前向传播步数，默认 2
```

### 7.7 转换后的数据集使用

#### 7.7.1 在 Python 中加载

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    repo_id="local/r1_pro_test_data_v30",
    root="bt/pi05/alig/data/r1_pro_test_data_v30",
)

sample = dataset[0]
print(sample.keys())
# dict_keys(['observation.images.head_rgb', 'observation.images.left_wrist_rgb',
#             'observation.images.right_wrist_rgb', 'observation.state', 'action',
#             'timestamp', 'frame_index', 'episode_index', 'index', 'task_index', 'task'])

print(sample["observation.state"].shape)  # torch.Size([23])
print(sample["action"].shape)             # torch.Size([23])
```

#### 7.7.2 用于 Pi0.5 训练

```python
from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.pi05.configuration_pi05 import PI05Config

# 1. 加载 metadata 和 features
ds_meta = LeRobotDatasetMetadata("local/r1_pro_chassis_v30", root="path/to/r1_pro_chassis_v30")
features = dataset_to_policy_features(ds_meta.info["features"])

# 2. 自动分类 input/output features
output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
input_features = {k: ft for k, ft in features.items() if k not in output_features}

# 3. 创建 PI05Config
config = PI05Config(
    input_features=input_features,    # 自动包含 3 个 VISUAL + 1 个 STATE
    output_features=output_features,  # 自动包含 1 个 ACTION (23-dim)
)
config.normalization_mapping = {
    "VISUAL": NormalizationMode.IDENTITY,
    "STATE": NormalizationMode.QUANTILES,
    "ACTION": NormalizationMode.QUANTILES,
}

# 后续训练流程参考 bt/pi05/train_pi05_local.py
```

### 7.8 转换前后对比

#### 文件结构

```
转换前 (v2.1, OpenPI 列名)              转换后 (v3.0, LeRobot 列名)
─────────────────────────              ─────────────────────────
data/                                  data/
  chunk-000/                             chunk-000/
    episode_000000.parquet                 file-000.parquet (合并多 episode)
    episode_000001.parquet                 file-001.parquet
    ...                                    ...
meta/                                  meta/
  info.json (v2.1)                       info.json (v3.0)
  tasks.jsonl                            tasks.parquet
  episodes.jsonl                         stats.json (新增，含分位数)
  episodes_stats.jsonl                   episodes/
                                           chunk-000/
                                             file-000.parquet (含统计)
```

#### Parquet 列名

```
转换前                      →    转换后
──────────────────────────────────────────────
head_rgb                   →    observation.images.head_rgb
left_wrist_rgb             →    observation.images.left_wrist_rgb
right_wrist_rgb            →    observation.images.right_wrist_rgb
state                      →    observation.state
actions                    →    action
timestamp / frame_index / episode_index / index / task_index → 不变
```

#### info.json 关键变化

| 字段 | 转换前 | 转换后 |
|------|--------|--------|
| `codebase_version` | `"v2.1"` | `"v3.0"` |
| `total_chunks` | 1 | (已移除) |
| `total_videos` | 0 | (已移除) |
| `data_files_size_in_mb` | (无) | 100 |
| `video_files_size_in_mb` | (无) | 200 |
| `data_path` | `data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet` | `data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet` |
| features keys | `head_rgb`, `state`, `actions` | `observation.images.head_rgb`, `observation.state`, `action` |

#### stats.json 新增分位数

```json
{
  "observation.state": {
    "min": [...], "max": [...], "mean": [...], "std": [...], "count": [...],
    "q01": [...], "q10": [...], "q50": [...], "q90": [...], "q99": [...]
  },
  "action": { /* 同上 */ }
}
```

### 7.9 注意事项

#### 内存

每个 episode parquet 约 700MB（嵌入 PNG 图像）。脚本逐文件处理，峰值内存约 1.5GB。

#### 磁盘空间

| 场景 | 需要空间 |
|------|---------|
| 测试集全量 (4 episodes) | ~6 GB |
| 训练集采样 10 episodes | ~14 GB |
| 训练集全量 (64 episodes) | ~90 GB |

转换完成后中间目录 `*_old` 会被自动清理。

#### 与 OpenPI 数据集共存

转换脚本**不修改原始数据集**（`--input` 只读，`--output` 独立目录）：
- OpenPI 继续使用原始数据集（`head_rgb`, `state`, `actions`）
- LeRobot 使用转换后的数据集（`observation.images.*`, `observation.state`, `action`）

#### Episode 采样与重编号

使用 `--sample-episodes` 时，脚本会：
1. 随机采样指定数量的 episode（可通过 `--seed` 控制可复现性）
2. 将采样的 episode 重编号为连续的 0-based 索引（官方转换器要求）
3. 重新计算全局帧索引 (`index` 列)
4. 重映射 task_index（仅保留被引用的 task）

这确保输出是一个独立、完整、可直接使用的 v3.0 数据集。

### 7.10 故障排查

| 错误 | 原因 | 解决 |
|------|------|------|
| `ModuleNotFoundError: No module named 'lerobot'` | 未使用正确的 Python 环境 | 使用 `/mnt/r/Venv/lerobot-venv/bin/python` |
| `BackwardCompatibilityError` | 尝试直接加载 v2.1 数据集 | 运行转换脚本先转为 v3.0 |
| `KeyError: 'total_chunks'` | Phase 1 输出的 info.json 缺少此字段 | 确保使用最新版转换脚本（脚本会保留此字段） |
| `stats.json 缺少 q01 分位数` | Phase 2.5 未执行 | 确保未使用 `--skip-v30`（Phase 2.5 在 Phase 2 之后自动执行） |
| `ValueError: Number of episodes is not the same` | 采样后 episode 索引未正确重编号 | 确保使用最新版转换脚本（已内置重编号逻辑） |
| 磁盘空间不足 | 全量转换需约 90GB | 先用 `--sample-episodes 10` 测试，或指定有足够空间的 `--output` 路径 |

---

## 8. LeRobot vs OpenPI 数据集 Key 映射差异深度分析

> **日期**: 2026-04-11
> **背景**: `pi05_alig.md` 2.2.3 节指出 LeRobot 和 OpenPI 对数据集 Key 的需求不同。本节基于两个框架源码和两个本地数据集的完整数据流追踪，分析差异的本质、影响，以及转换脚本的覆盖情况。

### 8.1 原始数据集的 Key（两个框架的共同起点）

两个本地数据集 (`r1_pro_data_convert_chassis`, `r1_pro_test_data`) 使用 OpenPI 命名约定：

| 原始 Key | 类型 | Shape | 说明 |
|----------|------|-------|------|
| `head_rgb` | image | [360,640,3] | 头部相机 |
| `left_wrist_rgb` | image | [480,640,3] | 左腕相机 |
| `right_wrist_rgb` | image | [480,640,3] | 右腕相机 |
| `state` | float32 | [23] | 关节状态 |
| `actions` | float32 | [23] | 动作 (**复数**) |
| `task_index` | int64 | [1] | 任务索引 |

### 8.2 两个框架的完整数据流

#### 8.2.1 OpenPI 数据流 (pi05_r1pro_chassis 配置)

```
原始数据集 (v2.1, OpenPI Key)
  head_rgb, left_wrist_rgb, right_wrist_rgb, state, actions, task_index
                              │
                              ▼
  PromptFromLeRobotTask          (data_loader.py:149)
    task_index → 查 tasks dict → 添加 "prompt" key
                              │
                              ▼
  R1ProChassisInputs             (r1pro_chassis_policy.py:55-80)
    head_rgb       → image["base_0_rgb"]
    left_wrist_rgb → image["left_wrist_0_rgb"]
    right_wrist_rgb→ image["right_wrist_0_rgb"]
    state          → state (不变)
    actions        → actions (不变)
    + image_mask (全 True)
                              │
                              ▼
  Normalize                      (transforms.py:144)
    norm_stats keys: "state", "actions"
    公式: (x - q01) / (q99 - q01 + 1e-6) * 2 - 1
                              │
                              ▼
  ModelTransformFactory → Pi0.5  (config.py:128-140)
    InjectDefaultPrompt → ResizeImages(224,224) →
    TokenizePrompt(discrete_state_input=True):
      读 normalized state → 离散化 256 bins
      构建 "Task: {text}, State: {bins};\nAction: "
      SentencePiece 编码 → tokenized_prompt + tokenized_prompt_mask
    → PadStatesAndActions(32): state/actions 补零到 32 维
                              │
                              ▼
  Observation.from_dict          (model.py:110-129)
    data["image"]              → images (dict: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb)
    data["state"]              → state (32 维)
    data["tokenized_prompt"]   → tokenized_prompt
    batch["actions"]           → actions (32 维)
```

**关键源码引用**:
- `action_sequence_keys=("actions",)`: `config.py:1036` — 用于构建 `delta_timestamps`，从数据集读取 action chunk
- `R1ProChassisInputs.__call__()`: `r1pro_chassis_policy.py:55-80` — 执行 `data["head_rgb"]` 等原始 key 读取
- `Observation.from_dict()`: `model.py:110-129` — 最终模型输入结构

#### 8.2.2 LeRobot 数据流 (Pi0.5 训练)

```
转换后数据集 (v3.0, LeRobot Key)
  observation.images.head_rgb, observation.images.left_wrist_rgb,
  observation.images.right_wrist_rgb, observation.state, action, task
                              │
                              ▼
  RenameObservationsProcessorStep(rename_map={})   (processor_pi05.py:130)
    (空映射, 无操作, 为 from_pretrained 兼容而保留)
                              │
                              ▼
  AddBatchDimensionProcessorStep                    (batch_processor.py)
    所有 tensor 增加 batch 维: [23] → [1,23], [C,H,W] → [1,C,H,W]
                              │
                              ▼
  NormalizerProcessorStep                           (normalize_processor.py)
    stats keys: "observation.state", "action" (来自 stats.json)
    公式: 2 * (x - q01) / max(q99 - q01, 1e-8) - 1
                              │
                              ▼
  Pi05PrepareStateTokenizerProcessorStep            (processor_pi05.py:57-85)
    读 observation.state → 离散化 256 bins (同 OpenPI 算法)
    构建 "Task: {text}, State: {bins};\nAction: " (同 OpenPI 格式)
    更新 complementary_data["task"]
                              │
                              ▼
  TokenizerProcessorStep                            (tokenizer_processor.py)
    HuggingFace AutoTokenizer("google/paligemma-3b-pt-224")
    → observation.language.tokens + observation.language.attention_mask
                              │
                              ▼
  DeviceProcessorStep → 移到 GPU
                              │
                              ▼
  PI05Policy.forward                                (modeling_pi05.py:1382-1419)
    _preprocess_images: 遍历 config.image_features
      → batch["observation.images.head_rgb"] 等
      → resize 224x224, [0,1] → [-1,1]
    prepare_action: batch["action"] → pad 到 32 维
    tokens: batch["observation.language.tokens"]
```

**关键源码引用**:
- `OBS_STATE = "observation.state"`: `constants.py:23` — Pi05PrepareStateTokenizerProcessorStep 读取此 key
- `ACTION = "action"`: `constants.py:33` — PI05Policy.prepare_action() 读取此 key
- `config.image_features`: `configuration_pi05.py` property — 过滤 `input_features` 中 `FeatureType.VISUAL` 类型

### 8.3 Key 差异全对照表

| 概念 | OpenPI 数据集 Key | OpenPI 模型内部 Key | LeRobot 数据集 Key | LeRobot 模型内部 Key |
|------|-------------------|--------------------|--------------------|---------------------|
| **头部相机** | `head_rgb` | `image.base_0_rgb` | `observation.images.head_rgb` | `observation.images.head_rgb` |
| **左腕相机** | `left_wrist_rgb` | `image.left_wrist_0_rgb` | `observation.images.left_wrist_rgb` | `observation.images.left_wrist_rgb` |
| **右腕相机** | `right_wrist_rgb` | `image.right_wrist_0_rgb` | `observation.images.right_wrist_rgb` | `observation.images.right_wrist_rgb` |
| **状态** | `state` | `state` | `observation.state` | `observation.state` |
| **动作** | `actions` (复数) | `actions` | `action` (单数) | `action` |
| **任务文本** | `task_index` → `prompt` | `tokenized_prompt` | `task_index` → `task` | `observation.language.tokens` |
| **归一化统计** | `state`/`actions` (norm_stats.json) | — | `observation.state`/`action` (stats.json) | — |
| **图像容器结构** | 扁平 dict | 嵌套 dict: `data["image"]["base_0_rgb"]` | 扁平 dict | 扁平 dict: `batch["observation.images.head_rgb"]` |
| **Action 序列读取** | `action_sequence_keys=("actions",)` | — | `ACTION = "action"` | — |

### 8.4 差异的本质

两个框架使用**完全不同的 Key 命名体系**，但差异是**形式上的而非实质上的**：

1. **LeRobot 约定**: `observation.` 前缀 + 层级式命名 (`observation.images.head_rgb`)，动作用单数 `action`
2. **OpenPI 约定**: 无前缀 + 扁平命名 (`head_rgb`)，内部重映射为通用 Key (`base_0_rgb`)，动作用复数 `actions`
3. **各框架的 transform 链负责从各自期望的数据集 Key 映射到各自模型需要的内部 Key**

关键要理解的是：**不存在一个统一的 "正确" Key 命名**。每个框架有自己的约定，并各自在 data transform 层解决映射问题：
- OpenPI 通过 `R1ProChassisInputs` transform 完成映射
- LeRobot 通过 `dataset_to_policy_features()` + processor pipeline 完成映射

### 8.5 转换脚本的评估

#### 8.5.1 转换脚本做了什么

`convert_r1pro_to_lerobot.py` 的 `COLUMN_RENAME_MAP`:

```python
"head_rgb"        → "observation.images.head_rgb"       # 匹配 LeRobot OBS_IMAGES 前缀约定
"left_wrist_rgb"  → "observation.images.left_wrist_rgb"  # 同上
"right_wrist_rgb" → "observation.images.right_wrist_rgb" # 同上
"state"           → "observation.state"                   # 匹配 OBS_STATE 常量 (constants.py:23)
"actions"         → "action"                              # 匹配 ACTION 常量 (constants.py:33)
```

同时重命名了 `info.json` features keys、`episodes_stats.jsonl` stats keys，并计算了 q01/q99 分位数。

#### 8.5.2 逐步验证

| LeRobot Pi0.5 链路步骤 | 需要的 Key | 转换后提供 | 验证状态 |
|------------------------|-----------|-----------|----------|
| `dataset_to_policy_features()` | `observation.*` → STATE/VISUAL, `action` → ACTION | ✓ | 正确分类 |
| `NormalizerProcessorStep` | stats.json 中 `observation.state` / `action` 含 q01/q99 | ✓ | 正确归一化 |
| `Pi05PrepareStateTokenizerProcessorStep` | transition 中 `observation.state` | ✓ | 正确读取 |
| `TokenizerProcessorStep` | complementary_data 中 `task` | ✓ (via LeRobotDataset) | 正确 tokenize |
| `PI05Policy._preprocess_images()` | batch 中 `observation.images.*` | ✓ | 正确读取 |
| `PI05Policy.prepare_action()` | batch 中 `action` | ✓ | 正确 pad 到 32 维 |

**结论**: **转换脚本完全正确，无遗漏。** Level 1+2 验证已在实际数据上通过：
- r1_pro_test_data: 4 episodes, 3366 frames, normalized state/action in [-1.000, 1.031]
- r1_pro_chassis: 10 episodes, 9830 frames, normalized state/action in [-1.000, 1.000]

### 8.6 次要差异（不影响训练正确性）

#### 8.6.1 归一化公式微差

| | OpenPI (`transforms.py:144`) | LeRobot (`normalize_processor.py`) |
|-|------------------------------|-------------------------------------|
| 公式 | `(x-q01) / (q99-q01 + 1e-6) * 2 - 1` | `2 * (x-q01) / max(q99-q01, 1e-8) - 1` |
| epsilon | 无条件加 `1e-6` | 仅当 `q99==q01` 时用 `1e-8` |

**影响**: 对 R1 Pro 数据 (q99-q01 通常 >> 1e-6)，两者差异 < 1e-6，被 256-bin 离散化完全吞没。**可忽略。**

#### 8.6.2 Tokenizer 实现差异

| | OpenPI | LeRobot |
|-|--------|---------|
| 实现 | `sentencepiece.SentencePieceProcessor` (GCS 下载 `paligemma_tokenizer.model`) | `AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")` |
| 编码 | `encode(prompt, add_bos=True)` | HuggingFace API |

两者底层使用相同的 SentencePiece model，产生相同 token 序列。**无影响。**

#### 8.6.3 图像 resize 方法

- OpenPI: `image_tools.resize_with_pad()` — 保持宽高比，黑边填充到 224×224
- LeRobot: `resize_with_pad_torch()` (`modeling_pi05.py`) — 功能等价的 torch 实现

**影响**: 当原图非正方形时 (360×640, 480×640)，pad 策略一致（都保持宽高比）。resize 算法细微差异可能在像素级产生微小偏差，但不影响训练质量。与 Key 映射无关。

### 8.7 双框架使用策略

| 框架 | 使用的数据集 | Key 命名 | 说明 |
|------|------------|---------|------|
| **OpenPI** | 原始数据集 (`/mnt/r/share/lkx/pi/data/r1_pro_*`) | `head_rgb`, `state`, `actions` | R1ProChassisInputs transform 处理映射 |
| **LeRobot** | 转换后数据集 (`*_v30/`) | `observation.images.*`, `observation.state`, `action` | Processor pipeline 处理映射 |

转换脚本**不修改原始数据集**，两个版本可共存：
- OpenPI 配置 `pi05_r1pro_chassis` → `repo_id="r1_pro_data_convert_chassis"` → 读原始 Key
- LeRobot 训练脚本 → `repo_id="local/r1_pro_chassis_v30"` → 读转换后 Key

### 8.8 结论

| 问题 | 答案 |
|------|------|
| LeRobot 和 OpenPI 对 Key 有差异吗？ | **有** — 命名体系完全不同 |
| 差异有多大？ | **形式上差异大（不同命名约定），实质上差异小（各框架 transform 链各自解决）** |
| 转换脚本能解决差异吗？ | **能** — 正确将 OpenPI Key 映射到 LeRobot Key，已通过 Level 1+2 验证 |
| 需要额外代码修改吗？ | **不需要** — 当前方案完整、正确 |
