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

---

## 9. Normalization Stats 对齐深度分析

> **日期**: 2026-04-10
> **背景**: `pi05_alig.md` 2.2.4 节、`pi05_alig_2.md` 4.5 节、`pi05_alig_3.md` 5.2 节均提出了 Normalization Stats 对齐的需求。本节基于 OpenPI 和 LeRobot 两个框架的源码深入追踪，分析归一化统计的计算方式差异、实际数值偏差，以及现有转换脚本的对齐能力。

### 9.1 问题定义

Pi0.5 使用 **分位数归一化 (Quantile Normalization)** 将 state 和 action 映射到 [-1, 1] 范围，然后将归一化后的 state 离散化为 256 个 bins 嵌入到文本 prompt 中。这意味着 **q01/q99 分位数的准确性直接影响模型输入**。

如果 LeRobot 训练使用的分位数统计与 OpenPI 训练（base model 微调时）使用的分位数有显著偏差，归一化后的 state/action 分布会不同，可能导致：
- 离散化后的 bin 索引偏移 → 模型看到不同的 token 输入
- 训练时梯度信号与预训练权重期望的不一致

`pi05_alig_3.md` 9.1 节设定的对齐阈值: **max_diff < 1e-3**

### 9.2 OpenPI 的 Norm Stats 计算管线

#### 9.2.1 入口脚本

`openpi/scripts/compute_norm_stats.py`:

```python
# Line 102-103: 只计算两个 key
keys = ["state", "actions"]
stats = {key: normalize.RunningStats() for key in keys}

# Line 105-107: 遍历全部数据
for batch in tqdm.tqdm(data_loader, ...):
    for key in keys:
        stats[key].update(np.asarray(batch[key]))
```

**关键点**:
- 数据经过 `R1ProChassisInputs` transform 后再计算统计（即 key 已经是 `state`/`actions`）
- **使用全部数据** — 对 `r1_pro_data_convert_chassis` 数据集，即全部 64 个 episodes
- 不设 `max_frames` 限制，逐 batch 流式处理

#### 9.2.2 RunningStats 核心算法

`openpi/src/openpi/shared/normalize.py`, `RunningStats` 类:

```python
class RunningStats:
    def __init__(self):
        self._num_quantile_bins = 5000  # 直方图 bin 数

    def update(self, batch):
        batch = batch.reshape(-1, batch.shape[-1])  # 展平所有 batch 维
        # ... 更新 running mean, mean_of_squares, min, max
        # ... 更新直方图 (5000 bins)

    def _compute_quantiles(self, quantiles):
        """基于直方图近似计算分位数"""
        for q in quantiles:
            target_count = q * self._count
            for hist, edges in zip(self._histograms, self._bin_edges):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                q_values.append(edges[idx])  # ← 取 bin 左边界，无插值
```

**算法特征**:
1. **5000-bin 直方图** 近似（非精确计算）
2. **动态调整 bin 范围**: 当 min/max 变化时，通过 `_adjust_histograms()` 重新分配
3. **无 bin 内插值**: 分位数 = `edges[idx]`，即找到的 bin 的左边界。精度 ≈ `(max-min)/5000`
4. **action 展平**: `batch.reshape(-1, batch.shape[-1])` — 对 action chunk（多个时间步）的每一步视为独立样本

#### 9.2.3 归一化公式

`openpi/src/openpi/transforms.py:141-145`:

```python
def _normalize_quantile(self, x, stats: NormStats):
    q01, q99 = stats.q01[..., :x.shape[-1]], stats.q99[..., :x.shape[-1]]
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
```

**注意**: 分母**无条件加** `1e-6`，即使 `q99 - q01` 远大于 0。

#### 9.2.4 实际 Norm Stats (64 episodes)

文件: `openpi/assets/pi05_r1pro_chassis/r1_pro_data_convert_chassis/norm_stats.json`

state q01 前 5 维:
```
[-0.7135, -0.0435, -0.4656, -1.5907, -0.2319]
```
state q99 前 5 维:
```
[ 0.3784,  0.4162,  0.3410,  0.1285,  0.4576]
```
actions q01 前 5 维:
```
[-0.7149, -0.0437, -0.4654, -1.5919, -0.2317]
```
actions q99 前 5 维:
```
[ 0.3875,  0.4168,  0.3420,  0.1287,  0.4581]
```

### 9.3 LeRobot 的 Norm Stats 计算管线

#### 9.3.1 官方计算方式

`lerobot/src/lerobot/datasets/compute_stats.py`, `RunningQuantileStats` 类:

```python
class RunningQuantileStats:
    def __init__(self, quantile_list=None, num_quantile_bins=5000):
        self._num_quantile_bins = num_quantile_bins  # 同 OpenPI: 5000 bins

    def update(self, batch):
        batch = batch.reshape(-1, batch.shape[-1])  # 同 OpenPI: 展平
        # ... 完全相同的 running stats 更新逻辑

    def _compute_single_quantile(self, hist, edges, target_count):
        """与 OpenPI 的关键差异: 有 bin 内线性插值"""
        cumsum = np.cumsum(hist)
        idx = np.searchsorted(cumsum, target_count)
        if idx == 0: return edges[0]
        if idx >= len(cumsum): return edges[-1]
        # 线性插值 (OpenPI 没有这一步)
        count_before = cumsum[idx - 1]
        count_in_bin = cumsum[idx] - count_before
        fraction = (target_count - count_before) / count_in_bin
        return edges[idx] + fraction * (edges[idx + 1] - edges[idx])
```

**与 OpenPI 的异同**:

| 方面 | OpenPI `RunningStats` | LeRobot `RunningQuantileStats` |
|------|----------------------|-------------------------------|
| Bin 数 | 5000 | 5000 |
| 数据展平 | `reshape(-1, shape[-1])` | `reshape(-1, shape[-1])` (相同) |
| Running mean/std | 增量更新 | 增量更新 (相同) |
| 直方图调整 | `np.histogram(old_edges, bins=new_edges, weights=hist)` | 按 old bin center 逐个重分配 |
| **分位数插值** | **无** (`edges[idx]`) | **有** (bin 内线性插值) |
| 计算分位数 | [0.01, 0.99] | [0.01, 0.10, 0.50, 0.90, 0.99] |

#### 9.3.2 归一化公式

`lerobot/src/lerobot/processor/normalize_processor.py:362-377`:

```python
# QUANTILES mode
denom = q99 - q01
denom = torch.where(
    denom == 0,
    torch.tensor(self.eps, device=tensor.device, dtype=tensor.dtype),  # eps=1e-8
    denom
)
return 2.0 * (tensor - q01) / denom - 1.0
```

**与 OpenPI 的差异**: 分母仅在 `q99==q01` 时用 `1e-8` 替代（**条件加**），而非无条件加 `1e-6`。

#### 9.3.3 Per-Episode 聚合

`compute_stats.py:565-602`, `aggregate_feature_stats()`:

```python
# 分位数聚合: 加权平均 (不是重新计算全局分位数)
for q_key in quantile_keys:
    quantile_values = np.stack([s[q_key] for s in stats_ft_list])
    weighted_quantiles = quantile_values * counts
    aggregated[q_key] = weighted_quantiles.sum(axis=0) / total_count
```

**问题**: 分位数的加权平均 **不等于** 全局分位数。这是一个已知的近似，对于分布相似的 episode 误差较小，但在 episode 间分布差异大时会引入偏差。

### 9.4 转换脚本的 Norm Stats 计算方式

#### 9.4.1 Phase 2.5 实现

`bt/pi05/alig/data/convert_r1pro_to_lerobot.py:285-419`, `phase2_5_compute_quantiles()`:

```python
# 读取全部数据 parquet，拼接
all_data = {key: [] for key in numeric_features}
for df_path in data_files:
    df = pd.read_parquet(df_path, columns=numeric_features)
    for key in numeric_features:
        arr = np.stack(col.values)
        all_data[key].append(arr)

# 精确计算分位数 (numpy linear interpolation)
for key in numeric_features:
    concatenated = np.concatenate(all_data[key], axis=0).astype(np.float64)
    for q, q_key in zip(quantiles, q_keys):
        q_val = np.quantile(concatenated, q, axis=0)  # ← 精确计算
```

**特征**:
1. **精确计算**: 使用 `np.quantile()`（线性插值），不是直方图近似
2. **全局计算**: 拼接所有帧后一次计算，不是 per-episode 聚合
3. **float64 精度**: 显式转为 float64 后计算
4. **数据范围**: 取决于 `--sample-episodes` 参数

### 9.5 三种计算方式对比

| 方面 | OpenPI (基准) | LeRobot 官方 | 转换脚本 Phase 2.5 |
|------|--------------|-------------|-------------------|
| **算法** | 5000-bin 直方图，无插值 | 5000-bin 直方图，有 bin 内插值 | `np.quantile()` 精确计算 |
| **数据范围** | 全部 64 episodes | per-episode → 加权平均聚合 | `--sample-episodes N` 指定 |
| **精度** | ≈ (max-min)/5000 | ≈ (max-min)/5000 (略优) | 精确 (float64 线性插值) |
| **数据流** | 经 R1ProChassisInputs transform 后 | 直接从 parquet 读取 | 直接从 parquet 读取 |
| **内存** | 流式 O(bins × dims) | per-episode 流式 | O(全部数据) 全量加载 |

### 9.6 实际数值偏差分析

#### 9.6.1 State q01 对比 (23 维)

| 维度 | OpenPI (64ep) | 转换脚本 (10ep) | 绝对偏差 | 超过 1e-3？ |
|------|--------------|----------------|---------|------------|
| 0 | -0.7135 | -0.6857 | **0.0278** | **是** |
| 1 | -0.0435 | -0.0383 | 0.0052 | **是** |
| 2 | -0.4656 | -0.4647 | 0.0010 | 约等于 |
| 3 | -1.5907 | -1.5872 | 0.0035 | **是** |
| 4 | -0.2319 | -0.2285 | 0.0034 | **是** |
| 5 | -0.9907 | -1.0453 | **0.0547** | **是** |
| 6 | -0.3509 | -0.2784 | **0.0726** | **是** |
| 7 | -1.2799 | -1.2224 | **0.0575** | **是** |
| 8 | -0.1801 | -0.1922 | 0.0121 | **是** |
| 9 | -0.3406 | -0.1702 | **0.1704** | **是** |
| 10 | -1.4571 | -1.4869 | 0.0298 | **是** |
| 11 | -0.4894 | -0.4323 | **0.0570** | **是** |
| 12 | -0.4993 | -0.4912 | 0.0081 | **是** |
| 13 | -0.2794 | -0.2314 | **0.0480** | **是** |
| 14 | 2.7527 | 2.8388 | 0.0862 | **是** |
| 15 | 2.8153 | 2.8388 | 0.0235 | **是** |
| 16 | 0.8993 | 0.8993 | 0.0000 | 否 |
| 17 | -1.5005 | -1.5005 | 0.0000 | 否 |
| 18 | -0.7005 | -0.7005 | 0.0000 | 否 |
| 19 | -0.0005 | -0.0005 | 0.0000 | 否 |
| 20 | -0.0081 | -0.0060 | 0.0021 | **是** |
| 21 | -0.0080 | -0.0050 | 0.0030 | **是** |
| 22 | -0.0070 | -0.0050 | 0.0020 | **是** |

**Max state q01 diff = 0.1704** (维度 9)，远超 1e-3 阈值。

#### 9.6.2 State q99 对比 (选取偏差较大的维度)

| 维度 | OpenPI (64ep) | 转换脚本 (10ep) | 绝对偏差 |
|------|--------------|----------------|---------|
| 2 | 0.3410 | 0.2304 | **0.1106** |
| 7 | 0.1308 | 0.0825 | **0.0483** |
| 9 | 0.6521 | 0.5180 | **0.1341** |
| 13 | 0.3169 | 0.2655 | **0.0514** |

**Max state q99 diff = 0.1341** (维度 9)。

#### 9.6.3 Actions 偏差类似

| Key | Max Diff (q01) | Max Diff (q99) | 最大偏差维度 |
|-----|---------------|---------------|------------|
| state | 0.1704 | 0.1341 | 9 |
| actions | ~0.1 级别 | ~0.1 级别 | 类似 |

#### 9.6.4 偏差对 256-bin 离散化的影响

Pi0.5 将归一化后的 state 离散化为 256 bins: `bin = round((normalized + 1) / 2 * 255)`

以维度 9 为例:
- OpenPI q01 = -0.3406, q99 = 0.6521 → range = 0.9927
- 转换脚本 q01 = -0.1702, q99 = 0.5180 → range = 0.6882

对于原始值 x = 0.0 (state dim 9 的中位数附近):
- OpenPI 归一化: `(0 - (-0.3406)) / (0.9927 + 1e-6) * 2 - 1 = -0.3135`
- 转换脚本归一化: `(0 - (-0.1702)) / (0.6882 + 1e-8) * 2 - 1 = -0.5055`
- Bin 差异: `round((-0.3135+1)/2 * 255) = 87` vs `round((-0.5055+1)/2 * 255) = 63` → **24 bins 偏差!**

**结论: 当前 10-episode 采样的 norm stats 偏差严重，足以改变 state token 输入，不可接受。**

### 9.7 偏差根因分析

#### 9.7.1 主要原因: 数据采样范围 (贡献 ~95% 偏差)

转换脚本使用 `--sample-episodes 10` 只取了 64 个 episode 中的 10 个。R1 Pro 数据中不同 episode 的关节运动范围差异大（不同任务场景），10 个 episode 的 q01/q99 不能代表全部 64 个 episode 的分布。

例如维度 9 (右臂某关节):
- 64 episodes 覆盖了更极端的运动: q01 = -0.3406
- 10 episodes 只覆盖了部分: q01 = -0.1702（缺少极端样本）

#### 9.7.2 次要原因: 计算方法差异 (贡献 ~5% 偏差)

即使使用全部 64 episodes，三种方法仍有差异:

| 对比 | 预计偏差量级 | 原因 |
|------|------------|------|
| OpenPI 直方图 vs 精确 `np.quantile()` | ~(max-min)/5000 ≈ 0.0002 | 5000 bins 精度足够 |
| OpenPI 无插值 vs LeRobot 有插值 | ~0.5 × bin_width ≈ 0.0001 | bin 内插值修正 |
| 两者综合 | < 0.001 | **在 1e-3 阈值内** |

**结论: 在相同数据范围下，计算方法差异可忽略。主要问题是采样数据不足。**

#### 9.7.3 归一化公式差异 (可忽略)

| | OpenPI | LeRobot |
|-|--------|---------|
| 公式 | `(x-q01) / (q99-q01 + 1e-6) * 2 - 1` | `2*(x-q01) / max(q99-q01, 1e-8) - 1` |
| 对正常维度 (range >> 1e-6) | `/ (range + 1e-6)` ≈ `/ range` | `/ range` |
| 差异 | `range / (range + 1e-6) - 1` ≈ `1e-6 / range` ≈ 1e-6 | (基准) |
| 对零方差维度 (q99==q01) | `/ 1e-6` → 可能溢出 | `/ 1e-8` → 更极端但被 where 保护 |

**对 R1 Pro 数据**: 大多数维度 range > 0.1，公式差异 < 1e-5。零方差维度 (如 actions dim 17/19/22) 输出都是 -1.0，两个公式结果相同。**可忽略。**

### 9.8 现有代码能否解决对齐问题？

#### 9.8.1 转换脚本评估

| 问题 | 现状 | 能否对齐 |
|------|------|---------|
| 采样范围不足 | `--sample-episodes 10` → 10/64 episodes | ❌ 不传 `--sample-episodes` 可解决 |
| 计算方法差异 | `np.quantile()` 精确 vs OpenPI 直方图近似 | ⚠️ 差异极小 (~0.0002)，可接受 |
| 归一化公式差异 | 条件 vs 无条件 epsilon | ✅ 差异可忽略 (<1e-5) |
| Key 命名差异 | `observation.state`/`action` vs `state`/`actions` | ✅ 已正确映射 |

**结论**: 转换脚本的**代码逻辑**是正确的，但**使用方式** (`--sample-episodes 10`) 导致了不可接受的偏差。

#### 9.8.2 两种解决方案

**方案 A: 全量转换 (推荐用于最终训练)**

不使用 `--sample-episodes`，转换全部 64 episodes:

```bash
python bt/pi05/alig/data/convert_r1pro_to_lerobot.py \
    --input /mnt/r/share/lkx/pi/data/r1_pro_data_convert_chassis \
    --output /path/to/r1_pro_chassis_full_v30
```

此方案让 `phase2_5_compute_quantiles()` 从全部 64 episodes 计算分位数。由于 `np.quantile()` 的精度优于 OpenPI 的 5000-bin 直方图，在相同数据范围下偏差将 < 0.001。

- **优点**: 简单，无需修改代码
- **缺点**: 需要 ~90GB 磁盘空间，约 30 分钟转换时间

**方案 B: 直接使用 OpenPI 的 norm_stats.json (推荐用于精确对齐)**

如果目标是与 OpenPI 训练结果精确对齐（例如验证两框架的输出一致性），最可靠的方式是直接复用 OpenPI 已计算好的 norm_stats.json，将其中的 q01/q99 值注入到 LeRobot 的 stats.json 中。

这消除了所有计算方法差异，确保归一化行为完全一致（除了公式中极小的 epsilon 差异）。

核心代码:

```python
import json
import numpy as np
from pathlib import Path
from openpi.shared.normalize import load as load_openpi_norm_stats

def inject_openpi_stats(
    openpi_norm_stats_dir: Path,
    lerobot_stats_path: Path,
) -> None:
    """
    将 OpenPI 的 norm_stats.json 中的 q01/q99 注入到 LeRobot 的 stats.json。
    
    OpenPI key → LeRobot key 映射:
      "state"   → "observation.state"
      "actions" → "action"
    """
    KEY_MAP = {
        "state": "observation.state",
        "actions": "action",
    }
    
    # 1. 加载 OpenPI norm_stats
    openpi_stats = load_openpi_norm_stats(openpi_norm_stats_dir)
    # openpi_stats: {"state": NormStats(mean, std, q01, q99),
    #                "actions": NormStats(mean, std, q01, q99)}

    # 2. 加载 LeRobot stats.json
    with open(lerobot_stats_path) as f:
        lr_stats = json.load(f)

    # 3. 注入 q01/q99
    for openpi_key, lerobot_key in KEY_MAP.items():
        if openpi_key not in openpi_stats:
            raise KeyError(f"OpenPI norm_stats 缺少 key: {openpi_key}")
        if lerobot_key not in lr_stats:
            raise KeyError(f"LeRobot stats.json 缺少 key: {lerobot_key}")

        ns = openpi_stats[openpi_key]
        lr_stats[lerobot_key]["q01"] = ns.q01.tolist()
        lr_stats[lerobot_key]["q99"] = ns.q99.tolist()
        # 可选: 同时注入 mean/std 以保持完全一致
        # lr_stats[lerobot_key]["mean"] = ns.mean.tolist()
        # lr_stats[lerobot_key]["std"] = ns.std.tolist()

    # 4. 写回
    with open(lerobot_stats_path, "w") as f:
        json.dump(lr_stats, f, indent=4)

    print(f"已注入 OpenPI norm stats → {lerobot_stats_path}")
```

使用方式:

```python
inject_openpi_stats(
    openpi_norm_stats_dir=Path("openpi/assets/pi05_r1pro_chassis/r1_pro_data_convert_chassis"),
    lerobot_stats_path=Path("bt/pi05/alig/data/r1_pro_chassis_v30/meta/stats.json"),
)
```

- **优点**: 精确对齐，偏差 = 0 (仅剩公式 epsilon 差异 < 1e-5)
- **缺点**: 依赖 OpenPI 预计算的 norm_stats.json 已存在

### 9.9 验证方案

无论选择方案 A 还是 B，都需要验证对齐效果。验证脚本:

```python
import json
import numpy as np
from pathlib import Path

def verify_norm_stats_alignment(
    openpi_norm_stats_path: Path,
    lerobot_stats_path: Path,
    threshold: float = 1e-3,
) -> bool:
    """
    验证 LeRobot stats.json 与 OpenPI norm_stats.json 的 q01/q99 对齐程度。
    """
    KEY_MAP = {
        "state": "observation.state",
        "actions": "action",
    }

    # 加载 OpenPI
    with open(openpi_norm_stats_path) as f:
        openpi_data = json.load(f)
    openpi_stats = openpi_data["norm_stats"]

    # 加载 LeRobot
    with open(lerobot_stats_path) as f:
        lr_stats = json.load(f)

    all_pass = True
    for openpi_key, lr_key in KEY_MAP.items():
        for q_name in ["q01", "q99"]:
            openpi_vals = np.array(openpi_stats[openpi_key][q_name])
            lr_vals = np.array(lr_stats[lr_key][q_name])
            
            diff = np.abs(openpi_vals - lr_vals)
            max_diff = diff.max()
            max_dim = diff.argmax()
            
            status = "PASS" if max_diff < threshold else "FAIL"
            if status == "FAIL":
                all_pass = False
            
            print(f"  {lr_key}.{q_name}: max_diff={max_diff:.6f} "
                  f"(dim {max_dim}) [{status}]")

    return all_pass

# 使用:
# verify_norm_stats_alignment(
#     "openpi/assets/pi05_r1pro_chassis/r1_pro_data_convert_chassis/norm_stats.json",
#     "bt/pi05/alig/data/r1_pro_chassis_v30/meta/stats.json",
# )
```

### 9.10 推荐策略

| 场景 | 推荐方案 | 预计偏差 |
|------|---------|---------|
| **快速验证/开发** | 方案 A: 全量转换 (64 episodes) | max_diff < 0.001 |
| **精确对齐验证** | 方案 B: 注入 OpenPI norm_stats | max_diff ≈ 0 |
| **生产训练** | 方案 B → 全量数据转换 + 注入 OpenPI stats | max_diff ≈ 0 |
| **当前采样 10 episodes** | ❌ 不可接受 | max_diff > 0.17 |

### 9.11 结论

| 问题 | 答案 |
|------|------|
| LeRobot 和 OpenPI 的 norm stats 计算方法有差异吗？ | **有** — 算法微差 (插值 vs 无插值)，但相同数据下差异 < 0.001 |
| 当前转换后的 stats 对齐吗？ | **不对齐** — max_diff = 0.17，因为只用了 10/64 episodes |
| 现有代码能解决吗？ | **代码正确，使用方式需调整** — 不传 `--sample-episodes` 即可 |
| 最优对齐方案？ | **直接注入 OpenPI 的 norm_stats.json** → 零计算偏差 |
| 归一化公式差异影响？ | **可忽略** — 对正常维度差异 < 1e-5 |

### 9.12 Norm Stats 计算方法优劣深度对比

> **日期**: 2026-04-11
> **目的**: 深入比较转换脚本 (`bt/pi05/alig/data/`) 的 `np.quantile()` 精确计算方式与 OpenPI 的 5000-bin 直方图近似方式，判断哪种更精确、更利于训练，并据此设计转换脚本的 norm stats 选项。

#### 9.12.1 三种计算方法的算法本质

**方法 1: OpenPI `RunningStats` — 5000-bin 直方图，无 bin 内插值**

`openpi/src/openpi/shared/normalize.py:106-117`:

```python
def _compute_quantiles(self, quantiles):
    for q in quantiles:
        target_count = q * self._count
        for hist, edges in zip(self._histograms, self._bin_edges):
            cumsum = np.cumsum(hist)
            idx = np.searchsorted(cumsum, target_count)
            q_values.append(edges[idx])  # ← 取 bin 左边界，无插值
```

- 将数据范围 [min, max] 均匀划分为 5000 个 bin
- 累积直方图找到第一个 cumsum ≥ target_count 的 bin
- 返回该 bin 的**左边界**作为分位数值
- 精度上界: `(max - min) / 5000`，即每个 bin 的宽度
- 分位数总是 bin 边界值之一，是一种阶梯函数近似

**方法 2: LeRobot 官方 `RunningQuantileStats` — 5000-bin 直方图，有 bin 内线性插值**

`lerobot/src/lerobot/datasets/compute_stats.py:170-190`:

```python
def _compute_single_quantile(self, hist, edges, target_count):
    cumsum = np.cumsum(hist)
    idx = np.searchsorted(cumsum, target_count)
    if idx == 0: return edges[0]
    if idx >= len(cumsum): return edges[-1]
    # 关键差异: bin 内线性插值
    count_before = cumsum[idx - 1]
    count_in_bin = cumsum[idx] - count_before
    fraction = (target_count - count_before) / count_in_bin
    return edges[idx] + fraction * (edges[idx + 1] - edges[idx])
```

- 同样 5000 bins，同样的直方图累积查找
- 但在找到目标 bin 后，**根据该 bin 内已消耗的样本比例做线性插值**
- 精度: 优于方法 1，因为在 bin 内做了平滑，但仍受限于 bin 宽度内的均匀分布假设
- 实际精度约 `(max - min) / 5000 / 2` (插值平均减半了误差)

**方法 3: 转换脚本 `np.quantile()` — 精确线性插值**

`bt/pi05/alig/data/convert_r1pro_to_lerobot.py:340-342`:

```python
concatenated = np.concatenate(all_data[key], axis=0).astype(np.float64)
q_val = np.quantile(concatenated, q, axis=0)
```

- `np.quantile()` 默认使用 `method='linear'`（NumPy ≥1.22）
- 内部算法: 先对数据排序，找到分位数对应的精确位置 `i = q * (n-1)`，然后在相邻排序值之间做线性插值: `x[floor(i)] + (i - floor(i)) * (x[ceil(i)] - x[floor(i)])`
- **精度: 精确到 float64 浮点精度** (~15-16 位有效数字)
- 无 bin 近似，无均匀分布假设
- 代价: 需要 O(n) 内存和 O(n log n) 排序时间

#### 9.12.2 精度定量对比

以 R1 Pro Chassis 数据为例，state 维度 0 的数据范围 [min, max] ≈ [-0.99, 0.46]，range ≈ 1.45:

| 方法 | 单 bin 宽度 | 最大分位数误差 | 相对误差 |
|------|-----------|-------------|---------|
| OpenPI (5000 bins, 无插值) | 1.45/5000 = 0.00029 | 0.00029 | 0.020% |
| LeRobot (5000 bins, 有插值) | 0.00029 | ~0.00015 | 0.010% |
| 转换脚本 (`np.quantile()`) | — | ~1e-15 | ~1e-13% |

**精度排序: 转换脚本 >> LeRobot 官方 > OpenPI**

但这仅是**计算精度**。对训练而言还需考虑其他因素。

#### 9.12.3 Action Chunk 展平的影响

OpenPI 在计算 norm stats 时，数据经过 `delta_timestamps` 加载，每个样本包含 `action_horizon=50` 步的 action chunk，然后在 `update()` 中被展平:

`normalize.py:37`: `batch = batch.reshape(-1, batch.shape[-1])`

即 shape `(batch_size, 50, 23)` → `(batch_size × 50, 23)`。每个 action frame 被多次采样。

**转换脚本中每个 parquet 行只有单步 action** (shape `[23]`)。

**这是否造成分布差异？** 分析:

- LeRobot 的 `delta_timestamps` 不跨 episode 边界
- 对于长度 N 的 episode，frame i 在 action chunk 中被采样的次数 ≈ min(i+1, 50, N-i)
- 中间帧被采样 ~50 次，边界帧被采样较少
- 但所有帧的 action 值本身不变，只是采样权重略有不同

**结论**: 对于 R1 Pro 数据 (episode 长度 ~1000 帧，action_horizon=50)，边界效应影响约 5% 的数据。由于分位数是基于排序位置的统计量，重复采样对分位数的影响极小 — 等价于对边界帧轻微欠采样。**不构成实质性差异。**

验证: 对均匀分布数据，无论每个值重复 1 次还是 50 次，q01 和 q99 保持不变。

#### 9.12.4 直方图动态调整的精度损失

OpenPI `_adjust_histograms()` (`normalize.py:88-98`):

```python
def _adjust_histograms(self):
    for i in range(len(self._histograms)):
        old_edges = self._bin_edges[i]
        new_edges = np.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1)
        # 将旧直方图的 counts 重新分配到新 bins
        new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])
        self._histograms[i] = new_hist
        self._bin_edges[i] = new_edges
```

每次 min/max 扩大时，旧 bin 的 counts 通过 `np.histogram(old_edges[:-1], ...)` 重新分配。这个过程:
- 将旧 bin 的所有 count 集中到一个新 bin 的位置 (`old_edges[:-1]` 是旧的左边界)
- 如果旧 bin 跨越多个新 bin，count 不会被正确分割
- **多次调整会累积误差**

LeRobot 的 `_adjust_histograms()` (`compute_stats.py:123-148`) 做了改进:
```python
old_centers = (old_edges[:-1] + old_edges[1:]) / 2  # 用 bin 中心而非左边界
for old_center, count in zip(old_centers, old_hist):
    bin_idx = np.searchsorted(new_edges, old_center) - 1
    new_hist[bin_idx] += count
```

用 bin 中心重新分配，比 OpenPI 的左边界方式更准确。

**但两者都有信息损失** — 一旦数据被 bin 化，原始分布细节就丢失了。`np.quantile()` 不存在这个问题，因为它直接操作原始数据。

#### 9.12.5 LeRobot 官方 Per-Episode 加权平均分位数的缺陷

`compute_stats.py:596-600`:

```python
quantile_values = np.stack([s[q_key] for s in stats_ft_list])
weighted_quantiles = quantile_values * counts
aggregated[q_key] = weighted_quantiles.sum(axis=0) / total_count
```

**数学问题**: 分位数不满足线性可加性。全局 q01 ≠ 各 episode q01 的加权平均。

反例: Episode A (100 帧) 的 q01 = 0.0，Episode B (100 帧) 的 q01 = 10.0。
- 加权平均 q01 = (0.0 × 100 + 10.0 × 100) / 200 = 5.0
- 但真实全局 q01 取决于 A 和 B 的数据分布，可能远不是 5.0

转换脚本的全局 `np.quantile()` 不存在此问题，因为它在拼接后的完整数据上直接计算。

#### 9.12.6 对训练质量的影响

分位数的精确性通过以下路径影响训练:

```
q01/q99 → 归一化公式 → normalized state → 256-bin 离散化 → text token → model input
                     → normalized action → loss 计算 → 梯度
```

**State 路径** (影响较大):
- normalized state 被离散化为 256 bins 嵌入到 prompt
- 1 个 bin = 1/256 ≈ 0.0039 的归一化范围
- 分位数误差如果导致归一化后偏移 > 0.004，就会改变 bin 索引
- OpenPI 5000-bin 误差 ~0.0003 → 归一化后偏移 ~0.0003/range × 2 ≈ 0.0006 → **不会改变 bin** ✓
- `np.quantile()` 误差 ~0 → **完全不会改变 bin** ✓
- **两者在 state 离散化上等价**

**Action 路径** (影响较小):
- action 是连续值，不做离散化
- 分位数微小偏差仅导致归一化后的值轻微偏移
- 对 loss 梯度的影响与偏移量成正比
- **两者在 action loss 上差异可忽略**

#### 9.12.7 综合评判

| 维度 | OpenPI 直方图 | `np.quantile()` (转换脚本) | 优胜 |
|------|-------------|-------------------------|------|
| **数学精度** | ~0.0003 误差 | ~0 误差 | `np.quantile()` |
| **直方图调整损失** | 有 (累积误差) | 无 | `np.quantile()` |
| **分位数聚合** | 全局流式 (准确) | 全局一次性 (准确) | 平手 |
| **内存效率** | O(bins × dims) ≈ 920KB | O(data) ≈ 数 GB | OpenPI |
| **增量更新** | 支持流式处理 | 不支持，需全量加载 | OpenPI |
| **对训练质量影响** | 与精确值差异不影响 bin | 精确 | **实质平手** |
| **实现复杂度** | 较高 | 极低 (3 行) | `np.quantile()` |

### 9.13 结论与决策

**结论: `np.quantile()` 在数学精度和实现简洁性上都优于 OpenPI 的直方图方法**，但对实际训练质量无实质影响（因为 5000-bin 精度已足够不改变 256-bin 离散化结果）。

两种方法在相同数据上的差异 < 0.001，远小于数据采样范围差异（0.17）。**影响训练的关键因素是"用多少数据计算分位数"，而非"用什么算法计算分位数"。**

**决策**: 转换脚本保留现有的 `np.quantile()` 精确计算方式，同时增加 `--norm-stats-path` 选项，允许用户直接导入已有的 norm stats 文件（如 OpenPI 的 `norm_stats.json`）以实现精确对齐。

### 9.14 实现方案: 双模式 Norm Stats

在 `convert_r1pro_to_lerobot.py` 中为 `--norm-stats-path` 参数提供两种模式:

| 模式 | 命令 | 行为 |
|------|------|------|
| **精确计算** (默认) | 不传 `--norm-stats-path` | Phase 2.5 使用 `np.quantile()` 从转换后数据精确计算 |
| **导入已有 stats** | `--norm-stats-path /path/to/norm_stats.json` | 跳过 Phase 2.5 计算，从指定文件导入 q01/q99 到 stats.json |

#### 9.14.1 命令行接口

```bash
# 模式 1: 精确计算 (默认) — 从转换后的数据计算分位数
python bt/pi05/alig/data/convert_r1pro_to_lerobot.py \
    --input /mnt/r/share/lkx/pi/data/r1_pro_data_convert_chassis \
    --output /path/to/output_v30

# 模式 2: 导入 OpenPI norm_stats.json
python bt/pi05/alig/data/convert_r1pro_to_lerobot.py \
    --input /mnt/r/share/lkx/pi/data/r1_pro_data_convert_chassis \
    --output /path/to/output_v30 \
    --norm-stats-path /mnt/r/share/lkx/pi/openpi/assets/pi05_r1pro_chassis/r1_pro_data_convert_chassis/norm_stats.json
```

#### 9.14.2 核心实现

**1. 参数解析新增:**

```python
parser.add_argument(
    "--norm-stats-path", type=Path, default=None,
    help="导入已有的 norm stats 文件 (OpenPI norm_stats.json 格式)。"
         "若不指定，则从转换后数据精确计算分位数。"
)
```

**2. 导入逻辑 (`phase2_5_import_norm_stats`):**

```python
# OpenPI norm_stats.json key → LeRobot stats.json key
NORM_STATS_KEY_MAP = {
    "state": "observation.state",
    "actions": "action",
}

def phase2_5_import_norm_stats(dataset_dir: Path, norm_stats_path: Path) -> None:
    """从已有的 norm_stats.json 导入 q01/q99 到 LeRobot stats.json。"""
    logger.info("Phase 2.5: 导入已有 norm stats: %s", norm_stats_path)

    with open(norm_stats_path) as f:
        src = json.load(f)

    # 支持 OpenPI 格式 (嵌套在 "norm_stats" 下) 和直接格式
    if "norm_stats" in src:
        src = src["norm_stats"]

    stats_path = dataset_dir / "meta" / "stats.json"
    with open(stats_path) as f:
        stats = json.load(f)

    for src_key, dst_key in NORM_STATS_KEY_MAP.items():
        if src_key not in src:
            logger.warning("  norm stats 文件缺少 key: %s, 跳过", src_key)
            continue
        if dst_key not in stats:
            logger.warning("  stats.json 缺少 key: %s, 跳过", dst_key)
            continue

        src_stats = src[src_key]
        for q_key in ["q01", "q99"]:
            if q_key in src_stats:
                stats[dst_key][q_key] = src_stats[q_key]
                logger.info("  %s.%s → %s.%s (%d dims)",
                            src_key, q_key, dst_key, q_key, len(src_stats[q_key]))

        # 同时导入其它可用的分位数
        for q_key in ["q10", "q50", "q90"]:
            if q_key in src_stats:
                stats[dst_key][q_key] = src_stats[q_key]

    # 补充图像和元数据特征的默认分位数 (与 phase2_5_compute_quantiles 相同逻辑)
    # ... (复用现有图像分位数近似代码)

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    logger.info("Phase 2.5 完成: 已从 %s 导入 norm stats", norm_stats_path.name)
```

**3. 主流程分派:**

```python
def main():
    # ... (现有参数解析)
    args = parser.parse_args()

    # Phase 0+1: 列名重命名 + Episode 采样
    phase0_1_rename_and_sample(...)

    # Phase 2: v2.1 → v3.0 转换
    phase2_convert_v21_to_v30(...)

    # Phase 2.5: Norm Stats — 二选一
    if args.norm_stats:
        phase2_5_import_norm_stats(output_dir, args.norm_stats)
    else:
        phase2_5_compute_quantiles(output_dir)

    # Phase 3: 验证
    if not args.skip_verify:
        phase3_verify(output_dir)
```

#### 9.14.3 两种模式的适用场景

| 场景 | 推荐模式 | 原因 |
|------|---------|------|
| **微调 OpenPI 预训练模型** | 导入 (`--norm-stats-path`) | 与预训练使用的归一化完全一致 |
| **从头训练新模型** | 精确计算 (默认) | 分位数基于当前数据，更准确 |
| **对齐验证 (两框架输出一致性)** | 导入 (`--norm-stats-path`) | 消除统计计算差异 |
| **新数据集 (无已有 norm stats)** | 精确计算 (默认) | 唯一选择 |
| **采样子集快速验证** | 导入 (`--norm-stats-path`) | 避免子集分位数不准确 |

### 9.15 测试方案: 双模式 Norm Stats

#### 9.15.1 单元测试: 导入模式正确性

```python
def test_import_norm_stats():
    """验证导入后 stats.json 中的 q01/q99 与源文件完全一致。"""
    import tempfile, shutil, json
    from pathlib import Path

    # 准备: 复制一个已转换的小数据集
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir) / "test_v30"
        shutil.copytree("bt/pi05/alig/data/r1_pro_test_data_v30", test_dir)

        # 执行导入
        norm_stats_path = Path(
            "openpi/assets/pi05_r1pro_chassis/r1_pro_data_convert_chassis/norm_stats.json"
        )
        phase2_5_import_norm_stats(test_dir, norm_stats_path)

        # 验证
        with open(test_dir / "meta" / "stats.json") as f:
            stats = json.load(f)
        with open(norm_stats_path) as f:
            src = json.load(f)["norm_stats"]

        import numpy as np
        for src_key, dst_key in NORM_STATS_KEY_MAP.items():
            for q_key in ["q01", "q99"]:
                np.testing.assert_array_equal(
                    stats[dst_key][q_key],
                    src[src_key][q_key],
                    err_msg=f"{dst_key}.{q_key} 不匹配"
                )
```

#### 9.15.2 集成测试: 两种模式输出对比

```bash
# 1. 模式 A: 精确计算 (全量 64 episodes)
python bt/pi05/alig/data/convert_r1pro_to_lerobot.py \
    --input /mnt/r/share/lkx/pi/data/r1_pro_data_convert_chassis \
    --output /tmp/test_compute_v30

# 2. 模式 B: 导入 OpenPI norm stats
python bt/pi05/alig/data/convert_r1pro_to_lerobot.py \
    --input /mnt/r/share/lkx/pi/data/r1_pro_data_convert_chassis \
    --output /tmp/test_import_v30 \
    --norm-stats-path openpi/assets/pi05_r1pro_chassis/r1_pro_data_convert_chassis/norm_stats.json

# 3. 比对两者的 stats.json
python -c "
import json, numpy as np
with open('/tmp/test_compute_v30/meta/stats.json') as f: a = json.load(f)
with open('/tmp/test_import_v30/meta/stats.json') as f: b = json.load(f)
for key in ['observation.state', 'action']:
    for q in ['q01', 'q99']:
        diff = np.abs(np.array(a[key][q]) - np.array(b[key][q])).max()
        print(f'{key}.{q}: max_diff = {diff:.6f}')
"
# 预期: 模式 A (全量) 与模式 B 的差异 < 0.001
```

#### 9.15.3 端到端验证: Level 1+2

```bash
# 使用导入模式转换后，运行 Pi0.5 兼容性验证
python bt/pi05/alig/data/verify_pi05.py \
    --dataset-dir /tmp/test_import_v30
# 预期: Level 1+2 全部通过，normalized range 在 [-1, 1] 附近
```

#### 9.15.4 Norm Stats 对齐验证

```bash
# 使用 9.9 节的 verify_norm_stats_alignment 函数
python -c "
import json, numpy as np
# 加载
with open('openpi/assets/pi05_r1pro_chassis/r1_pro_data_convert_chassis/norm_stats.json') as f:
    openpi = json.load(f)['norm_stats']
with open('/tmp/test_import_v30/meta/stats.json') as f:
    lr = json.load(f)
# 对比
for src_k, dst_k in [('state','observation.state'), ('actions','action')]:
    for q in ['q01','q99']:
        diff = np.abs(np.array(openpi[src_k][q]) - np.array(lr[dst_k][q])).max()
        status = 'PASS' if diff < 1e-3 else 'FAIL'
        print(f'{dst_k}.{q}: max_diff={diff:.8f} [{status}]')
"
# 导入模式预期: max_diff = 0.0 (完全一致)
# 精确计算模式 (全量) 预期: max_diff < 0.001
```

---

## 10. 使用指南 V2

> **日期**: 2026-04-11
> **适用代码版本**: 双模式 Norm Stats (`--norm-stats-path`) 实现后的最新版本
> **替代**: 第 7 节 (使用指南 V1) — V1 未涵盖导入模式和验证脚本的完整说明

### 10.1 文件清单

`bt/pi05/alig/data/` 目录下有 3 个脚本和 2 个输出目录:

| 文件 | 类型 | 行数 | 说明 |
|------|------|------|------|
| `convert_r1pro_to_lerobot.py` | 转换脚本 | 675 | 主脚本：将 OpenPI 格式 R1 Pro 数据集转换为 LeRobot v3.0 格式 |
| `verify_pi05.py` | 验证脚本 | 334 | Pi0.5 兼容性验证（3 个级别） |
| `run_convert.sh` | 一键脚本 | 83 | 一键运行转换 + 验证 (两个数据集) |
| `r1_pro_test_data_v30/` | 输出目录 | — | 转换后的测试集 (4 episodes, 3366 frames) |
| `r1_pro_chassis_v30/` | 输出目录 | — | 转换后的训练集采样 (10 episodes, 9830 frames) |

### 10.2 convert_r1pro_to_lerobot.py — 转换脚本

#### 10.2.1 功能概述

将 OpenPI 格式的 R1 Pro LeRobot v2.1 数据集转换为 LeRobot 标准 v3.0 格式。转换分为 4 个 Phase:

```
Phase 0+1  →  Phase 2  →  Phase 2.5  →  Phase 3
采样+重命名    v2.1→v3.0    Norm Stats    基本验证
```

#### 10.2.2 常量定义

**`COLUMN_RENAME_MAP`** (line 54-60) — OpenPI → LeRobot 列名映射:

| OpenPI 列名 | LeRobot 列名 |
|-------------|-------------|
| `head_rgb` | `observation.images.head_rgb` |
| `left_wrist_rgb` | `observation.images.left_wrist_rgb` |
| `right_wrist_rgb` | `observation.images.right_wrist_rgb` |
| `state` | `observation.state` |
| `actions` | `action` |

**`IMAGE_COLUMNS`** (line 62-66) — 需要标记为 HuggingFace `Image()` 类型的列名列表。

**`NORM_STATS_KEY_MAP`** (line 69-72) — 导入模式用，OpenPI norm_stats.json 的 key → LeRobot stats.json 的 key:

| 源 Key (OpenPI) | 目标 Key (LeRobot) |
|-----------------|-------------------|
| `state` | `observation.state` |
| `actions` | `action` |

#### 10.2.3 辅助函数

| 函数 | 行号 | 作用 |
|------|------|------|
| `sample_episode_indices(total, n, seed)` | 75 | 随机采样 n 个 episode 索引，返回排序列表。可复现 (seed 控制) |
| `load_episodes_jsonl(path)` | 86 | 加载 `episodes.jsonl`，按 episode_index 排序返回 |
| `load_episodes_stats_jsonl(path)` | 91 | 加载 `episodes_stats.jsonl`，按 episode_index 排序返回 |
| `load_tasks_jsonl(path)` | 96 | 加载 `tasks.jsonl`，按 task_index 排序返回 |
| `write_parquet_with_image_features(df, path)` | 101 | 写入 parquet，为图像列标记 HuggingFace `Image()` 类型 |

#### 10.2.4 Phase 函数

**`phase01_sample_rename(input_dir, output_dir, sampled_indices)`** (line 113-272)

Phase 0+1 合并操作：采样 episodes 并同时重命名列，避免双重 I/O。

处理流程:
1. 加载源数据集元数据 (`info.json`, `episodes.jsonl`, `episodes_stats.jsonl`, `tasks.jsonl`)
2. 构建 task 重映射表 (只保留被采样 episode 引用的 task，重编号为连续 0-based)
3. 逐 episode 处理 parquet:
   - 重编号 `episode_index` (0-based 连续)
   - 重算全局 `index` (跨 episode 连续递增)
   - 重映射 `task_index`
   - 执行列重命名 (`COLUMN_RENAME_MAP`)
   - 写入带 Image 标记的 parquet
   - 更新 episode 元数据和统计
4. 写入更新后的 `info.json`, `tasks.jsonl`, `episodes.jsonl`, `episodes_stats.jsonl`

如果 `sampled_indices` 为 `None`，处理全部 episodes（仅重命名）。

**`phase2_convert_v21_to_v30(dataset_dir)`** (line 275-294)

调用 LeRobot 官方 `convert_dataset()` 进行 v2.1 → v3.0 格式升级。完成后自动清理 `*_old` 中间目录。

**`phase2_5_compute_quantiles(dataset_dir)`** (line 297-431) — **精确计算模式 (默认)**

从转换后的 v3.0 数据 parquet 直接精确计算分位数:
1. 识别数值特征 (`observation.state`, `action`) 和图像特征
2. 读取所有 parquet 文件，按 feature 拼接全部帧
3. 使用 `np.quantile()` 在 float64 精度下计算 q01/q10/q50/q90/q99
4. 更新 `stats.json` — 添加全局分位数
5. 更新 `meta/episodes/*.parquet` — 添加 per-episode 分位数列
6. 图像特征使用 min/max 线性插值近似

**`phase2_5_import_norm_stats(dataset_dir, norm_stats_path)`** (line 434-573) — **导入模式**

从已有的 norm_stats.json（如 OpenPI 预计算文件）导入 q01/q99:
1. 加载源文件，支持 OpenPI 嵌套格式 (`{"norm_stats": {...}}`) 和直接格式
2. 通过 `NORM_STATS_KEY_MAP` 映射 key
3. 导入 q01/q99 到 stats.json
4. 如果源文件缺少 q10/q50/q90，通过线性插值补充
5. 为图像和 metadata 特征添加默认分位数
6. 更新 episode metadata parquet

**`phase3_verify(dataset_dir)`** (line 576-626) — **基本验证**

执行 3 项基本检查:
1. `info.json` 版本 = v3.0，features 包含所有预期 key
2. `stats.json` 存在且包含 q01/q99 分位数
3. `LeRobotDataset` 成功加载，样本 key 和 shape 正确 (state/action = 23 维)

#### 10.2.5 `main()` 流程 (line 629-674)

```
解析参数 → 采样索引 → Phase 0+1 → Phase 2 → Phase 2.5 → Phase 3
                                                ↑
                                    有 --norm-stats-path?
                                    是 → import_norm_stats()
                                    否 → compute_quantiles()
```

#### 10.2.6 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | Path | **必填** | 输入数据集目录 (OpenPI v2.1 格式) |
| `--output` | Path | **必填** | 输出数据集目录 (LeRobot v3.0 格式) |
| `--sample-episodes` | int | None (全部) | 随机采样的 episode 数量 |
| `--seed` | int | 42 | 随机种子 (控制采样可复现性) |
| `--skip-v30` | flag | False | 跳过 Phase 2 (v2.1→v3.0)，同时跳过 Phase 2.5 |
| `--skip-verify` | flag | False | 跳过 Phase 3 验证 |
| `--norm-stats-path` | Path | None | 导入已有 norm stats 文件路径。不指定则精确计算 |

#### 10.2.7 使用示例

```bash
# 环境: 使用 LeRobot venv
export PYTHON=/mnt/r/Venv/lerobot-venv/bin/python
cd /home/Luogang/SRC/Robot/lerobot

# 示例 1: 全量转换测试集 (4 episodes, 精确计算分位数)
$PYTHON bt/pi05/alig/data/convert_r1pro_to_lerobot.py \
    --input /mnt/r/share/lkx/pi/data/r1_pro_test_data \
    --output bt/pi05/alig/data/r1_pro_test_data_v30

# 示例 2: 采样 10 episodes + 精确计算分位数
$PYTHON bt/pi05/alig/data/convert_r1pro_to_lerobot.py \
    --input /mnt/r/share/lkx/pi/data/r1_pro_data_convert_chassis \
    --output bt/pi05/alig/data/r1_pro_chassis_v30 \
    --sample-episodes 10

# 示例 3: 采样 10 episodes + 导入 OpenPI norm stats (推荐用于微调对齐)
$PYTHON bt/pi05/alig/data/convert_r1pro_to_lerobot.py \
    --input /mnt/r/share/lkx/pi/data/r1_pro_data_convert_chassis \
    --output bt/pi05/alig/data/r1_pro_chassis_v30 \
    --sample-episodes 10 \
    --norm-stats-path /mnt/r/share/lkx/pi/openpi/assets/pi05_r1pro_chassis/r1_pro_data_convert_chassis/norm_stats.json

# 示例 4: 全量转换训练集 (64 episodes, 精确计算, ~90GB 磁盘)
$PYTHON bt/pi05/alig/data/convert_r1pro_to_lerobot.py \
    --input /mnt/r/share/lkx/pi/data/r1_pro_data_convert_chassis \
    --output /path/to/r1_pro_chassis_full_v30
```

### 10.3 verify_pi05.py — Pi0.5 兼容性验证脚本

#### 10.3.1 功能概述

验证转换后的 LeRobot v3.0 数据集是否兼容 Pi0.5 训练和推理。包含 3 个递进的验证级别。

#### 10.3.2 辅助函数

**`bootstrap_lerobot_policies_package()`** (line 40-50)

避免执行 `lerobot/policies/__init__.py` 的全量导入（该文件会导入所有 policy 模块及其依赖）。通过创建空的 `lerobot.policies` package 模块，让后续 `from lerobot.policies.pi05.xxx import ...` 能正常工作而不触发全量导入。

#### 10.3.3 验证级别

**Level 1: `level1_data_loading(dataset_dir)`** (line 53-113) — 无需 GPU

验证内容:
1. `LeRobotDatasetMetadata` 加载: codebase_version = v3.0
2. `stats` 包含 `observation.state`/`action` 的 q01/q99
3. `LeRobotDataset` 成功加载，报告 episodes/frames 数量
4. 随机读取 5 个样本，验证 key 存在和 shape (state/action = 23 维)

返回: `{"dataset": ..., "ds_meta": ...}` 供后续 Level 使用。

**Level 2: `level2_preprocessor(dataset_dir, context)`** (line 116-195) — 无需 GPU

验证内容:
1. `dataset_to_policy_features()`: 正确分类 features
   - `observation.state` → `FeatureType.STATE`
   - `observation.images.*` → `FeatureType.VISUAL`
   - `action` → `FeatureType.ACTION`
2. `PI05Config` 创建: 使用 QUANTILES 归一化模式
3. `NormalizerProcessorStep`: 初始化 + 实际归一化一个 batch
4. 报告归一化后 state/action 的范围 (预期接近 [-1, 1])

返回: context 增加 `config`, `input_features`, `output_features`。

**Level 3: `level3_forward_pass(dataset_dir, context, pretrained_path, tokenizer_name, num_steps)`** (line 198-299) — **需要 GPU**

验证内容:
1. 加载 Tokenizer (PaLIGemma)
2. 加载 PI05Policy (从预训练权重)
3. 构建完整 preprocessor pipeline:
   - `RenameObservationsProcessorStep` → `AddBatchDimensionProcessorStep` → `NormalizerProcessorStep` → `Pi05PrepareStateTokenizerProcessorStep` → `TokenizerProcessorStep` → `DeviceProcessorStep`
4. 运行 num_steps 步 forward pass，验证 loss 有限

可在 CPU 上运行但会很慢。

#### 10.3.4 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset-dir` | Path | **必填** | 转换后的 v3.0 数据集目录 |
| `--run-forward-pass` | flag | False | 运行 Level 3 前向传播验证 |
| `--pretrained-path` | str | `lerobot/pi05_base` | Pi0.5 预训练模型路径 |
| `--tokenizer-name` | str | `google/paligemma-3b-pt-224` | Tokenizer 名称 |
| `--num-steps` | int | 2 | Level 3 前向传播步数 |

#### 10.3.5 使用示例

```bash
# Level 1+2 (无需 GPU)
$PYTHON bt/pi05/alig/data/verify_pi05.py \
    --dataset-dir bt/pi05/alig/data/r1_pro_test_data_v30

# Level 1+2+3 (需要 GPU)
$PYTHON bt/pi05/alig/data/verify_pi05.py \
    --dataset-dir bt/pi05/alig/data/r1_pro_chassis_v30 \
    --run-forward-pass \
    --pretrained-path lerobot/pi05_base \
    --num-steps 2
```

### 10.4 run_convert.sh — 一键转换 + 验证脚本

#### 10.4.1 功能概述

一键运行两个数据集的转换和验证，封装了 `convert_r1pro_to_lerobot.py` 和 `verify_pi05.py` 的调用。

#### 10.4.2 执行步骤

| Step | 操作 | 数据集 | 说明 |
|------|------|--------|------|
| 1 | `convert_r1pro_to_lerobot.py` | r1_pro_test_data | 4 episodes 全量，精确计算模式 |
| 2 | `convert_r1pro_to_lerobot.py` | r1_pro_data_convert_chassis | 采样 10 episodes，精确计算或导入模式 |
| 3 | `verify_pi05.py` | r1_pro_test_data_v30 | Level 1+2 验证 |
| 4 | `verify_pi05.py` | r1_pro_chassis_v30 | Level 1+2 验证 |

#### 10.4.3 命令行选项

| 选项 | 说明 |
|------|------|
| `--forward-pass` | 在 Step 3/4 中添加 `--run-forward-pass`，启用 Level 3 (需要 GPU) |
| `--norm-stats-path` | Step 2 使用导入模式，从 OpenPI 的 `norm_stats.json` 导入分位数 |

#### 10.4.4 关键路径配置 (脚本内定义)

| 变量 | 路径 | 说明 |
|------|------|------|
| `INPUT_TEST` | `/mnt/r/share/lkx/pi/data/r1_pro_test_data` | 测试集源数据 |
| `INPUT_CHASSIS` | `/mnt/r/share/lkx/pi/data/r1_pro_data_convert_chassis` | 训练集源数据 |
| `OUTPUT_TEST` | `$SCRIPT_DIR/r1_pro_test_data_v30` | 测试集输出目录 |
| `OUTPUT_CHASSIS` | `$SCRIPT_DIR/r1_pro_chassis_v30` | 训练集输出目录 |
| `OPENPI_NORM_STATS` | `/mnt/r/share/lkx/pi/openpi/assets/.../norm_stats.json` | OpenPI 预计算的 norm stats |

#### 10.4.5 使用示例

```bash
cd /home/Luogang/SRC/Robot/lerobot

# 默认模式: 两个数据集都精确计算分位数，Level 1+2 验证
bash bt/pi05/alig/data/run_convert.sh

# 导入模式: chassis 数据集使用 OpenPI norm stats
bash bt/pi05/alig/data/run_convert.sh --norm-stats-path

# 完整验证 (含 GPU forward pass)
bash bt/pi05/alig/data/run_convert.sh --forward-pass

# 导入模式 + 完整验证
bash bt/pi05/alig/data/run_convert.sh --norm-stats-path --forward-pass
```

### 10.5 Norm Stats 双模式详解

#### 10.5.1 模式选择指南

| 场景 | 推荐模式 | 命令 |
|------|---------|------|
| **微调 OpenPI 预训练模型** | 导入 | `--norm-stats-path /path/to/norm_stats.json` |
| **从头训练新模型** | 精确计算 | 不传 `--norm-stats-path` |
| **对齐验证** | 导入 | `--norm-stats-path ...` |
| **新数据集 (无已有 stats)** | 精确计算 | 不传 `--norm-stats-path` |
| **采样子集快速开发** | 导入 | `--norm-stats-path ...` (避免子集分位数不准) |
| **全量转换生产训练** | 精确计算 | 不传 `--norm-stats-path`，且不传 `--sample-episodes` |

#### 10.5.2 两种模式的技术差异

| 方面 | 精确计算模式 (默认) | 导入模式 (`--norm-stats-path`) |
|------|-------------------|-------------------------------|
| 算法 | `np.quantile()` float64 精确计算 | 直接复制 q01/q99 值 |
| 数据来源 | 转换后 parquet 中的实际数据 | 外部 norm_stats.json 文件 |
| 分位数 | q01, q10, q50, q90, q99 全部精确 | q01/q99 从文件导入，q10/q50/q90 线性插值补充 |
| 与 OpenPI 对齐 | 相同全量数据时 max_diff < 0.001 | max_diff = 0 (完全一致) |
| 适用场景 | 新数据集、独立训练 | 微调预训练模型、对齐验证 |

#### 10.5.3 OpenPI norm_stats.json 格式

导入模式期望的 JSON 结构 (支持两种):

```json
// 格式 1: OpenPI 嵌套格式 (自动解包 "norm_stats" 层)
{
    "norm_stats": {
        "state": {"mean": [...], "std": [...], "q01": [...], "q99": [...]},
        "actions": {"mean": [...], "std": [...], "q01": [...], "q99": [...]}
    }
}

// 格式 2: 直接格式
{
    "state": {"q01": [...], "q99": [...]},
    "actions": {"q01": [...], "q99": [...]}
}
```

当前可用的 OpenPI norm_stats.json 路径:
```
/mnt/r/share/lkx/pi/openpi/assets/pi05_r1pro_chassis/r1_pro_data_convert_chassis/norm_stats.json
```

### 10.6 测试方法

#### 10.6.1 完整测试矩阵

以下 6 个测试覆盖了所有主要功能路径:

| # | 测试名 | 验证内容 | 命令 |
|---|--------|---------|------|
| 1 | 默认转换 (test_data) | Phase 0+1 全量 + Phase 2 + Phase 2.5 精确计算 + Phase 3 | 见下方 |
| 2 | 默认转换 (chassis) | Phase 0+1 采样 + Phase 2 + Phase 2.5 精确计算 + Phase 3 | 见下方 |
| 3 | 导入转换 (chassis) | Phase 0+1 采样 + Phase 2 + Phase 2.5 导入 + Phase 3 | 见下方 |
| 4 | 导入对齐验证 | 验证导入后 q01/q99 与源文件完全一致 | 见下方 |
| 5 | Level 1+2 验证 (test_data) | 数据加载 + 预处理器兼容性 | 见下方 |
| 6 | Level 1+2 验证 (chassis) | 数据加载 + 预处理器兼容性 | 见下方 |

#### 10.6.2 测试 1: 默认转换 — 测试集

```bash
$PYTHON bt/pi05/alig/data/convert_r1pro_to_lerobot.py \
    --input /mnt/r/share/lkx/pi/data/r1_pro_test_data \
    --output bt/pi05/alig/data/r1_pro_test_data_v30
```

预期输出:
```
Phase 0+1: 采样 + 列名重命名
Phase 2: v2.1 → v3.0 格式升级
Phase 2.5: 计算分位数统计 (精确计算)
Phase 3: 验证数据集
  info.json 验证通过 (v3.0, features 正确)
  stats.json 验证通过 (含分位数)
  LeRobotDataset 加载成功
  state shape: torch.Size([23]), action shape: torch.Size([23])
Phase 3 验证通过
全部转换完成!
```

#### 10.6.3 测试 2: 默认转换 — 训练集采样

```bash
$PYTHON bt/pi05/alig/data/convert_r1pro_to_lerobot.py \
    --input /mnt/r/share/lkx/pi/data/r1_pro_data_convert_chassis \
    --output bt/pi05/alig/data/r1_pro_chassis_v30 \
    --sample-episodes 10
```

预期: 与测试 1 相同结构，10 episodes。

#### 10.6.4 测试 3: 导入模式转换

```bash
$PYTHON bt/pi05/alig/data/convert_r1pro_to_lerobot.py \
    --input /mnt/r/share/lkx/pi/data/r1_pro_data_convert_chassis \
    --output bt/pi05/alig/data/r1_pro_chassis_v30 \
    --sample-episodes 10 \
    --norm-stats-path /mnt/r/share/lkx/pi/openpi/assets/pi05_r1pro_chassis/r1_pro_data_convert_chassis/norm_stats.json
```

预期输出包含:
```
Phase 2.5: 导入已有 norm stats: ...norm_stats.json
  state.q01 → observation.state.q01 (23 dims)
  state.q99 → observation.state.q99 (23 dims)
  actions.q01 → action.q01 (23 dims)
  actions.q99 → action.q99 (23 dims)
  stats.json 更新完成 (导入分位数)
Phase 2.5 完成: 分位数统计已添加 (从 norm_stats.json 导入)
```

#### 10.6.5 测试 4: 导入对齐验证

在测试 3 完成后运行:

```bash
$PYTHON -c "
import json, numpy as np
with open('/mnt/r/share/lkx/pi/openpi/assets/pi05_r1pro_chassis/r1_pro_data_convert_chassis/norm_stats.json') as f:
    openpi = json.load(f)['norm_stats']
with open('bt/pi05/alig/data/r1_pro_chassis_v30/meta/stats.json') as f:
    lr = json.load(f)
for src_k, dst_k in [('state','observation.state'), ('actions','action')]:
    for q in ['q01','q99']:
        diff = np.abs(np.array(openpi[src_k][q]) - np.array(lr[dst_k][q])).max()
        status = 'PASS' if diff < 1e-10 else 'FAIL'
        print(f'{dst_k}.{q}: max_diff={diff:.2e} [{status}]')
"
```

预期: 全部 PASS，max_diff = 0.00e+00。

#### 10.6.6 测试 5-6: Level 1+2 验证

```bash
# 测试 5: 测试集
$PYTHON bt/pi05/alig/data/verify_pi05.py \
    --dataset-dir bt/pi05/alig/data/r1_pro_test_data_v30

# 测试 6: 训练集
$PYTHON bt/pi05/alig/data/verify_pi05.py \
    --dataset-dir bt/pi05/alig/data/r1_pro_chassis_v30
```

预期输出:
```
Level 1: 数据加载验证
  Metadata 加载成功
  Stats 验证通过 (含 q01/q99 分位数)
  Dataset 加载成功
  样本读取验证通过
Level 1 通过!
Level 2: 预处理器兼容性验证
  Feature 分类验证通过
  PI05Config 创建成功
  NormalizerProcessorStep 初始化成功
  样本归一化通过
    normalized state range: [-1.000, ~1.0]
    normalized action range: [-1.000, ~1.0]
Level 2 通过!
全部验证通过!
```

#### 10.6.7 一键测试 (使用 run_convert.sh)

```bash
# 精确计算模式: 运行测试 1+2+5+6
bash bt/pi05/alig/data/run_convert.sh

# 导入模式: 运行测试 1+3+5+6
bash bt/pi05/alig/data/run_convert.sh --norm-stats-path

# 含 GPU 验证
bash bt/pi05/alig/data/run_convert.sh --forward-pass
```

### 10.7 输出目录结构

转换完成后的 v3.0 数据集目录结构:

```
r1_pro_chassis_v30/
├── data/
│   └── chunk-000/
│       ├── file-000.parquet          # 数据帧 (含 observation.*, action, metadata)
│       ├── file-001.parquet
│       └── ...
├── videos/
│   ├── observation.images.head_rgb/
│   │   └── chunk-000/
│   │       ├── file-000.mp4
│   │       └── ...
│   ├── observation.images.left_wrist_rgb/
│   │   └── chunk-000/...
│   └── observation.images.right_wrist_rgb/
│       └── chunk-000/...
└── meta/
    ├── info.json                     # 数据集元信息 (v3.0, features, fps, splits)
    ├── stats.json                    # 全局统计 (含 q01/q10/q50/q90/q99)
    ├── tasks.jsonl                   # 任务描述
    ├── episodes.jsonl                # Episode 元数据
    └── episodes/
        └── chunk-000/
            └── file-000.parquet      # Per-episode 统计 (含分位数列)
```

### 10.8 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `ModuleNotFoundError: lerobot` | 未使用正确的 Python 环境 | 使用 `/mnt/r/Venv/lerobot-venv/bin/python` |
| `--norm-stats-path` 文件找不到 | 路径错误 | 确认路径: `/mnt/r/share/lkx/pi/openpi/assets/pi05_r1pro_chassis/r1_pro_data_convert_chassis/norm_stats.json` |
| 导入模式 q10/q50/q90 不精确 | OpenPI 只提供 q01/q99，中间分位数由线性插值补充 | 对训练无影响 (训练只用 q01/q99) |
| 采样 10 episodes 的分位数与 OpenPI 偏差大 | 数据范围不同 (10 vs 64 episodes) | 使用 `--norm-stats-path` 导入，或全量转换 |
| `ValueError: 未能从 ... 导入任何分位数` | norm_stats.json 格式不匹配 | 检查文件是否包含 `state`/`actions` 的 q01/q99 |
| Phase 2 报错 `BackwardCompatibilityError` | 直接加载 v2.1 数据 | 确保按顺序执行，Phase 0+1 先完成 |
| Level 3 OOM | GPU 显存不足 | 仅在有足够 GPU 显存时使用 `--run-forward-pass` |
| 磁盘空间不足 | 全量转换需 ~90GB | 用 `--sample-episodes 10` 先测试 |

### 10.9 与第 7 节 (使用指南 V1) 的差异

| 方面 | V1 (第 7 节) | V2 (本节) |
|------|-------------|-----------|
| Norm Stats | 仅精确计算模式 | 双模式: 精确计算 + 导入 |
| CLI 参数 | 无 `--norm-stats-path` | 新增 `--norm-stats-path` |
| 验证脚本说明 | 简略 | 完整的 3 级验证文档 |
| 一键脚本选项 | 仅 `--forward-pass` | 新增 `--norm-stats-path` |
| 测试矩阵 | 基本测试 | 6 个测试完整覆盖 |
| 函数文档 | 按 Phase 说明 | 按函数逐一说明 (含行号) |
